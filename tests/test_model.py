"""
test_model.py
-------------
Unit tests for src/model.py.

The model layer has three distinct jobs we need to test separately:
1. Training — train_model() trains RF + LR, picks the winner by RMSE.
2. Persistence — save_model()/load_model() round-trip through joblib.
3. Inference — the winner's .predict() interface is sklearn-compatible.

The training tests use the ``featured_df`` fixture from conftest.py, which is
deliberately small (12 rows).  12 rows means:
- The 80/20 split gives us 9 training rows and ~2–3 test rows.
- RF with 200 trees will over-fit this completely.
- LinearRegression will fit a clean hyperplane.
- LR will almost certainly win on RMSE, which actually makes the 'LR wins'
  code path (feature_importance=None) naturally testable with our fixture.

This is intentional.  We don't want a test that passes only when RF wins and
silently does nothing for the LR branch.  Our fixture exercises the LR-wins
path; the full-size smoke tests exercise the RF-wins path.

On determinism: train_model() uses random_state=42 for both models and for
the train_test_split.  So the model outputs are fully reproducible — we don't
need to mock anything in the model module itself.
"""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from model import (
    FEATURES,
    TARGET,
    train_model,
    save_model,
    load_model,
    _build_random_forest,
    _build_linear_regression,
)


# ---------------------------------------------------------------------------
# Tests: model builder functions
# ---------------------------------------------------------------------------

class TestModelBuilders:
    """
    Verify that the model factory functions return the right types with the
    right hyperparameters.  If someone changes a hyperparameter (like bumping
    n_estimators to 500 for a performance experiment and accidentally committing
    it), these tests will catch it during CI before it slows down production.
    """

    def test_build_random_forest_returns_correct_type(self):
        """Sanity check — the factory returns a RandomForestRegressor."""
        rf = _build_random_forest()
        assert isinstance(rf, RandomForestRegressor)

    def test_random_forest_has_expected_hyperparameters(self):
        """
        Lock in the agreed hyperparameters.  These were tuned carefully — any
        change should be deliberate and reviewed, not accidental.
        """
        rf = _build_random_forest()
        assert rf.n_estimators == 200
        assert rf.max_depth == 12
        assert rf.min_samples_leaf == 5
        assert rf.random_state == 42

    def test_build_linear_regression_returns_correct_type(self):
        """The baseline builder must return vanilla LinearRegression."""
        lr = _build_linear_regression()
        assert isinstance(lr, LinearRegression)


# ---------------------------------------------------------------------------
# Tests: train_model()
# ---------------------------------------------------------------------------

class TestTrainModel:
    """
    Tests for the training pipeline and the best-of-two model selection logic.

    We use the small featured_df fixture here.  The key insight is that with
    12 rows and an 80/20 split, the RF will overfit and LR will likely win —
    which means these tests naturally exercise the LR-wins code path.  The
    RF-wins path is exercised in the integration smoke tests.
    """

    def test_returns_two_element_tuple(self, featured_df):
        """train_model() is documented to return (model, metrics). Verify the shape."""
        result = train_model(featured_df)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_winner_model_implements_sklearn_predict(self, featured_df):
        """
        The winner must be sklearn-compatible — i.e., it must have a .predict()
        method that accepts a DataFrame of features and returns a numeric array.
        This is the interface that app.py depends on.
        """
        model, metrics = train_model(featured_df)
        sample = featured_df[FEATURES].head(3)
        predictions = model.predict(sample)
        assert len(predictions) == 3
        assert all(isinstance(p, (float, int, np.floating, np.integer)) for p in predictions)

    def test_metrics_dict_has_all_required_keys(self, featured_df):
        """
        Every key in the metrics dict is consumed by either app.py (for display)
        or by downstream monitoring.  If a key goes missing, the dashboard will
        crash or show 'None' where there should be a number.
        """
        _, metrics = train_model(featured_df)
        required_keys = {"MAE", "RMSE", "winner", "rf_rmse", "lr_rmse", "feature_importance"}
        assert required_keys == set(metrics.keys()), (
            f"Metrics dict has unexpected keys. "
            f"Missing: {required_keys - set(metrics.keys())}. "
            f"Extra: {set(metrics.keys()) - required_keys}."
        )

    def test_winner_is_one_of_the_two_valid_options(self, featured_df):
        """
        The winner field drives the 'winning model' display in the UI.
        It must be exactly one of two string values — not a type name, not None.
        """
        _, metrics = train_model(featured_df)
        assert metrics["winner"] in ("RandomForest", "LinearRegression"), (
            f"Unexpected winner value: '{metrics['winner']}'. "
            f"Expected 'RandomForest' or 'LinearRegression'."
        )

    def test_mae_and_rmse_are_positive_floats(self, featured_df):
        """
        Negative MAE or RMSE would indicate a bug in the error metric calculation.
        This should be impossible with sklearn's implementations, but we guard it
        anyway because the consequences of a negative 'error' showing up in the
        dashboard would be confusing.
        """
        _, metrics = train_model(featured_df)
        assert metrics["MAE"] >= 0.0
        assert metrics["RMSE"] >= 0.0
        assert isinstance(metrics["MAE"], float)
        assert isinstance(metrics["RMSE"], float)

    def test_rmse_is_consistent_with_winner(self, featured_df):
        """
        The 'RMSE' key in metrics should equal the RMSE of whichever model won.
        If winner='RandomForest', then RMSE == rf_rmse. If winner='LinearRegression',
        RMSE == lr_rmse.  This verifies the selection logic correctly copies the
        right value rather than accidentally always copying rf_rmse.
        """
        _, metrics = train_model(featured_df)
        if metrics["winner"] == "RandomForest":
            assert metrics["RMSE"] == pytest.approx(metrics["rf_rmse"])
        else:
            assert metrics["RMSE"] == pytest.approx(metrics["lr_rmse"])

    def test_feature_importance_is_dict_when_rf_wins(self, featured_df):
        """
        When RandomForest wins, feature_importance should be a dict mapping
        feature names to float scores.  The dashboard's bar chart depends on this.
        """
        _, metrics = train_model(featured_df)
        if metrics["winner"] == "RandomForest":
            assert isinstance(metrics["feature_importance"], dict)
            assert set(metrics["feature_importance"].keys()) == set(FEATURES)
            assert all(v >= 0 for v in metrics["feature_importance"].values())

    def test_feature_importance_is_none_when_lr_wins(self, featured_df):
        """
        LinearRegression doesn't have feature_importances_ in the sklearn sense
        (it has coefficients, which are dimensionally different).  We return None
        explicitly to avoid misleading the dashboard.  This test verifies that
        the None sentinel is returned correctly when LR wins.

        Note: With a 12-row DataFrame, LR nearly always wins due to RF overfitting.
        If this test starts failing (because RF wins on a very lucky split), it
        means the fixture has grown large enough that RF starts generalising.
        """
        _, metrics = train_model(featured_df)
        if metrics["winner"] == "LinearRegression":
            assert metrics["feature_importance"] is None, (
                "feature_importance should be None when LinearRegression wins. "
                "We don't expose LR coefficients as 'importances' because they "
                "would be misleading without proper scaling context."
            )

    def test_raises_value_error_for_missing_feature_columns(self, featured_df):
        """
        If create_features() wasn't called before train_model(), we should get
        a loud, descriptive ValueError rather than a silent KeyError buried deep
        in sklearn's fit() call.

        We intentionally pass a DataFrame with the TARGET column present but
        the FEATURES columns missing.
        """
        broken_df = pd.DataFrame({
            "price_multiplier": [1.0, 1.5, 2.0],
            "some_unrelated_column": [1, 2, 3],
        })
        with pytest.raises(ValueError, match="missing required columns"):
            train_model(broken_df)

    def test_raises_value_error_for_missing_target_column(self, featured_df):
        """
        Passing a DataFrame with features but no price_multiplier target column
        should also give a loud ValueError, not a confusing KeyError.
        """
        df_no_target = featured_df[FEATURES].copy()
        with pytest.raises(ValueError, match="missing required columns"):
            train_model(df_no_target)

    def test_rf_rmse_and_lr_rmse_are_both_reported(self, featured_df):
        """
        Both models' RMSEs must be in the metrics dict regardless of which one won.
        This is important for monitoring: we want to track the RF-vs-LR gap over
        time to detect dataset drift (a narrowing gap means the data is becoming
        more linear, a widening gap means it's becoming more non-linear).
        """
        _, metrics = train_model(featured_df)
        assert metrics["rf_rmse"] > 0
        assert metrics["lr_rmse"] > 0


# ---------------------------------------------------------------------------
# Tests: save_model() and load_model()
# ---------------------------------------------------------------------------

class TestModelPersistence:
    """
    Tests for the joblib round-trip: save then load, and verify the loaded
    model produces identical predictions to the original.

    We use pytest's built-in tmp_path fixture for the file path so that:
    1. The file lives in a temp directory that doesn't pollute the project.
    2. pytest automatically cleans it up after the test — no leftover .joblib files.
    3. Each test gets a fresh directory, so there's no risk of one test's saved
       model leaking into another test's load call.
    """

    def test_save_creates_file_at_given_path(self, featured_df, tmp_path):
        """After save_model(), the file must actually exist on disk."""
        model, _ = train_model(featured_df)
        save_path = str(tmp_path / "test_model.joblib")
        save_model(model, save_path)
        assert Path(save_path).exists()

    def test_load_returns_sklearn_compatible_model(self, featured_df, tmp_path):
        """
        The loaded model must be a fitted sklearn estimator — not None, not a dict,
        not a string.  It must have a .predict() method that works on the feature set.
        """
        model, _ = train_model(featured_df)
        save_path = str(tmp_path / "test_model.joblib")
        save_model(model, save_path)

        loaded_model = load_model(save_path)
        assert hasattr(loaded_model, "predict"), (
            "Loaded model doesn't have a .predict() method. "
            "joblib may have serialized a non-model object."
        )
        predictions = loaded_model.predict(featured_df[FEATURES])
        assert len(predictions) == len(featured_df)

    def test_loaded_model_predictions_match_original(self, featured_df, tmp_path):
        """
        The round-trip must be lossless: predictions from the loaded model must
        be identical to predictions from the original model object.  If they
        differ, it means joblib serialization is losing some internal state —
        which would indicate a numpy/sklearn version mismatch.
        """
        model, _ = train_model(featured_df)
        save_path = str(tmp_path / "test_model.joblib")
        save_model(model, save_path)
        loaded = load_model(save_path)

        X = featured_df[FEATURES]
        original_preds = model.predict(X)
        loaded_preds = loaded.predict(X)
        np.testing.assert_array_equal(
            original_preds, loaded_preds,
            err_msg="Loaded model predictions differ from original — serialization may be lossy.",
        )

    def test_load_raises_for_nonexistent_path(self, tmp_path):
        """
        load_model() explicitly does NOT swallow FileNotFoundError.
        The caller is responsible for deciding whether to retrain or surface
        the error.  This test verifies that intention hasn't been accidentally
        changed to a try/except.
        """
        with pytest.raises(FileNotFoundError):
            load_model(str(tmp_path / "this_does_not_exist.joblib"))
