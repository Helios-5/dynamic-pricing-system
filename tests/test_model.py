"""
test_model.py
-------------
Unit tests for src/model.py.

We are testing the core Random Forest training pipeline and ensuring that
the model can be successfully serialized (saved) and deserialized (loaded)
without losing its predictive state.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from model import train_model, save_model, load_model

# We define the features here for testing purposes, matching what is inside train_model()
FEATURES = ['demand_ratio', 'is_rainy', 'is_peak_hour', 'is_city_center', 'is_weekend']

class TestTrainModel:
    """
    Tests for the training pipeline and metric generation.
    """

    def test_returns_tuple(self, featured_df):
        """train_model() should return (model, metrics)."""
        result = train_model(featured_df)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_model_is_sklearn_compatible(self, featured_df):
        """
        The returned model must be sklearn-compatible — i.e., it must have a .predict()
        method that accepts a DataFrame of features and returns a numeric array.
        """
        model, metrics = train_model(featured_df)
        sample = featured_df[FEATURES].head(3)
        predictions = model.predict(sample)

        assert len(predictions) == 3
        assert all(isinstance(p, (float, int, np.floating, np.integer)) for p in predictions)

    def test_metrics_dict_has_required_keys(self, featured_df):
        """
        The dashboard depends on MAE, RMSE, and feature_importance.
        If these keys are missing, the UI will crash.
        """
        _, metrics = train_model(featured_df)
        assert "MAE" in metrics
        assert "RMSE" in metrics
        assert "feature_importance" in metrics

        # Verify feature importance is a dictionary mapping features to scores
        assert isinstance(metrics["feature_importance"], dict)
        assert set(metrics["feature_importance"].keys()) == set(FEATURES)

    def test_raises_key_error_for_missing_feature_columns(self):
        """
        If we pass raw data that hasn't gone through create_features(),
        pandas should raise a KeyError because the columns are missing.
        """
        broken_df = pd.DataFrame({
            "price_multiplier": [1.0, 1.5, 2.0],
            "some_unrelated_column": [1, 2, 3],
        })
        with pytest.raises(KeyError):
            train_model(broken_df)


class TestModelPersistence:
    """
    Tests for the joblib round-trip: save then load, and verify the loaded
    model produces identical predictions to the original.
    """

    def test_save_creates_file_at_given_path(self, featured_df, tmp_path):
        """After save_model(), the file must actually exist on disk."""
        model, _ = train_model(featured_df)
        save_path = str(tmp_path / "test_model.joblib")

        save_model(model, save_path)
        assert Path(save_path).exists()

    def test_load_returns_sklearn_compatible_model(self, featured_df, tmp_path):
        """The loaded object must act like a model."""
        model, _ = train_model(featured_df)
        save_path = str(tmp_path / "test_model.joblib")
        save_model(model, save_path)

        loaded_model = load_model(save_path)
        assert hasattr(loaded_model, "predict")

    def test_loaded_model_predictions_match_original(self, featured_df, tmp_path):
        """
        The round-trip must be lossless: predictions from the loaded model must
        be identical to predictions from the original model object.
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
        """load_model() should surface a FileNotFoundError if the file is missing."""
        with pytest.raises(FileNotFoundError):
            load_model(str(tmp_path / "this_does_not_exist.joblib"))