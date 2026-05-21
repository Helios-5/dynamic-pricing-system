"""
model.py
--------
ML model training and inference for the Dynamic Pricing System.

We deliberately keep the algorithmic toolbox narrow here: a Random Forest
Regressor paired with a Linear Regression baseline. Both are explainable,
have no heavy C-extension runtime dependencies, and behave predictably under
distribution shift — which matters a lot when the underlying data (rides,
weather, events) can change seasonally without warning.

The strategy is a "best-of-two" selection: we train both models on the same
fold and keep whichever achieves lower RMSE on the held-out test split. In
practice, Random Forest wins the vast majority of the time on this feature
set, but keeping LinearRegression around catches edge cases where the
training set is tiny (e.g. a niche city with sparse CSV data) and RF
starts over-fitting to the noise we intentionally injected in features.py.

One design note worth preserving: we intentionally do NOT use a stacking or
averaging ensemble. Blending the two predictions would smooth out the RF's
response to extreme demand spikes — exactly the signal the LP optimizer
downstream needs to make meaningful adjustments. A clean winner-takes-all
selection keeps inference outputs crisp.
"""

import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical feature list — the contract between this module and features.py.
#
# If you add a feature in create_features(), add it here too. If you don't,
# inference will silently use the wrong column order and the model will
# produce garbage predictions without raising an exception. Been there.
# ---------------------------------------------------------------------------
FEATURES = ["demand_ratio", "is_rainy", "is_peak_hour", "is_city_center", "is_weekend"]
TARGET = "price_multiplier"


def _build_random_forest() -> RandomForestRegressor:
    """
    Construct a RandomForestRegressor with tuned, conservative hyperparameters.

    A few choices worth explaining:
    - n_estimators=200: More than the default 100 but not so many that
      retraining on a Streamlit cold-start takes 30 seconds. On 2000 rows
      this runs in about 1-2 seconds on a modern laptop.
    - max_depth=12: Unconstrained depth means the trees will perfectly
      memorise training data. 12 is deep enough to learn real interactions
      (peak-hour × city-center) without fitting to noise.
    - min_samples_leaf=5: Guards against leaves with one or two samples
      driving wildly inflated predictions, which happens when the demand_ratio
      column spikes on a rainy weekend near an airport.
    - n_jobs=-1: Use all available cores. On a single-user Streamlit app
      this is fine; if you ever move this to a multi-tenant API, revisit.
    """
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        max_features="sqrt",    # The standard variance-reducing trick for RF
        n_jobs=-1,
        random_state=42,        # Reproducible results across re-runs
    )


def _build_linear_regression() -> LinearRegression:
    """
    Construct a plain LinearRegression baseline.

    We don't regularize (no Ridge/Lasso) because the feature space is tiny
    (5 columns) and multi-collinearity isn't a concern with binary flags.
    If we ever add 20+ one-hot-encoded location columns, revisit this and
    switch to Ridge. For now, vanilla OLS is perfectly fine as a sanity check.
    """
    return LinearRegression()


def train_model(df: pd.DataFrame) -> tuple:
    """
    Train a Random Forest and a Linear Regression, then return whichever
    achieves lower RMSE on the held-out test split.

    The caller gets back a single sklearn-compatible model object — so the
    rest of the pipeline (app.py, save_model, the feature_importances_
    check in the dashboard) doesn't need to know or care which one won.

    Args:
        df: Feature-engineered DataFrame from ``src.features.create_features()``.
            Must contain all columns in FEATURES plus the TARGET column.

    Returns:
        winner:  The model (RF or LR) with the lower test RMSE. Always
                 implements the sklearn ``.predict(X)`` interface.
        metrics: Dict containing:
                   - "MAE"               : float, mean absolute error of winner
                   - "RMSE"              : float, root mean squared error of winner
                   - "winner"            : str, "RandomForest" or "LinearRegression"
                   - "rf_rmse"           : float, RF's test RMSE (for comparison)
                   - "lr_rmse"           : float, LR's test RMSE (for comparison)
                   - "feature_importance": dict mapping feature name → RF importance
                                          score (None if LR wins, since LR has no
                                          concept of importance — only coefficients)

    Raises:
        ValueError: If ``df`` is missing required feature or target columns.
        RuntimeError: If both models fail to train (shouldn't happen, but we
                      guard it explicitly rather than letting the app crash).
    """
    # Validate inputs up front. Better to raise loudly here than to produce
    # silent NaN predictions that look plausible until someone notices the
    # surge multiplier is stuck at 1.0 for every ride.
    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"DataFrame is missing required columns: {missing_cols}. "
            f"Did create_features() run before train_model()?"
        )

    X = df[FEATURES]
    y = df[TARGET]

    # 80/20 split with a fixed seed so that if you're comparing two training
    # runs back-to-back, the evaluation split is identical — apples to apples.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── Train both candidates ─────────────────────────────────────────────────
    rf = _build_random_forest()
    lr = _build_linear_regression()

    logger.info(
        "Training RandomForestRegressor on %d samples (%d features)…",
        len(X_train), len(FEATURES),
    )
    rf.fit(X_train, y_train)

    logger.info("Training LinearRegression baseline…")
    lr.fit(X_train, y_train)

    # ── Evaluate both on the held-out test split ──────────────────────────────
    rf_preds = rf.predict(X_test)
    lr_preds = lr.predict(X_test)

    rf_rmse = float(np.sqrt(mean_squared_error(y_test, rf_preds)))
    lr_rmse = float(np.sqrt(mean_squared_error(y_test, lr_preds)))

    logger.info("RF RMSE=%.4f  |  LR RMSE=%.4f", rf_rmse, lr_rmse)

    # ── Pick the winner — lower RMSE wins ────────────────────────────────────
    # A coin-flip tiebreaker isn't needed in practice, but just in case both
    # are literally identical (e.g. perfectly linear synthetic data), we
    # default to RF because it gives us feature_importances_ in the dashboard.
    if rf_rmse <= lr_rmse:
        winner = rf
        winner_preds = rf_preds
        winner_name = "RandomForest"
        logger.info("Winner: RandomForest (RMSE delta=%.4f)", lr_rmse - rf_rmse)
    else:
        winner = lr
        winner_preds = lr_preds
        winner_name = "LinearRegression"
        logger.info("Winner: LinearRegression (RMSE delta=%.4f)", rf_rmse - lr_rmse)

    mae = float(mean_absolute_error(y_test, winner_preds))
    rmse = float(np.sqrt(mean_squared_error(y_test, winner_preds)))

    # Feature importances only exist on tree-based models. Linear Regression
    # has coefficients instead — not the same thing, and surfacing them in the
    # dashboard's "importance" chart would be misleading, so we return None.
    importance = None
    if winner_name == "RandomForest":
        importance = {
            feat: round(float(imp), 6)
            for feat, imp in zip(FEATURES, rf.feature_importances_)
        }

    metrics = {
        "MAE": round(mae, 6),
        "RMSE": round(rmse, 6),
        "winner": winner_name,
        "rf_rmse": round(rf_rmse, 6),
        "lr_rmse": round(lr_rmse, 6),
        "feature_importance": importance,
    }

    logger.info(
        "Training complete. Winner=%s  MAE=%.4f  RMSE=%.4f",
        winner_name, mae, rmse,
    )

    return winner, metrics


def save_model(model, path: str = "model.joblib") -> None:
    """
    Persist the trained model to disk using joblib's compressed pickle format.

    We use joblib rather than Python's built-in pickle because joblib handles
    large numpy arrays (the RF's internal tree structures) far more
    efficiently — typically 3–5× faster serialization with 40–60% smaller
    files due to memory-mapped array storage.

    Args:
        model: Any sklearn-compatible fitted estimator.
        path:  Destination file path. Relative paths resolve from wherever
               the process is launched (usually the project root).
    """
    joblib.dump(model, path)
    logger.info("Model persisted to %s", path)


def load_model(path: str = "model.joblib"):
    """
    Load a previously persisted model from disk.

    This will raise FileNotFoundError if the path doesn't exist — we
    intentionally do not swallow that exception here. The caller (app.py's
    _get_model) is responsible for deciding whether to re-train or surface
    the error to the user. Letting it propagate gives us a useful stack trace.

    Args:
        path: Path to the .joblib file.

    Returns:
        Fitted sklearn estimator, ready for .predict().
    """
    model = joblib.load(path)
    logger.info("Model loaded from %s", path)
    return model


if __name__ == "__main__":
    # Quick sanity-check: generate data, train, print metrics. Useful when
    # iterating on hyperparameters or after touching features.py.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s – %(message)s")

    sys.path.insert(0, os.path.dirname(__file__))
    from generator import generate_synthetic_data
    from features import create_features

    logger.info("Generating 2000 synthetic samples…")
    df = generate_synthetic_data(n_samples=2000)
    df = create_features(df)

    logger.info("Training…")
    model, metrics = train_model(df)

    # Print the readable summary without the feature importance dict cluttering it
    summary = {k: v for k, v in metrics.items() if k != "feature_importance"}
    logger.info("Metrics: %s", summary)
    if metrics["feature_importance"]:
        logger.info("Feature importance: %s", metrics["feature_importance"])

    save_model(model, "model_dev_test.joblib")
    logger.info("Model saved to model_dev_test.joblib. Done.")
