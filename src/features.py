"""
features.py
-----------
Feature engineering pipeline for the Dynamic Pricing System.

This module is the bridge between raw operational data (ride requests, driver
counts, timestamps) and the numeric vectors our ML models can actually reason
about. It needs to be stable and boring — every change here has a direct,
sometimes non-obvious, knock-on effect on model accuracy, and "let's just
add one more feature" is a sentence that has bitten us before.

The target variable, ``price_multiplier``, is a simulation artefact: we don't
have actual historical "correct" prices, so we derive them from business rules
plus deliberate stochastic noise. That noise is the important part — see the
comment block above _deterministic_multiplier() for why.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Noise parameters for the target variable.
#
# The σ=0.05 value (±5% standard deviation) was chosen empirically. It's small
# enough that the model still learns the real signal (demand_ratio drives price)
# but large enough that the training loss surface isn't perfectly smooth. In
# other words: the RF can't just memorize "demand_ratio=1.8 → multiplier=1.15"
# because each time it sees that ratio, the target wobbles by a few cents.
#
# If you tighten this to 0.01 or less, run a quick check: the test RMSE will
# drop dramatically but the model will be brittle on real CSV data where the
# signal-to-noise ratio is much lower than our simulator implies.
# ---------------------------------------------------------------------------
_MULTIPLIER_NOISE_STD: float = 0.05
_MULTIPLIER_MIN: float = 1.0
_MULTIPLIER_MAX: float = 5.0


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw ride-request data into an ML-ready feature DataFrame.

    This function must be called on both training data and on live inference
    data — the exact same function, same column order, same logic. Any drift
    between training-time and inference-time transformations is the single most
    common source of silent model degradation in production. Keeping it in one
    place means you fix it once and it's fixed everywhere.

    Features produced
    -----------------
    demand_ratio   : Continuous. The core signal. requests / drivers. Watch the
                     denominator — if the driver count is 0 (goes offline), this
                     blows up to inf. We clip downstream, but log a warning here.
    hour           : 0–23 integer. Kept for the time_slot categorisation below.
    day_of_week    : 0=Monday to 6=Sunday. Weekends pattern differently.
    is_weekend     : Binary flag derived from day_of_week. Simpler for the RF
                     than the raw integer, which has no ordinal meaning (Thursday
                     is not "more" than Monday in any useful sense).
    time_slot      : String category. Not used directly by the RF (it'd need
                     encoding), but useful for downstream analytics and charts.
    is_rainy       : Binary. Rainy weather shifts both demand (up) and supply
                     (down, drivers avoid rain) simultaneously, making it one of
                     the strongest predictors in the feature set.
    is_peak_hour   : Binary. Covers 08:00–10:00 and 17:00–20:00 — classic
                     commute windows. Doesn't overlap perfectly with demand peaks
                     in practice (some cities peak later), but it's consistent
                     across our synthetic and real datasets, so we keep it.
    is_city_center : Binary. City Center rides have systematically higher base
                     fares and higher demand volatility. Worth flagging explicitly
                     rather than relying on the RF to infer it from coordinates.

    Target (training only)
    ----------------------
    price_multiplier : Derived from business rules + Gaussian noise. Clipped to
                       [1.0, 5.0]. The model is trained to predict this. During
                       live inference the column is generated but ignored — the
                       model's .predict() output is what actually drives pricing.

    Args:
        df: Raw DataFrame. Expected to contain at minimum: timestamp, requests,
            drivers, weather, location_name. The 'event' column is optional —
            many CSV datasets don't have it and we handle that gracefully.

    Returns:
        A copy of the input DataFrame with all feature and target columns
        appended. We copy() up front to avoid mutating the caller's data,
        which has caused subtle bugs when Streamlit's caching was involved.
    """
    df = df.copy()

    # ------------------------------------------------------------------
    # Demand Ratio — the heartbeat of the pricing signal.
    #
    # We calculate this first because everything else — the multiplier
    # formula, the LP optimizer's utilization input — depends on it.
    # Note: Keep an eye on the denominator. If drivers drop offline
    # unexpectedly, pandas will yield 'inf'. The downstream pipeline clips
    # demand_ratio to a sensible ceiling before feeding it to the model,
    # but we surface a warning here so ops can correlate it with driver
    # app crashes or GPS blackouts.
    # ------------------------------------------------------------------
    df["demand_ratio"] = df["requests"] / df["drivers"]

    if df["demand_ratio"].isin([np.inf, -np.inf]).any():
        logger.warning(
            "Infinite demand_ratio detected — %d row(s) have drivers=0. "
            "This usually means a data pipeline issue, not genuine supply collapse. "
            "The values will be clipped at %.1f before model inference.",
            df["demand_ratio"].isin([np.inf, -np.inf]).sum(),
            _MULTIPLIER_MAX,
        )
    # Clip inf/nan so they don't propagate into the model as NaN
    df["demand_ratio"] = df["demand_ratio"].replace([np.inf, -np.inf], _MULTIPLIER_MAX)
    df["demand_ratio"] = df["demand_ratio"].fillna(1.0)

    # ------------------------------------------------------------------
    # Time-based features.
    #
    # We extract hour and day_of_week from the timestamp and then derive
    # binary flags from those. The flags are what the model actually trains
    # on — raw integers like hour=18 imply a linear relationship to price
    # (hour 18 is "twice as much" as hour 9), which isn't true. Binary
    # flags let the RF split on them freely without that assumption.
    # ------------------------------------------------------------------
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # Saturday=5, Sunday=6. Weekends show higher leisure-trip demand and
    # lower driver availability (more drivers take time off). Both shift
    # the multiplier upward, which the model learns quickly.
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    def _time_slot(h: int) -> str:
        """
        Bucket hours into named time windows.

        These buckets loosely correspond to distinct rider behaviour modes:
        Morning = commute in, Afternoon = errands/lunch, Evening = commute out
        + social, Night = late-night/airport runs. The boundaries are slightly
        arbitrary but have stayed stable across city datasets, so we leave them.
        """
        if 5 <= h < 12:
            return "Morning"
        elif 12 <= h < 17:
            return "Afternoon"
        elif 17 <= h < 21:
            return "Evening"
        return "Night"

    df["time_slot"] = df["hour"].apply(_time_slot)

    # ------------------------------------------------------------------
    # Context flags — weather and location.
    #
    # is_rainy is a proxy for the combined demand-up / supply-down shock
    # that bad weather creates. It's a blunt instrument (Foggy weather also
    # affects supply but isn't captured), but rain is by far the dominant
    # weather signal in the data and keeping it binary keeps inference simple.
    # ------------------------------------------------------------------
    df["is_rainy"] = (df["weather"] == "Rainy").astype(int)

    df["is_peak_hour"] = df["hour"].apply(
        lambda h: 1 if (8 <= h <= 10 or 17 <= h <= 20) else 0
    )

    # City Center trips have a different economic profile: shorter distances,
    # higher base fares, wealthier riders with lower price sensitivity. The RF
    # will pick this up from the data, but flagging it explicitly makes the
    # feature importance chart more interpretable for stakeholders.
    df["is_city_center"] = (df["location_name"] == "City Center").astype(int)

    # ------------------------------------------------------------------
    # Target variable: stochastic price multiplier.
    #
    # The deterministic part encodes our best understanding of what the
    # "right" surge should be given the inputs. The Gaussian noise we add
    # on top is intentional friction — it prevents the model from learning
    # a perfect lookup table of our own business rules (which would make it
    # useless on real data that doesn't follow those rules exactly).
    #
    # Think of it this way: we're teaching the model "high demand_ratio
    # tends to mean higher price" rather than "demand_ratio=1.8 always
    # means price=1.15000". The former generalises; the latter doesn't.
    # ------------------------------------------------------------------
    base_multipliers = df.apply(_deterministic_multiplier, axis=1)

    # Non-seeded RNG intentionally — we want different noise every training
    # run. If you need reproducibility for a specific debugging session,
    # pass seed=42 here temporarily, but don't commit it. Seeded noise
    # defeats the whole point of the stochastic target design.
    rng = np.random.default_rng(seed=None)
    noise = rng.normal(loc=0.0, scale=_MULTIPLIER_NOISE_STD, size=len(df))

    df["price_multiplier"] = np.clip(
        base_multipliers + noise,
        _MULTIPLIER_MIN,
        _MULTIPLIER_MAX,
    ).round(4)

    logger.debug(
        "create_features(): %d rows processed. "
        "price_multiplier: μ=%.3f, σ=%.3f, min=%.3f, max=%.3f",
        len(df),
        df["price_multiplier"].mean(),
        df["price_multiplier"].std(),
        df["price_multiplier"].min(),
        df["price_multiplier"].max(),
    )

    return df


def _deterministic_multiplier(row: pd.Series) -> float:
    """
    Calculate the "ground truth" surge multiplier for a single ride record,
    before noise is added.

    This encodes three business rules that the pricing team has validated:
    1. When demand exceeds supply by more than 50% (ratio > 1.5), start
       surging. The 0.5 slope is deliberately conservative — aggressive
       surging at low demand ratios destroys retention faster than it
       recovers revenue.
    2. Rain adds a flat +0.20 premium. This compensates for the supply shock
       (fewer drivers available) rather than exploiting rider desperation.
       The ethics of surge pricing in bad weather are genuinely contested;
       the 0.20 cap reflects that we want to balance availability with fairness.
    3. Events add +0.30 flat. Concerts and sports events create localised,
       predictable demand spikes that riders generally accept as normal.

    Returns:
        The raw deterministic multiplier before noise. Note: this is NOT
        clipped — clipping happens after noise addition in create_features().
        This function can theoretically return values above 5.0 if demand_ratio
        is very large, but the clip in the caller handles it correctly.
    """
    multiplier = 1.0

    # Only start surging once demand clearly outpaces supply. Below 1.5,
    # the network is in rough balance and a multiplier above 1.0 would
    # just be margin extraction, not supply-side incentive.
    if row["demand_ratio"] > 1.5:
        multiplier += (row["demand_ratio"] - 1.5) * 0.5

    if row["is_rainy"]:
        multiplier += 0.20

    # The 'event' column doesn't exist in all datasets (the sample CSV
    # doesn't include it). We use .get() with a default to avoid a KeyError
    # here rather than enforcing the column upstream, which would break
    # backwards compatibility with legacy CSVs.
    if row.get("event", "None") != "None":
        multiplier += 0.30

    return multiplier


if __name__ == "__main__":
    import os
    import sys

    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s – %(message)s")
    sys.path.insert(0, os.path.dirname(__file__))

    from generator import generate_synthetic_data

    df_raw = generate_synthetic_data(n_samples=20)
    df_feat = create_features(df_raw)
    print(
        df_feat[
            ["requests", "drivers", "demand_ratio", "is_rainy", "price_multiplier"]
        ].to_string()
    )
