"""
test_features.py
----------------
Unit tests for src/features.py.

The feature engineering pipeline is the contract between raw data and the ML
model.  It's the most fragile part of the system — a one-line change here can
silently break model accuracy in ways that only surface weeks later when someone
notices the surge multiplier has been wrong.  We test it obsessively.

The main testing challenge here is the stochastic noise.  create_features()
calls np.random.default_rng(seed=None) to inject Gaussian noise into the
price_multiplier target variable.  Non-seeded means different output every run
— which is correct for production, but makes assertions impossible in tests.

Our solution: we patch numpy's default_rng at the module level using
unittest.mock.patch.  We inject a seeded RNG that returns a fixed, predictable
array of zeros for the noise.  Zero noise means the price_multiplier equals
the deterministic baseline exactly, which we can then assert against.

Note: we patch ``features.np.random.default_rng``, not ``numpy.random.default_rng``.
The difference matters.  features.py imports numpy as np, so the RNG call lives
in the features module's namespace.  Patching the numpy source wouldn't intercept
it — we have to patch the reference inside the module that uses it.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

import features
from features import (
    create_features,
    _deterministic_multiplier,
    _MULTIPLIER_MIN,
    _MULTIPLIER_MAX,
    _MULTIPLIER_NOISE_STD,
)


# ---------------------------------------------------------------------------
# Helper: a seeded mock RNG that returns predictable noise
# ---------------------------------------------------------------------------

def _make_zero_noise_rng():
    """
    Return a mock that looks like np.random.default_rng() but always produces
    an array of zeros for .normal().

    Why zeros?  Because then price_multiplier = deterministic_base + 0 = deterministic_base,
    which we can assert against.  Any non-zero seed would still produce
    variability that changes with numpy version upgrades, making tests brittle.
    Using zeros sidesteps that entirely.
    """
    rng = MagicMock()
    # .normal() is called with (loc=0.0, scale=..., size=N) — we return zeros
    rng.normal = MagicMock(side_effect=lambda loc, scale, size: np.zeros(size))
    return rng


# ---------------------------------------------------------------------------
# Tests: create_features() — structural correctness
# ---------------------------------------------------------------------------

class TestCreateFeaturesStructure:
    """Verify that create_features() produces the right columns and shapes."""

    def test_returns_dataframe(self, raw_ride_df):
        """The most basic check — should always return a DataFrame, never None."""
        result = create_features(raw_ride_df)
        assert isinstance(result, pd.DataFrame)

    def test_output_has_same_number_of_rows_as_input(self, raw_ride_df):
        """
        Feature engineering must be a row-preserving operation.
        We add columns, not rows. If this fails, something is doing an
        accidental group-by or join somewhere.
        """
        result = create_features(raw_ride_df)
        assert len(result) == len(raw_ride_df)

    def test_all_expected_feature_columns_are_present(self, raw_ride_df):
        """
        This is the contract test — if a feature is added or renamed in features.py,
        this test will catch whether it's still being produced.  It's also
        documentation: here's the exact set of columns the model depends on.
        """
        result = create_features(raw_ride_df)
        expected = {
            "demand_ratio", "hour", "day_of_week", "is_weekend",
            "time_slot", "is_rainy", "is_peak_hour", "is_city_center",
            "price_multiplier",
        }
        assert expected.issubset(set(result.columns))

    def test_does_not_mutate_input_dataframe(self, raw_ride_df):
        """
        create_features() starts with df.copy() — this test proves that promise.
        Mutating the caller's DataFrame would break Streamlit's caching in subtle
        ways that are very hard to reproduce in development.
        """
        original_cols = set(raw_ride_df.columns)
        create_features(raw_ride_df)
        assert set(raw_ride_df.columns) == original_cols

    def test_works_without_optional_event_column(self):
        """
        The 'event' column is optional — not all CSVs have it.  The function must
        not raise KeyError when it's absent.  This tests the .get("event", "None")
        path in _deterministic_multiplier().
        """
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-08 09:00"]),
            "requests": [100],
            "drivers": [10],
            "weather": ["Clear"],
            "location_name": ["City Center"],
            # Deliberately no 'event' column
        })
        result = create_features(df)
        assert "price_multiplier" in result.columns
        assert not result["price_multiplier"].isna().any()


# ---------------------------------------------------------------------------
# Tests: demand_ratio calculation
# ---------------------------------------------------------------------------

class TestDemandRatio:
    """
    demand_ratio is the most important feature in the model — it's consistently
    the top predictor in the RF feature importance chart.  Getting it wrong
    would be catastrophic.
    """

    def test_demand_ratio_is_requests_divided_by_drivers(self, raw_ride_df):
        """Basic arithmetic: 100 requests / 10 drivers = 10.0."""
        result = create_features(raw_ride_df)
        expected = raw_ride_df["requests"] / raw_ride_df["drivers"]
        # We can't assert exact equality because inf values may have been clipped,
        # but none of our fixture rows have zero drivers, so this should match.
        pd.testing.assert_series_equal(
            result["demand_ratio"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_infinite_demand_ratio_is_clipped_to_multiplier_max(self):
        """
        If drivers=0, the raw ratio is inf.  We must clip it to _MULTIPLIER_MAX
        (5.0) rather than letting inf propagate into the model, which would
        produce NaN predictions and silently break the entire inference pipeline.

        This was discovered in production when a data pipeline bug sent a batch
        with all-zero driver counts.  We added the clip as a result.
        """
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-08 09:00", "2024-01-08 09:00"]),
            "requests": [100, 50],
            "drivers": [0, 0],   # Division by zero — this is the bug we're guarding against
            "weather": ["Clear", "Clear"],
            "location_name": ["City Center", "Suburbs"],
        })
        result = create_features(df)
        assert not result["demand_ratio"].isin([np.inf, -np.inf]).any(), (
            "Infinite values survived into demand_ratio — the clipping logic is broken."
        )
        assert (result["demand_ratio"] <= _MULTIPLIER_MAX).all()

    def test_nan_demand_ratio_is_filled_with_one(self):
        """
        NaN can appear if requests is also NaN (bad data).  We fill NaN with 1.0
        (balanced supply) rather than propagating it into the model.
        """
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-08 09:00"]),
            "requests": [float("nan")],
            "drivers": [10],
            "weather": ["Clear"],
            "location_name": ["City Center"],
        })
        result = create_features(df)
        assert not result["demand_ratio"].isna().any()


# ---------------------------------------------------------------------------
# Tests: time-based features
# ---------------------------------------------------------------------------

class TestTimeFeatures:
    """
    Time features are derived from the timestamp column.  They need to be correct
    because is_peak_hour and is_weekend are two of the top-5 predictors.
    """

    def _make_df(self, timestamp: str, **kwargs) -> pd.DataFrame:
        """Helper to build a single-row DataFrame for time feature assertions."""
        base = {
            "requests": [100],
            "drivers": [10],
            "weather": ["Clear"],
            "location_name": ["City Center"],
        }
        base.update(kwargs)
        base["timestamp"] = pd.to_datetime([timestamp])
        return pd.DataFrame(base)

    def test_weekday_monday_is_not_weekend(self):
        """2024-01-08 is a Monday. is_weekend must be 0."""
        result = create_features(self._make_df("2024-01-08 12:00"))
        assert result["is_weekend"].iloc[0] == 0

    def test_saturday_is_weekend(self):
        """2024-01-06 is a Saturday. is_weekend must be 1."""
        result = create_features(self._make_df("2024-01-06 12:00"))
        assert result["is_weekend"].iloc[0] == 1

    def test_sunday_is_weekend(self):
        """2024-01-07 is a Sunday. is_weekend must be 1."""
        result = create_features(self._make_df("2024-01-07 12:00"))
        assert result["is_weekend"].iloc[0] == 1

    @pytest.mark.parametrize("hour,expected", [
        (8,  1),   # Start of morning peak
        (9,  1),   # Deep in morning peak
        (10, 1),   # End of morning peak
        (11, 0),   # Just after morning peak
        (17, 1),   # Start of evening peak
        (18, 1),   # Evening peak
        (20, 1),   # Last hour of evening peak
        (21, 0),   # Just after evening peak
        (4,  0),   # Dead of night
        (14, 0),   # Afternoon — not a peak window
    ])
    def test_is_peak_hour_boundaries(self, hour, expected):
        """
        The peak hour boundaries (08-10, 17-20) are business-critical.  A one-off
        error on the boundary would silently miscategorise thousands of rides.
        We test every boundary hour explicitly rather than just the 'obvious' cases.
        """
        ts = f"2024-01-08 {hour:02d}:00"
        result = create_features(self._make_df(ts))
        assert result["is_peak_hour"].iloc[0] == expected, (
            f"Hour {hour} should give is_peak_hour={expected}"
        )

    @pytest.mark.parametrize("hour,expected_slot", [
        (4,  "Night"),
        (5,  "Morning"),
        (11, "Morning"),
        (12, "Afternoon"),
        (16, "Afternoon"),
        (17, "Evening"),
        (20, "Evening"),
        (21, "Night"),
        (23, "Night"),
    ])
    def test_time_slot_boundaries(self, hour, expected_slot):
        """
        time_slot boundaries are used in the analytics charts.  Wrong bucketing
        here doesn't affect model accuracy (the RF uses is_peak_hour, not time_slot)
        but it would produce misleading dashboard numbers.
        """
        ts = f"2024-01-08 {hour:02d}:00"
        result = create_features(self._make_df(ts))
        assert result["time_slot"].iloc[0] == expected_slot, (
            f"Hour {hour} should map to '{expected_slot}'"
        )


# ---------------------------------------------------------------------------
# Tests: condition flags (weather, location)
# ---------------------------------------------------------------------------

class TestConditionFlags:
    """Tests for the binary feature flags derived from categorical columns."""

    def test_rainy_weather_sets_is_rainy_to_one(self, raw_ride_df):
        """Row 0 in our fixture has weather='Rainy'. is_rainy must be 1."""
        result = create_features(raw_ride_df)
        rainy_rows = raw_ride_df[raw_ride_df["weather"] == "Rainy"].index
        assert (result.loc[rainy_rows, "is_rainy"] == 1).all()

    def test_non_rainy_weather_sets_is_rainy_to_zero(self, raw_ride_df):
        """Clear and Foggy weather should both produce is_rainy=0."""
        result = create_features(raw_ride_df)
        non_rainy_rows = raw_ride_df[raw_ride_df["weather"] != "Rainy"].index
        assert (result.loc[non_rainy_rows, "is_rainy"] == 0).all()

    def test_city_center_location_sets_flag(self, raw_ride_df):
        """Rows where location_name='City Center' must get is_city_center=1."""
        result = create_features(raw_ride_df)
        city_center_rows = raw_ride_df[raw_ride_df["location_name"] == "City Center"].index
        assert (result.loc[city_center_rows, "is_city_center"] == 1).all()

    def test_non_city_center_location_clears_flag(self, raw_ride_df):
        """All other locations must get is_city_center=0."""
        result = create_features(raw_ride_df)
        other_rows = raw_ride_df[raw_ride_df["location_name"] != "City Center"].index
        assert (result.loc[other_rows, "is_city_center"] == 0).all()


# ---------------------------------------------------------------------------
# Tests: target variable (price_multiplier)
# ---------------------------------------------------------------------------

class TestPriceMultiplier:
    """
    Tests for the stochastic price_multiplier target variable.

    We split these into two concerns:
    1. The deterministic business rules (_deterministic_multiplier) — tested
       directly by calling the function with hand-crafted inputs.
    2. The full pipeline output — tested with a zero-noise mock so we can
       assert exact values without being at the mercy of random seeds.
    """

    def test_multiplier_is_always_within_bounds(self, raw_ride_df):
        """
        No matter what noise is added, the multiplier must always stay in [1.0, 5.0].
        This is the core safety guarantee of the clip() call.
        """
        result = create_features(raw_ride_df)
        assert (result["price_multiplier"] >= _MULTIPLIER_MIN).all()
        assert (result["price_multiplier"] <= _MULTIPLIER_MAX).all()

    def test_multiplier_has_non_zero_variance(self, raw_ride_df):
        """
        If all multipliers are the same value, the model is learning a constant —
        not a useful pricing signal.  A healthy variance tells us the noise is
        working AND that the deterministic baseline varies across rows.
        """
        result = create_features(raw_ride_df)
        assert result["price_multiplier"].std() > 0.0, (
            "All multiplier values are identical — something is very wrong with "
            "the deterministic baseline or the noise injection."
        )

    def test_deterministic_multiplier_base_case(self):
        """
        At demand_ratio=1.0, no rain, no event, the multiplier should be 1.0.
        The pricing team calls this 'flat fare' — no surge, no discount.
        """
        row = pd.Series({
            "demand_ratio": 1.0,
            "is_rainy": 0,
            "event": "None",
        })
        assert _deterministic_multiplier(row) == pytest.approx(1.0)

    def test_deterministic_multiplier_high_demand(self):
        """
        At demand_ratio=2.5, the surge formula is: 1.0 + (2.5 - 1.5) * 0.5 = 1.5.
        The slope of 0.5 above the 1.5 threshold was chosen to be conservative —
        this test locks in that business rule.
        """
        row = pd.Series({
            "demand_ratio": 2.5,
            "is_rainy": 0,
            "event": "None",
        })
        assert _deterministic_multiplier(row) == pytest.approx(1.5)

    def test_deterministic_multiplier_rain_premium(self):
        """
        Rain should add exactly +0.20 to the base multiplier.
        This is a business rule, not a model parameter, so we test it directly.
        """
        row = pd.Series({
            "demand_ratio": 1.0,
            "is_rainy": 1,
            "event": "None",
        })
        assert _deterministic_multiplier(row) == pytest.approx(1.20)

    def test_deterministic_multiplier_event_premium(self):
        """Events add +0.30. Combined with rain at low demand: 1.0 + 0.20 + 0.30 = 1.50."""
        row = pd.Series({
            "demand_ratio": 1.0,
            "is_rainy": 1,
            "event": "Concert",
        })
        assert _deterministic_multiplier(row) == pytest.approx(1.50)

    def test_deterministic_multiplier_no_event_key_at_all(self):
        """
        When 'event' key is completely absent from the row (not just 'None'),
        the .get() default should kick in and we shouldn't get a KeyError.
        """
        row = pd.Series({
            "demand_ratio": 1.0,
            "is_rainy": 0,
            # No 'event' key at all
        })
        result = _deterministic_multiplier(row)
        assert result == pytest.approx(1.0)

    def test_price_multiplier_with_zero_noise(self, raw_ride_df):
        """
        The critical determinism test.

        We patch default_rng to return a mock that gives zero noise,
        then verify that the output price_multiplier matches the deterministic
        baseline exactly.  This proves the pipeline structure is correct even
        though we can't test the stochastic version exactly.

        If this test fails after a features.py change, it means the deterministic
        formula or the column computation order has changed — which requires a
        deliberate review, not an accidental fix.
        """
        # The mock RNG will return zeros — noise = 0 everywhere
        mock_rng = _make_zero_noise_rng()

        with patch("features.np.random.default_rng", return_value=mock_rng):
            result = create_features(raw_ride_df)

        # With zero noise, every multiplier should equal the deterministic baseline
        # computed row-by-row.  We check a specific known row:
        #
        # Row 0: requests=100, drivers=10 → demand_ratio = 100/10 = 10.0
        #   Note: create_features() only clips *inf* values (division-by-zero guard).
        #   A finite demand_ratio of 10.0 passes through unchanged to _deterministic_multiplier.
        #   is_rainy=1, event='None'
        #
        # _deterministic_multiplier(demand_ratio=10.0, is_rainy=1, event='None'):
        #   = 1.0 + (10.0 - 1.5) * 0.5 + 0.20
        #   = 1.0 + 4.25 + 0.20
        #   = 5.45
        #
        # np.clip(5.45 + 0_noise, 1.0, 5.0) = 5.0   ← the final clip in create_features()
        row0_multiplier = result["price_multiplier"].iloc[0]
        expected_row0 = 5.0  # 5.45 raw base, clipped to _MULTIPLIER_MAX
        assert row0_multiplier == pytest.approx(expected_row0, abs=0.001), (
            f"Expected {expected_row0:.3f} with zero noise (raw base was 5.45, clipped to 5.0), "
            f"got {row0_multiplier:.3f}. The deterministic formula or clipping logic may have changed."
        )
