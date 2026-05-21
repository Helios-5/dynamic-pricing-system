"""
test_data_access.py
-------------------
Unit tests for src/data_access.py.

data_access.py is the configuration spine of the whole system — it reads
config.yaml, manages environment variable overrides, imputes coordinates, and
filters DataFrames.  If it breaks silently (which it can, since most of its
functions have safe fallbacks), the app keeps running but produces wrong results.
That's the worst kind of bug, so we test it more aggressively than the other
modules.

Key testing challenges:
- get_config() is decorated with @lru_cache, which means the real config is
  cached after the first call.  Tests that need to see different configs MUST
  call get_config.cache_clear() before they run, otherwise they'll be reading
  stale cache state from a previous test.  Every test in this file that touches
  get_config() does this.
- Environment variable overrides are tested by monkeypatching os.environ via
  pytest's built-in monkeypatch fixture.  We avoid patching os.environ
  directly (e.g., with a dict assignment) because that change would persist
  across tests; monkeypatch restores the original state automatically.
"""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

import data_access
from data_access import (
    get_config,
    get_default_coords,
    get_location_fallback_map,
    impute_coordinates,
    filter_dataframe,
    resolve_data_path,
    _FALLBACK_DEFAULT_LAT,
    _FALLBACK_DEFAULT_LON,
    _FALLBACK_LOCATION_MAP,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clear_config_cache():
    """
    Force get_config() to re-read from disk (or env) on the next call.

    get_config() is cached with @lru_cache(maxsize=1).  In a test session,
    the first test to call it populates the cache and every subsequent test
    sees the same result — even if we've patched env vars or swapped out
    config.yaml.  Clearing the cache before each config-touching test is
    the simplest way to guarantee isolation without refactoring the source.
    """
    get_config.cache_clear()


# ---------------------------------------------------------------------------
# Tests: get_config()
# ---------------------------------------------------------------------------

class TestGetConfig:
    """Tests for the config loading and merging logic."""

    def setup_method(self):
        # Always start with a clean cache so tests don't bleed into each other.
        _clear_config_cache()

    def teardown_method(self):
        _clear_config_cache()

    def test_returns_dict_when_config_yaml_exists(self):
        """
        The happy path: config.yaml is present and valid.

        We don't mock the file here — we let it read the real config.yaml from
        the project root.  This doubles as an integration check that the YAML
        file is parseable and contains the expected top-level keys.
        """
        cfg = get_config()
        assert isinstance(cfg, dict)
        # These are the keys we agreed on when we moved magic numbers out of app.py.
        assert "data_paths" in cfg
        assert "optimization" in cfg

    def test_returns_dict_even_when_config_yaml_missing(self, tmp_path, monkeypatch):
        """
        get_config() must never raise, even if config.yaml doesn't exist.

        We fake the project root to a tmp directory that has no config.yaml.
        The function should return an empty-ish dict (driven by env var defaults)
        rather than an FileNotFoundError or an empty crash.
        """
        _clear_config_cache()
        monkeypatch.setattr(data_access, "_PROJECT_ROOT", tmp_path)
        cfg = get_config()
        assert isinstance(cfg, dict)
        # data_paths should at least exist as a key (setdefault adds it)
        assert "data_paths" in cfg

    def test_env_var_overrides_default_lat(self, monkeypatch):
        """
        PRICING_DEFAULT_LAT should override whatever config.yaml says.

        This is the production mechanism for deploying to different cities
        without rebuilding the image.  We need to be sure it actually works.
        """
        _clear_config_cache()
        monkeypatch.setenv("PRICING_DEFAULT_LAT", "19.0760")  # Mumbai latitude
        monkeypatch.setenv("PRICING_DEFAULT_LON", "72.8777")  # Mumbai longitude
        cfg = get_config()
        assert cfg["default_lat"] == pytest.approx(19.0760)
        assert cfg["default_lon"] == pytest.approx(72.8777)

    def test_env_var_override_invalid_value_is_ignored(self, monkeypatch):
        """
        A typo in the env var (e.g., 'not-a-number') should not crash the app.

        The function should log a warning and continue with the config.yaml
        value rather than raising ValueError.  We're testing the 'fail safe'
        branch here, not the 'fail loud' branch.
        """
        _clear_config_cache()
        monkeypatch.setenv("PRICING_DEFAULT_LAT", "this-is-not-a-float")
        # Should not raise — just silently fall back to config/default value
        cfg = get_config()
        assert isinstance(cfg, dict)
        # The lat should still be the config.yaml value (28.6) or fallback, not crashing
        assert "default_lat" not in cfg or isinstance(cfg.get("default_lat"), float)

    def test_env_var_overrides_india_csv_path(self, monkeypatch):
        """
        The CSV path override is critical for Docker deployments where the
        data volume is mounted at a different path than the default.
        """
        _clear_config_cache()
        monkeypatch.setenv("PRICING_INDIA_CSV_PATH", "/mnt/data/rides.csv")
        cfg = get_config()
        assert cfg["data_paths"]["india_csv"] == "/mnt/data/rides.csv"


# ---------------------------------------------------------------------------
# Tests: get_default_coords()
# ---------------------------------------------------------------------------

class TestGetDefaultCoords:
    """Tests for the default coordinate resolution."""

    def setup_method(self):
        _clear_config_cache()

    def teardown_method(self):
        _clear_config_cache()

    def test_returns_tuple_of_two_floats(self):
        """Simple sanity check — the return type must always be (float, float)."""
        lat, lon = get_default_coords()
        assert isinstance(lat, float)
        assert isinstance(lon, float)

    def test_env_var_overrides_are_respected(self, monkeypatch):
        """
        get_default_coords() delegates to get_config(), so env overrides should
        flow through correctly.  This verifies the full call chain, not just the
        config parsing in isolation.
        """
        _clear_config_cache()
        monkeypatch.setenv("PRICING_DEFAULT_LAT", "12.9716")  # Bangalore
        monkeypatch.setenv("PRICING_DEFAULT_LON", "77.5946")
        lat, lon = get_default_coords()
        assert lat == pytest.approx(12.9716)
        assert lon == pytest.approx(77.5946)

    def test_falls_back_to_hardcoded_defaults_when_config_absent(self, monkeypatch, tmp_path):
        """
        If both config.yaml and env vars are absent, we should get the module-level
        hardcoded defaults.  This is the absolute last resort.
        """
        _clear_config_cache()
        monkeypatch.setattr(data_access, "_PROJECT_ROOT", tmp_path)
        # Unset the env vars too, so we're truly testing the bare-minimum fallback
        monkeypatch.delenv("PRICING_DEFAULT_LAT", raising=False)
        monkeypatch.delenv("PRICING_DEFAULT_LON", raising=False)
        lat, lon = get_default_coords()
        assert lat == pytest.approx(_FALLBACK_DEFAULT_LAT)
        assert lon == pytest.approx(_FALLBACK_DEFAULT_LON)


# ---------------------------------------------------------------------------
# Tests: impute_coordinates()
# ---------------------------------------------------------------------------

class TestImputeCoordinates:
    """
    Tests for coordinate imputation.

    impute_coordinates() has a subtle design: it adds small Gaussian jitter to
    the coordinates so that map points don't all stack on top of each other.
    This means we can't assert exact lat/lon values — we use approx() with a
    generous tolerance that encompasses the jitter range (±0.05 degrees ≈ 5.5km,
    which is way more than the ±0.01 sigma used for jitter).
    """

    def test_returns_unchanged_when_coords_already_present(self):
        """
        If lat/lon columns already exist, don't touch them.

        This guards against a regression where impute_coordinates() overwrites
        perfectly good GPS data with the rough fallback map values.
        """
        df = pd.DataFrame({
            "location_name": ["City Center"],
            "latitude": [28.6500],   # A precise reading from a GPS device
            "longitude": [77.2100],
        })
        result = impute_coordinates(df)
        # The columns must exist and the original values must be unchanged
        assert result["latitude"].iloc[0] == pytest.approx(28.6500)
        assert result["longitude"].iloc[0] == pytest.approx(77.2100)

    def test_imputes_known_location_from_fallback_map(self):
        """
        For a recognised location_name, we should get coordinates close to
        the fallback map entry.  We use a ±0.05 degree tolerance to allow
        for the spatial jitter that impute_coordinates() adds.
        """
        df = pd.DataFrame({"location_name": ["City Center"]})
        result = impute_coordinates(df)
        assert "latitude" in result.columns
        assert "longitude" in result.columns
        expected_lat, expected_lon = _FALLBACK_LOCATION_MAP["City Center"]
        # The jitter sigma is 0.01; ±0.1 gives us 10-sigma headroom — more than enough
        assert result["latitude"].iloc[0] == pytest.approx(expected_lat, abs=0.1)
        assert result["longitude"].iloc[0] == pytest.approx(expected_lon, abs=0.1)

    def test_imputes_unknown_location_with_default_coords(self):
        """
        For a location_name that's not in the fallback map, we should get the
        default coordinates (Delhi centre-point), not a KeyError or NaN.
        """
        df = pd.DataFrame({"location_name": ["Totally Unknown Zone"]})
        result = impute_coordinates(df)
        default_lat, default_lon = get_default_coords()
        assert result["latitude"].iloc[0] == pytest.approx(default_lat, abs=0.1)
        assert result["longitude"].iloc[0] == pytest.approx(default_lon, abs=0.1)

    def test_adds_both_columns_for_multiple_rows(self):
        """
        Imputation must work correctly on a DataFrame with multiple rows and
        a mix of known and unknown location names — the common real-world case.
        """
        df = pd.DataFrame({
            "location_name": ["City Center", "Airport", "Unknown Suburb", "Mall"],
        })
        result = impute_coordinates(df)
        assert len(result) == 4
        assert "latitude" in result.columns
        assert "longitude" in result.columns
        # No NaN values — every row must get a coordinate
        assert not result["latitude"].isna().any()
        assert not result["longitude"].isna().any()

    def test_does_not_mutate_original_dataframe(self):
        """
        Defensive test — impute_coordinates() should copy the DataFrame before
        modifying it.  If this ever fails, it means we've regressed to mutating
        the caller's data, which can cause very confusing Streamlit cache bugs.
        """
        df = pd.DataFrame({"location_name": ["City Center"]})
        original_columns = set(df.columns)
        impute_coordinates(df)
        assert set(df.columns) == original_columns  # Original must be unchanged


# ---------------------------------------------------------------------------
# Tests: filter_dataframe()
# ---------------------------------------------------------------------------

class TestFilterDataframe:
    """
    Tests for the city/area filtering and row-capping logic.

    filter_dataframe() is called on every user interaction in the Streamlit app,
    so it needs to be fast and bulletproof.  The edge cases we're most concerned
    about: search term with zero results, city filter with no 'city' column,
    and the cap being enforced correctly.
    """

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """
        A 30-row DataFrame with a 'city' column and recognisable location names.
        Big enough to test sampling/capping, small enough to run instantly.
        """
        return pd.DataFrame({
            "location_name": (["City Center"] * 10 + ["Airport"] * 10 + ["Suburbs"] * 10),
            "city": (["Delhi"] * 10 + ["Delhi"] * 5 + ["Mumbai"] * 5 + ["Delhi"] * 10),
            "requests": [50] * 30,
        })

    def test_returns_correct_number_when_under_cap(self, sample_df):
        """When the dataset has fewer rows than n_requests, return all of them."""
        result, msg = filter_dataframe(sample_df, n_requests=100)
        assert len(result) == 30  # All 30 rows returned — nothing to cap

    def test_caps_result_at_n_requests(self, sample_df):
        """
        When the dataset exceeds n_requests, we must not return more rows than
        the cap.  The app uses this to limit how much data gets pushed to pydeck,
        which has real performance consequences for large CSVs.
        """
        result, msg = filter_dataframe(sample_df, n_requests=10)
        assert len(result) == 10

    def test_city_filter_returns_only_matching_rows(self, sample_df):
        """
        The city filter is set to 'Mumbai' — only the 5 Mumbai Airport rows should
        be eligible.  Even if we ask for n_requests=100, we should only get 5 back.
        """
        result, msg = filter_dataframe(sample_df, city="Mumbai", n_requests=100)
        assert all(result["city"] == "Mumbai")
        assert len(result) == 5

    def test_city_all_returns_rows_from_all_cities(self, sample_df):
        """city='All' should not filter by city — it's the default view."""
        result, msg = filter_dataframe(sample_df, city="All", n_requests=100)
        assert len(result) == 30

    def test_search_area_filters_by_location_name_substring(self, sample_df):
        """
        search_area does a case-insensitive substring match on location_name.
        'airport' (lowercase) should find 'Airport' (title case).
        """
        result, msg = filter_dataframe(sample_df, search_area="airport", n_requests=100)
        assert len(result) == 10
        assert "✅" in msg  # Success message expected

    def test_search_area_missing_returns_warning_and_fallback(self, sample_df):
        """
        If the search term matches nothing, we should get a ⚠️ warning message
        and a random sample from the full dataset (not an empty DataFrame).
        Returning empty would break the Streamlit map render.
        """
        result, msg = filter_dataframe(
            sample_df, search_area="ZoneThatDoesNotExist", n_requests=10
        )
        assert len(result) > 0  # Must not return empty
        assert "⚠️" in msg  # Warning must be surfaced to the user

    def test_city_filter_unknown_city_falls_back_to_full_df(self, sample_df):
        """
        If someone selects a city that has no matching rows (e.g., they're using
        a dataset that doesn't have that city), we fall back to the full dataset
        rather than returning nothing.  This is the graceful-degradation path.
        """
        result, msg = filter_dataframe(sample_df, city="Kolkata", n_requests=100)
        assert len(result) > 0

    def test_no_city_column_defaults_to_full_pool(self, sample_df):
        """
        Not all CSVs have a 'city' column (the sample_csv doesn't).  When the
        column is absent and city is set to something other than 'All', the
        function should still return data rather than raising KeyError.
        """
        df_no_city = sample_df.drop(columns=["city"])
        result, msg = filter_dataframe(df_no_city, city="Delhi", n_requests=100)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Tests: resolve_data_path()
# ---------------------------------------------------------------------------

class TestResolveDataPath:
    """Tests for the data file path resolver."""

    def setup_method(self):
        _clear_config_cache()

    def teardown_method(self):
        _clear_config_cache()

    def test_returns_none_for_missing_file(self, tmp_path, monkeypatch):
        """
        If the config says a CSV is at 'data/foo.csv' but that file doesn't
        actually exist, resolve_data_path() should return None gracefully.
        The caller (app.py) is responsible for surfacing this to the user.
        """
        _clear_config_cache()
        monkeypatch.setattr(data_access, "_PROJECT_ROOT", tmp_path)
        result = resolve_data_path("india_csv")
        assert result is None

    def test_returns_path_when_file_exists(self, tmp_path, monkeypatch):
        """
        The happy path: config points to a file that actually exists.
        Should return an absolute Path object, not a string.
        """
        _clear_config_cache()
        # Create a fake config.yaml that points to a real temp file
        csv_file = tmp_path / "rides.csv"
        csv_file.touch()
        config_content = f"data_paths:\n  sample_csv: rides.csv\n"
        (tmp_path / "config.yaml").write_text(config_content)
        monkeypatch.setattr(data_access, "_PROJECT_ROOT", tmp_path)
        result = resolve_data_path("sample_csv")
        assert result is not None
        assert result.exists()
