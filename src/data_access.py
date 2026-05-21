"""
data_access.py
--------------
Data-access and business-logic helpers for the Dynamic Pricing System.

This module is the single point of truth for:
  • Loading and validating the application configuration (``config.yaml``).
  • Resolving canonical CSV data-file paths.
  • Imputing missing geographic coordinates into ride DataFrames.
  • Filtering ride DataFrames by city and area search term.

By concentrating these concerns here, ``src/app.py`` is free to focus
exclusively on Streamlit UI rendering and routing.

Configuration override precedence (highest → lowest):
  1. Environment variable (e.g. ``PRICING_DEFAULT_LAT``)
  2. ``config.yaml`` value
  3. Hard-coded default in this module
"""

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level defaults (last-resort fallbacks if config.yaml is absent)
# ---------------------------------------------------------------------------
_FALLBACK_DEFAULT_LAT: float = 28.6
_FALLBACK_DEFAULT_LON: float = 77.2

_FALLBACK_LOCATION_MAP: dict[str, tuple[float, float]] = {
    "City Center": (28.6139, 77.2090),
    "Suburbs": (28.5355, 77.3910),
    "Airport": (28.5562, 77.1000),
    "Mall": (28.5244, 77.2188),
    "Tech Park": (28.4950, 77.0895),
}

# Root of the project (two levels up from this file: src/ → project root)
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_config() -> dict:
    """
    Load and cache ``config.yaml`` from the project root.

    Environment variables take precedence over file values for scalar keys:
      • ``PRICING_DEFAULT_LAT``   → ``default_lat``
      • ``PRICING_DEFAULT_LON``   → ``default_lon``
      • ``PRICING_INDIA_CSV_PATH``  → ``data_paths.india_csv``
      • ``PRICING_SAMPLE_CSV_PATH`` → ``data_paths.sample_csv``

    Returns:
        Merged configuration dictionary.  Always returns a valid dict even if
        ``config.yaml`` is missing (uses module-level defaults).
    """
    config_path = _PROJECT_ROOT / "config.yaml"
    cfg: dict = {}

    if config_path.exists():
        try:
            import yaml  # Optional import – only needed at runtime
            with config_path.open("r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
            logger.debug("Config loaded from %s", config_path)
        except Exception as exc:
            logger.warning("Failed to parse config.yaml (%s). Using defaults.", exc)
    else:
        logger.warning(
            "config.yaml not found at %s. Using built-in defaults.", config_path
        )

    # Apply environment variable overrides (scalar keys only)
    env_overrides = {
        "default_lat": ("PRICING_DEFAULT_LAT", float),
        "default_lon": ("PRICING_DEFAULT_LON", float),
    }
    for cfg_key, (env_key, cast) in env_overrides.items():
        val = os.environ.get(env_key)
        if val is not None:
            try:
                cfg[cfg_key] = cast(val)
                logger.debug("Config override from env: %s=%s", cfg_key, val)
            except ValueError:
                logger.warning(
                    "Invalid value for env var %s='%s'; ignoring.", env_key, val
                )

    # data_paths overrides
    data_paths = cfg.setdefault("data_paths", {})
    for cfg_key, env_key in [
        ("india_csv", "PRICING_INDIA_CSV_PATH"),
        ("sample_csv", "PRICING_SAMPLE_CSV_PATH"),
    ]:
        val = os.environ.get(env_key)
        if val is not None:
            data_paths[cfg_key] = val
            logger.debug("Data path override from env: %s=%s", cfg_key, val)

    return cfg


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_location_fallback_map() -> dict[str, tuple[float, float]]:
    """
    Return a mapping of ``location_name`` → ``(latitude, longitude)``.

    Values are sourced from ``config.yaml`` (``location_fallbacks`` section)
    and fall back to built-in defaults if the config is unavailable.

    Returns:
        Dict mapping canonical location name strings to (lat, lon) tuples.
    """
    cfg = get_config()
    raw = cfg.get("location_fallbacks", {})

    if not raw:
        logger.debug("No location_fallbacks in config; using built-in defaults.")
        return dict(_FALLBACK_LOCATION_MAP)

    result: dict[str, tuple[float, float]] = {}
    for name, coords in raw.items():
        if isinstance(coords, (list, tuple)) and len(coords) == 2:
            result[name] = (float(coords[0]), float(coords[1]))
        else:
            logger.warning("Malformed coordinate for '%s' in config: %s", name, coords)

    return result


def get_default_coords() -> tuple[float, float]:
    """
    Return the ``(latitude, longitude)`` to use when a location has no
    entry in the fallback map.

    Priority: env vars → config.yaml → built-in defaults.
    """
    cfg = get_config()
    lat = float(cfg.get("default_lat", _FALLBACK_DEFAULT_LAT))
    lon = float(cfg.get("default_lon", _FALLBACK_DEFAULT_LON))
    return lat, lon


def resolve_data_path(key: str) -> Optional[Path]:
    """
    Return the absolute ``Path`` for a named data file.

    Args:
        key: One of ``"india_csv"`` or ``"sample_csv"``.

    Returns:
        Resolved ``Path`` if the file exists, else ``None``.
    """
    cfg = get_config()
    relative = cfg.get("data_paths", {}).get(key)
    if not relative:
        logger.warning("No data_paths.%s entry in config.", key)
        return None

    path = (_PROJECT_ROOT / relative).resolve()
    if not path.exists():
        logger.warning("Data file not found: %s", path)
        return None

    return path


def impute_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a DataFrame has ``latitude`` and ``longitude`` columns.

    If either column is missing, coordinates are derived from
    ``location_name`` using the fallback map from ``get_location_fallback_map()``.
    A small amount of Gaussian jitter is added so map points don't all
    collapse to the same pixel.

    Args:
        df: DataFrame that must contain a ``location_name`` column.

    Returns:
        Copy of ``df`` with ``latitude`` and ``longitude`` columns guaranteed.
    """
    if "latitude" in df.columns and "longitude" in df.columns:
        return df  # Already has coordinates – nothing to do

    logger.warning(
        "DataFrame missing lat/lon columns; imputing from location_name."
    )

    fallback_map = get_location_fallback_map()
    default_lat, default_lon = get_default_coords()

    df = df.copy()

    df["latitude"] = df["location_name"].map(
        lambda name: fallback_map.get(name, (default_lat, default_lon))[0]
    )
    df["longitude"] = df["location_name"].map(
        lambda name: fallback_map.get(name, (default_lat, default_lon))[1]
    )

    # Add spatial jitter so individual points are distinguishable on the map
    rng = np.random.default_rng()
    df["latitude"] += rng.normal(0, 0.01, len(df))
    df["longitude"] += rng.normal(0, 0.01, len(df))

    return df


def filter_dataframe(
    df: pd.DataFrame,
    city: str = "All",
    search_area: str = "",
    n_requests: int = 120,
) -> tuple[pd.DataFrame, Optional[str]]:
    """
    Filter a ride DataFrame by city and/or area name.

    Args:
        df:           Full ride DataFrame (e.g. loaded from the India CSV).
        city:         City name to filter on, or ``"All"`` for no city filter.
        search_area:  Substring to search within ``location_name`` (case-insensitive).
        n_requests:   Maximum number of rows to return (down-sampled if needed).

    Returns:
        Tuple of:
          - Filtered (and possibly down-sampled) DataFrame.
          - User-facing status message string, or ``None`` if no special status.
    """
    status_msg: Optional[str] = None

    if search_area:
        mask = df["location_name"].str.contains(search_area, case=False, na=False)
        filtered = df[mask]

        if filtered.empty:
            status_msg = f"⚠️ No data found for '{search_area}'. Showing random sample."
            logger.info("Search '%s' returned no results; falling back to city filter.", search_area)
            filtered = _city_filter(df, city, n_requests)
        else:
            count = len(filtered)
            status_msg = f"✅ Found {count} rides in '{search_area}'."
            if count > n_requests:
                filtered = filtered.sample(n=n_requests, random_state=None)
    else:
        filtered = _city_filter(df, city, n_requests)

    return filtered, status_msg


def _city_filter(df: pd.DataFrame, city: str, n: int) -> pd.DataFrame:
    """Return up to ``n`` rows from ``df``, optionally filtered to ``city``."""
    if city != "All" and "city" in df.columns:
        pool = df[df["city"] == city]
        if pool.empty:
            logger.warning("No rows for city='%s'; returning from full dataset.", city)
            pool = df
    else:
        pool = df

    sample_size = min(n, len(pool))
    return pool.sample(n=sample_size, random_state=None)
