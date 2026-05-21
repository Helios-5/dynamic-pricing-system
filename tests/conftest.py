"""
conftest.py
-----------
Shared pytest fixtures and path configuration for the Dynamic Pricing test suite.

The most important thing this file does is insert ``src/`` into sys.path so
that every test module can ``import features``, ``import model``, etc. without
needing a relative-import hack in each file.  pytest discovers conftest.py
automatically in the tests/ directory and in the project root, so this is the
canonical place to put setup that every test file needs.

We also define a handful of shared DataFrame fixtures here rather than
duplicating the same 10-row construction in every test module.  If the raw
data schema changes (e.g., we add a new required column), updating this one
fixture is all that's needed — the tests themselves don't need to change.
"""

import sys
import os
from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Ensure the src/ directory is on the Python path.
#
# pytest is typically run from the project root, but it doesn't automatically
# add src/ to sys.path the way a proper package install would.  This means
# ``import features`` would fail without this line.  We do it in conftest.py
# rather than in each test file because conftest.py is guaranteed to run first.
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


# ---------------------------------------------------------------------------
# Shared raw-data fixture.
#
# This is the minimum viable DataFrame that create_features() and train_model()
# need — six columns, twelve rows, carefully chosen to exercise different code
# paths (rainy vs. clear, peak vs. off-peak, weekend vs. weekday, city center
# vs. suburb, with and without the optional 'event' column).
#
# We use a fixed timestamp so time-derived features (hour, day_of_week,
# is_weekend, is_peak_hour, time_slot) are deterministic.  The dates below
# cover a Monday morning peak (2024-01-08 09:00 — is_peak_hour=1, is_weekend=0)
# and a Saturday night (2024-01-06 22:00 — is_peak_hour=0, is_weekend=1).
# ---------------------------------------------------------------------------
@pytest.fixture
def raw_ride_df() -> pd.DataFrame:
    """
    A deterministic, small raw ride DataFrame for unit tests.

    'Small' is intentional — we want tests that run in milliseconds, not seconds.
    200 rows would also work, but 12 rows is enough to exercise every branch.
    """
    return pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2024-01-08 09:00",  # Monday, 09h → peak hour, weekday
            "2024-01-08 09:00",  # Same slot, different conditions
            "2024-01-08 14:00",  # Monday, 14h → afternoon, not peak
            "2024-01-08 18:00",  # Monday, 18h → evening peak
            "2024-01-06 22:00",  # Saturday, 22h → weekend, night, no peak
            "2024-01-06 22:00",  # Same slot
            "2024-01-08 02:00",  # Monday, 02h → night, not peak
            "2024-01-08 06:00",  # Monday, 06h → morning (just inside 05-12 window)
            "2024-01-07 08:30",  # Sunday, 08h → peak, weekend
            "2024-01-07 08:30",
            "2024-01-08 09:00",
            "2024-01-08 09:00",
        ]),
        "requests": [100, 80, 50, 120, 60, 30, 20, 40, 90, 70, 110, 55],
        "drivers":  [10,  20, 25, 15,  30, 15, 10, 20, 18, 14, 22,  11],
        "weather":  [
            "Rainy", "Clear", "Clear", "Rainy",
            "Clear", "Foggy", "Clear", "Clear",
            "Rainy", "Clear", "Clear", "Rainy",
        ],
        "location_name": [
            "City Center", "Suburbs", "Airport", "City Center",
            "Mall",        "Tech Park", "Suburbs", "Airport",
            "City Center", "Mall",     "Suburbs",  "City Center",
        ],
        "event": [
            "None", "None", "Concert", "None",
            "None", "None", "None",    "None",
            "Sports", "None", "None",  "None",
        ],
    })


@pytest.fixture
def featured_df(raw_ride_df) -> pd.DataFrame:
    """
    A feature-engineered version of raw_ride_df, ready for model training.

    Most model and optimization tests need features already computed — this
    fixture saves them from calling create_features() themselves and keeps
    test failures localized.  If test_features.py starts failing, the model
    tests should still give useful signal, not also fail due to a shared
    setup step.
    """
    from features import create_features
    return create_features(raw_ride_df)
