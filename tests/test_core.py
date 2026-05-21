"""
test_core.py
------------
Smoke tests for the core pricing components.
Refactored to align with the new decoupled architecture and import paths.
"""
import pandas as pd
import pytest
from features import create_features
from optimization import optimize_price
from generator import generate_synthetic_data


def test_generate_synthetic_data():
    """Test that data generation returns a DataFrame with correct columns."""
    df = generate_synthetic_data(n_samples=10)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10
    required_cols = ['requests', 'drivers', 'weather', 'timestamp']
    for col in required_cols:
        assert col in df.columns


def test_create_features():
    """Test feature engineering logic and target variable noise."""
    data = {
        'requests': [100, 50],
        'drivers': [10, 10],
        'timestamp': pd.to_datetime(['2023-01-01 09:00:00', '2023-01-01 22:00:00']),
        'weather': ['Rainy', 'Clear'],
        'location_name': ['City Center', 'Suburbs'],
        'event': ['None', 'None']
    }
    df = pd.DataFrame(data)
    df_features = create_features(df)

    # Check demand ratio calculation
    assert df_features.loc[0, 'demand_ratio'] == 10.0
    assert df_features.loc[1, 'demand_ratio'] == 5.0

    # Check binary flags
    assert df_features.loc[0, 'is_rainy'] == 1
    assert df_features.loc[1, 'is_rainy'] == 0

    # Check target exists
    assert 'price_multiplier' in df_features.columns


def test_optimize_price_logic():
    """
    Test optimization soft constraints. 
    Unlike the old version, this should never fail with infeasibility.
    """
    # Test 1: Basic case
    price = optimize_price(predicted_price=2.0, current_utilization=0.5)
    assert isinstance(price, float)
    assert 1.0 <= price <= 5.0

    # Test 2: High utilization
    # The LP solver will use slack variables to balance this rather than crashing
    price_high_util = optimize_price(predicted_price=1.5, current_utilization=0.95)
    assert isinstance(price_high_util, float)

    # Test 3: Low retention check
    price_retention = optimize_price(predicted_price=4.0, current_utilization=0.5)
    assert isinstance(price_retention, float)