import pandas as pd
import numpy as np

def create_features(df):
    """
    Creates features from the raw data.
    
    Args:
        df (pd.DataFrame): Raw data DataFrame.
        
    Returns:
        pd.DataFrame: DataFrame with added features.
    """
    df = df.copy()
    
    # Demand Ratio
    df['demand_ratio'] = df['requests'] / df['drivers']
    
    # Time features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Time of day encoding
    def get_time_slot(h):
        if 5 <= h < 12: return 'Morning'
        elif 12 <= h < 17: return 'Afternoon'
        elif 17 <= h < 21: return 'Evening'
        else: return 'Night'
        
    df['time_slot'] = df['hour'].apply(get_time_slot)
    
    # One-hot encoding for categorical variables
    # Note: For a real model, we'd use a pipeline, but for this prototype we'll do it manually or rely on model's ability
    # We will create numeric flags for simplicity in the regression model
    
    df['is_rainy'] = (df['weather'] == 'Rainy').astype(int)
    df['is_peak_hour'] = df['hour'].apply(lambda h: 1 if (8 <= h <= 10 or 17 <= h <= 20) else 0)
    df['is_city_center'] = (df['location_name'] == 'City Center').astype(int)
    
    # Target Variable Simulation (for training purposes)
    # In a real scenario, this would be historical data. Here we simulate the "ideal" multiplier.
    # Logic: High demand ratio -> High multiplier
    
    def calculate_ideal_multiplier(row):
        multiplier = 1.0
        if row['demand_ratio'] > 1.5:
            multiplier += (row['demand_ratio'] - 1.5) * 0.5
        
        if row['is_rainy']:
            multiplier += 0.2
            
        if row['event'] != 'None':
            multiplier += 0.3
            
        return round(max(1.0, min(5.0, multiplier)), 2)
        
    df['price_multiplier'] = df.apply(calculate_ideal_multiplier, axis=1)
    
    return df

if __name__ == "__main__":
    # Test
    from generator import generate_synthetic_data
    df = generate_synthetic_data(n_samples=10)
    df_features = create_features(df)
    print(df_features[['requests', 'drivers', 'demand_ratio', 'price_multiplier']].head())
