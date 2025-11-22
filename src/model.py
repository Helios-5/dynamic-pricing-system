from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import pandas as pd
import numpy as np

def train_model(df):
    """
    Trains the pricing model.
    
    Args:
        df (pd.DataFrame): DataFrame with features and target.
        
    Returns:
        model: Trained model.
        metrics (dict): Evaluation metrics.
    """
    # Features to use
    features = ['demand_ratio', 'is_rainy', 'is_peak_hour', 'is_city_center', 'is_weekend']
    target = 'price_multiplier'
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    metrics = {"MAE": mae, "RMSE": rmse}
    
    # Feature Importance
    importance = dict(zip(features, model.feature_importances_))
    metrics["feature_importance"] = importance
    
    return model, metrics

def save_model(model, path="model.joblib"):
    joblib.dump(model, path)

def load_model(path="model.joblib"):
    return joblib.load(path)

if __name__ == "__main__":
    from generator import generate_synthetic_data
    from features import create_features
    
    print("Generating data...")
    df = generate_synthetic_data(n_samples=1000)
    df = create_features(df)
    
    print("Training model...")
    model, metrics = train_model(df)
    print(f"Metrics: {metrics}")
    
    save_model(model)
    print("Model saved.")
