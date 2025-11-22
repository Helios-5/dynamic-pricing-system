import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_data(n_samples=1000, start_date=None):
    """
    Generates synthetic data for the dynamic pricing system.
    
    Args:
        n_samples (int): Number of samples to generate.
        start_date (datetime): Starting timestamp.
        
    Returns:
        pd.DataFrame: DataFrame containing synthetic ride data.
    """
    if start_date is None:
        start_date = datetime.now()
        
    data = []
    
    # Define some location clusters (Lat, Long) - e.g., City Center, Suburbs, Airport
    locations = {
        "City Center": (28.6139, 77.2090),
        "Suburbs": (28.5355, 77.3910),
        "Airport": (28.5562, 77.1000)
    }
    
    weather_conditions = ["Clear", "Rainy", "Foggy"]
    traffic_conditions = ["Low", "Medium", "High"]
    event_types = ["None", "Concert", "Sports", "Festival"]
    
    for i in range(n_samples):
        # Time increment
        current_time = start_date + timedelta(minutes=i*15)
        hour = current_time.hour
        
        # Randomly select a location
        loc_name, (base_lat, base_long) = list(locations.items())[np.random.randint(0, len(locations))]
        
        # Add some noise to location
        lat = base_lat + np.random.normal(0, 0.01)
        long = base_long + np.random.normal(0, 0.01)
        
        # Simulate Demand (Requests) based on time and location
        base_demand = 50
        if 8 <= hour <= 10 or 17 <= hour <= 20: # Peak hours
            base_demand *= 2.0
        if loc_name == "City Center":
            base_demand *= 1.5
            
        # Weather impact
        weather = np.random.choice(weather_conditions, p=[0.7, 0.2, 0.1])
        if weather == "Rainy":
            base_demand *= 1.3
            
        # Event impact
        event = np.random.choice(event_types, p=[0.9, 0.03, 0.03, 0.04])
        if event != "None":
            base_demand *= 1.5
            
        requests = int(np.random.poisson(base_demand))
        
        # Simulate Supply (Drivers)
        base_drivers = 40
        if 8 <= hour <= 10 or 17 <= hour <= 20:
            base_drivers *= 1.2 # More drivers in peak, but maybe not enough
        
        if weather == "Rainy":
            base_drivers *= 0.8 # Fewer drivers in rain
            
        drivers = int(np.random.poisson(base_drivers))
        drivers = max(1, drivers) # Avoid division by zero
        
        # Base Fare
        base_fare = 100 + np.random.randint(-10, 20)
        
        data.append({
            "timestamp": current_time,
            "location_name": loc_name,
            "latitude": lat,
            "longitude": long,
            "requests": requests,
            "drivers": drivers,
            "weather": weather,
            "traffic": np.random.choice(traffic_conditions),
            "event": event,
            "base_fare": base_fare
        })
        
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = generate_synthetic_data(n_samples=10)
    print(df.head())
