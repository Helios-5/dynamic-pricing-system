import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_sample_csv(filename="data/sample_ride_data.csv", n_samples=5000):
    """
    Creates a realistic CSV dataset for the dynamic pricing system.
    """
    np.random.seed(42)
    
    start_date = datetime.now() - timedelta(days=30)
    data = []
    
    locations = ["City Center", "Suburbs", "Airport", "Tech Park", "Mall"]
    weather_conditions = ["Clear", "Rainy", "Foggy", "Cloudy"]
    events = ["None", "Concert", "Sports", "Festival", "Conference"]
    
    for i in range(n_samples):
        timestamp = start_date + timedelta(minutes=np.random.randint(0, 30*24*60))
        hour = timestamp.hour
        day = timestamp.weekday()
        
        loc = np.random.choice(locations, p=[0.3, 0.3, 0.1, 0.2, 0.1])
        
        # Base demand logic
        base_demand = 50
        if loc == "City Center": base_demand += 30
        if loc == "Tech Park" and (8 <= hour <= 10 or 17 <= hour <= 19): base_demand += 50
        if loc == "Airport": base_demand += 20
        
        # Time impact
        if 17 <= hour <= 21: base_demand *= 1.5 # Evening peak
        if day >= 5: base_demand *= 1.2 # Weekend
        
        # Weather
        weather = np.random.choice(weather_conditions, p=[0.6, 0.2, 0.1, 0.1])
        if weather == "Rainy": base_demand *= 1.4
        
        # Event
        event = np.random.choice(events, p=[0.9, 0.02, 0.02, 0.02, 0.04])
        if event != "None": base_demand *= 1.6
        
        requests = int(np.random.poisson(base_demand))
        
        # Supply logic
        drivers = int(requests * np.random.uniform(0.6, 1.2)) # Supply fluctuates around demand
        if weather == "Rainy": drivers = int(drivers * 0.8)
        drivers = max(1, drivers)
        
        # Actual Price (Historical) - what was charged
        # Simple logic: Base + Surge
        base_fare = 100 + np.random.randint(0, 50)
        demand_ratio = requests / drivers
        surge = 1.0
        if demand_ratio > 1.2: surge += (demand_ratio - 1.2) * 0.8
        if weather == "Rainy": surge += 0.2
        
        final_price = base_fare * surge
        
        data.append({
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "location_name": loc,
            "requests": requests,
            "drivers": drivers,
            "weather": weather,
            "event": event,
            "base_fare": base_fare,
            "actual_price_paid": round(final_price, 2)
        })
        
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Created {filename} with {n_samples} rows.")

if __name__ == "__main__":
    create_sample_csv()
