# 📘 Technical Reference Manual

## System Architecture

The AI-Driven Dynamic Pricing System consists of four main modules that work together to produce the final pricing decision.

### 1. Data Simulation (`src/generator.py`)
- **Purpose**: Simulates a realistic ride-sharing environment.
- **Logic**:
    - Uses Poisson distribution to simulate arrival rates of requests and drivers.
    - Adjusts base parameters based on:
        - **Time**: Peak hours (8-10 AM, 5-8 PM) have higher demand.
        - **Location**: "City Center" and "Tech Park" have specific demand spikes.
        - **Weather**: Rain increases demand and decreases supply.
- **Output**: A DataFrame containing raw ride events.

### 2. Feature Engineering (`src/features.py`)
- **Purpose**: Transforms raw data into model-ready features.
- **Key Features**:
    - `demand_ratio`: `requests / drivers`. The most critical indicator.
    - `is_peak_hour`: Boolean flag for peak times.
    - `is_rainy`: Boolean flag for adverse weather.
    - `is_weekend`: Boolean flag for weekends.
    - `time_slot`: Categorical binning of hours (Morning, Afternoon, Evening, Night).

### 3. Machine Learning Model (`src/model.py`)
- **Algorithm**: Random Forest Regressor (`sklearn.ensemble.RandomForestRegressor`).
- **Why Random Forest?**:
    - Handles non-linear relationships well (e.g., price sensitivity isn't always linear).
    - Robust to outliers.
    - Provides feature importance for explainability.
- **Training**:
    - Target Variable: `price_multiplier` (Simulated based on a rule-based "ideal" logic for prototype purposes).
    - Metrics: MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error).

### 4. Optimization Layer (`src/optimization.py`)
- **Purpose**: Refines the ML prediction to ensure business constraints are met.
- **Method**: Linear Programming using `PuLP`.
- **Objective**: Maximize Price (Revenue) subject to constraints.
- **Constraints**:
    1. **Retention Constraint**: Price shouldn't be so high that customer retention drops below 80%.
       - Formula: `(1.0 - 0.1 * (price - 1)) >= 0.8`
    2. **Utilization Constraint**: Price should adjust to keep driver utilization below 90% (to ensure availability).
       - Formula: `(current_utilization - 0.05 * (price - 1)) <= 0.9`
    3. **Trust Region**: The optimized price must be within ±20% of the ML predicted price to avoid erratic behavior.

## API Reference

### `src.model.train_model(df)`
Trains the Random Forest model.
- **Args**: `df` (pd.DataFrame) - The training dataset.
- **Returns**: `(model, metrics)` - The trained model object and a dictionary of metrics.

### `src.optimization.optimize_price(predicted_price, current_utilization)`
Calculates the optimal price multiplier.
- **Args**:
    - `predicted_price` (float): The multiplier output by the ML model.
    - `current_utilization` (float): Current driver utilization rate (0.0 to 1.0).
- **Returns**: `float` - The final optimized price multiplier.

## Future Improvements
- **Reinforcement Learning**: Replace the supervised regression model with a Q-Learning agent that learns optimal pricing policies by interacting with the environment.
- **Real-time Data Stream**: Integrate with Kafka or RabbitMQ to ingest real-time ride data instead of simulation.
- **A/B Testing**: Implement a framework to test different pricing strategies on live traffic.
