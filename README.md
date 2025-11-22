# 🚖 AI-Driven Dynamic Pricing System

A smart, AI-powered dynamic pricing engine that adjusts service fares in real-time based on demand, supply, weather, and other contextual factors. This system simulates a ride-sharing environment to demonstrate how Machine Learning and Optimization techniques can maximize revenue while maintaining customer satisfaction.

![Dashboard Preview](https://via.placeholder.com/800x400?text=AI+Dynamic+Pricing+Dashboard)
*(Note: Replace with actual screenshot)*

## 🚀 Features

- **Real-time Simulation**: Generates synthetic ride requests and driver availability based on time, location, and weather.
- **AI Price Prediction**: Uses a **Random Forest Regressor** to predict the optimal price multiplier based on historical patterns.
- **Constraint Optimization**: Applies **Linear Programming (PuLP)** to ensure prices remain within ethical and business bounds (e.g., maintaining driver utilization and customer retention).
- **Interactive Dashboard**: A premium **Streamlit** interface with:
    - Live Heatmap of demand.
    - Real-time metrics (Demand Ratio, Utilization, Surge Factor).
    - Interactive charts (Gauge, Trends, Sensitivity Analysis).
- **Dual Data Mode**: Train the model on **Synthetic Data** (generated on-the-fly) or **Real-world Data** (CSV upload).

## 🛠️ Tech Stack

- **Language**: Python 3.9+
- **Dashboard**: Streamlit
- **Machine Learning**: Scikit-learn (Random Forest)
- **Optimization**: PuLP (Linear Programming)
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, PyDeck

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/dynamic-pricing-system.git
   cd dynamic-pricing-system
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🚦 Usage

1. **Run the Dashboard**:
   ```bash
   streamlit run src/app.py
   ```

2. **Explore the App**:
   - **Sidebar**: Adjust simulation parameters like Time of Day, Weather, and Active Requests.
   - **Data Source**: Toggle between "Synthetic Simulation" and "Real Data (CSV)" to see how the model adapts.
   - **Tabs**:
     - **Live Map**: View the geographical distribution of demand.
     - **Analytics**: Analyze demand/supply trends and price sensitivity.
     - **Model Performance**: Check feature importance and model metrics.

## 📂 Project Structure

```
dynamic-pricing-system/
├── data/                   # Data storage
│   └── sample_ride_data.csv # Generated sample dataset
├── docs/                   # Documentation
│   └── technical_reference.md
├── src/                    # Source code
│   ├── app.py              # Main Streamlit application
│   ├── generator.py        # Synthetic data generator
│   ├── features.py         # Feature engineering pipeline
│   ├── model.py            # ML Model training & inference
│   └── optimization.py     # Optimization logic
├── create_sample_data.py   # Script to generate CSV data
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## 🧠 How It Works

1. **Data Generation**: The system simulates ride requests with specific attributes (Location, Time, Weather).
2. **Feature Engineering**: Raw data is converted into features like `demand_ratio`, `is_peak_hour`, `is_rainy`.
3. **Prediction**: The ML model predicts a "base" surge multiplier based on these features.
4. **Optimization**: The optimization layer adjusts this multiplier to satisfy constraints (e.g., "Don't surge if utilization is low", "Cap surge at 5x").
5. **Visualization**: The final price and insights are displayed to the user.

## 📄 License

This project is licensed under the MIT License.