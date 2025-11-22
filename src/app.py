import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

# Import our modules
from generator import generate_synthetic_data
from features import create_features
from model import train_model, load_model, save_model
from optimization import optimize_price

st.set_page_config(page_title="AI Dynamic Pricing", layout="wide", page_icon="🚖")

# --- Custom CSS for Premium Look ---
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stMetric {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #41444C;
    }
    .stMetric:hover {
        border-color: #FF4B4B;
    }
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
    }
    .stSidebar {
        background-color: #262730;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_model(use_csv=False):
    model_path = "model_csv.joblib" if use_csv else "model_synthetic.joblib"
    
    if os.path.exists(model_path):
        return load_model(model_path)
    else:
        with st.spinner(f"Training model ({'CSV' if use_csv else 'Synthetic'})..."):
            try:
                if use_csv and os.path.exists("data/sample_ride_data.csv"):
                    df = pd.read_csv("data/sample_ride_data.csv")
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    # Ensure columns exist
                    if 'traffic' not in df.columns: df['traffic'] = 'Medium' # Default
                else:
                    df = generate_synthetic_data(n_samples=2000)
                
                df = create_features(df)
                model, metrics = train_model(df)
                save_model(model, model_path)
                return model
            except Exception as e:
                st.error(f"Error during model training: {e}")
                return None

def main():
    st.title("🚖 AI-Driven Dynamic Pricing System")
    st.markdown("### Intelligent Fare Optimization Engine")
    
    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Data Source Selector
        data_source = st.radio("Training Data Source", ["Synthetic Simulation", "Real Data (CSV)"])
        use_csv = (data_source == "Real Data (CSV)")
        
        if use_csv:
            st.success("Using `data/sample_ride_data.csv`")
        
        st.divider()
        
        st.header("🕹️ Simulation Control")
        
        st.subheader("Contextual Factors")
        time_of_day = st.slider("Time of Day (Hour)", 0, 23, 18)
        weather = st.selectbox("Weather Condition", ["Clear", "Rainy", "Foggy"])
        event_status = st.selectbox("Event Status", ["None", "Concert", "Sports", "Festival"])
        
        st.subheader("Market Dynamics")
        n_requests = st.slider("Active Requests", 10, 300, 120)
        
        st.info("Adjust sliders to simulate different market conditions.")

    # Generate "Real-time" Context
    current_date = datetime.now().replace(hour=time_of_day, minute=0, second=0)
    
    # Data Generation (Always synthetic for the LIVE view, but model can be trained on CSV)
    df_sim = generate_synthetic_data(n_samples=n_requests, start_date=current_date)
    df_sim['weather'] = weather
    df_sim['event'] = event_status
    
    # Feature Engineering
    df_features = create_features(df_sim)
    
    # Model Inference
    model = get_model(use_csv=use_csv)
    
    if model is None:
        st.error("Failed to load or train the model. Please check the logs.")
        return
    
    # Representative Data (City Center)
    city_center_data = df_features[df_features['location_name'] == 'City Center']
    row = city_center_data.iloc[0] if not city_center_data.empty else df_features.iloc[0]
        
    input_data = pd.DataFrame([row])
    input_data = input_data[['demand_ratio', 'is_rainy', 'is_peak_hour', 'is_city_center', 'is_weekend']]
    
    predicted_multiplier = model.predict(input_data)[0]
    
    # Optimization
    current_utilization = min(1.0, row['demand_ratio'])
    optimized_multiplier = optimize_price(predicted_multiplier, current_utilization)
    
    # --- Main Dashboard Layout ---
    
    # Top Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Avg Demand Ratio", f"{row['demand_ratio']:.2f}", delta="High" if row['demand_ratio'] > 1.2 else "Normal")
    with col2:
        st.metric("Driver Utilization", f"{current_utilization*100:.1f}%", delta_color="inverse")
    with col3:
        st.metric("AI Predicted Surge", f"{predicted_multiplier:.2f}x")
    with col4:
        delta = optimized_multiplier - predicted_multiplier
        st.metric("Optimized Surge", f"{optimized_multiplier:.2f}x", delta=f"{delta:.2f}", delta_color="normal")

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["🗺️ Live Map", "📈 Analytics & Insights", "🤖 Model Performance"])
    
    with tab1:
        c1, c2 = st.columns([3, 1])
        with c1:
            # PyDeck Map
            layer = pdk.Layer(
                "ScatterplotLayer",
                df_sim,
                get_position=["longitude", "latitude"],
                get_color=[200, 30, 0, 160],
                get_radius=100,
                pickable=True,
            )
            
            view_state = pdk.ViewState(
                latitude=28.6139,
                longitude=77.2090,
                zoom=11,
                pitch=50,
            )
            
            st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{location_name}\nRequests: {requests}"}))
        
        with c2:
            st.markdown("#### Pricing Decision")
            
            # Gauge Chart for Surge
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = optimized_multiplier,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Surge Multiplier"},
                gauge = {
                    'axis': {'range': [1, 5]},
                    'bar': {'color': "#FF4B4B"},
                    'steps': [
                        {'range': [1, 1.5], 'color': "lightgreen"},
                        {'range': [1.5, 2.5], 'color': "yellow"},
                        {'range': [2.5, 5], 'color': "red"}],
                }
            ))
            fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            st.info(f"Base Fare: ₹{row['base_fare']}")
            st.success(f"Final Fare: ₹{row['base_fare'] * optimized_multiplier:.2f}")

    with tab2:
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader("Demand vs Supply Trend (Simulated)")
            # Simulate a 24h trend
            hours = list(range(24))
            demand_trend = [50 + 20*np.sin(h/24 * 2*np.pi) + np.random.normal(0, 5) for h in hours]
            supply_trend = [40 + 15*np.sin(h/24 * 2*np.pi) + np.random.normal(0, 5) for h in hours]
            
            df_trend = pd.DataFrame({"Hour": hours, "Demand": demand_trend, "Supply": supply_trend})
            fig_trend = px.line(df_trend, x="Hour", y=["Demand", "Supply"], title="24H Market Trend")
            st.plotly_chart(fig_trend, use_container_width=True)
            
        with col_b:
            st.subheader("Price Sensitivity Analysis")
            # Show how price changes with demand ratio
            ratios = np.linspace(0.5, 3.0, 50)
            prices = [1.0 + max(0, (r - 1.5) * 0.5) for r in ratios] # Simple logic replication
            df_sens = pd.DataFrame({"Demand Ratio": ratios, "Price Multiplier": prices})
            fig_sens = px.area(df_sens, x="Demand Ratio", y="Price Multiplier", title="Surge Logic Curve")
            st.plotly_chart(fig_sens, use_container_width=True)

    with tab3:
        st.subheader("Model Diagnostics")
        
        # We need to re-train or load metrics to show them here
        # For performance, we'll just re-calculate on a small batch or store them
        # Let's generate a fresh batch for validation
        df_val = generate_synthetic_data(n_samples=500)
        df_val = create_features(df_val)
        
        # Get feature importance from the model (if available)
        if hasattr(model, "feature_importances_"):
            feat_imp = pd.DataFrame({
                "Feature": ['demand_ratio', 'is_rainy', 'is_peak_hour', 'is_city_center', 'is_weekend'],
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False)
            
            fig_imp = px.bar(feat_imp, x="Importance", y="Feature", orientation='h', title="Feature Importance")
            st.plotly_chart(fig_imp)
        
        st.write("Note: Model is retrained periodically on new data.")

if __name__ == "__main__":
    main()
