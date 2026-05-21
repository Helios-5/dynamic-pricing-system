"""
app.py
------
Streamlit UI entry point for the AI-Driven Dynamic Pricing System.

Responsibilities (this file only):
  • Streamlit page configuration and CSS theming.
  • Sidebar controls and user-input collection.
  • Routing between data sources (synthetic / CSV).
  • Rendering metrics, charts, and the live map.

All data-loading, coordinate imputation, DataFrame filtering, and
business-logic helpers live in ``src/data_access.py``.
All ML training/inference lives in ``src/model.py``.
All optimisation lives in ``src/optimization.py``.
"""

import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

from data_access import (
    filter_dataframe,
    impute_coordinates,
    resolve_data_path,
)
from features import create_features
from generator import generate_synthetic_data
from model import FEATURES, load_model, save_model, train_model
from optimization import optimize_price

# ── Logging setup ─────────────────────────────────────────────────────────────
# Streamlit does not call __main__, so configure logging here once.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Dynamic Pricing", layout="wide", page_icon="💎")

# ── Professional UI CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --white: #ffffff;
        --off-white: #f8f9fa;
        --light-grey: #e9ecef;
        --grey: #6c757d;
        --dark-grey: #495057;
        --black: #212529;
        --blue: #0d6efd;
        --cyan: #0dcaf0;
        --blue-light: #d0e7ff;
        --shadow: rgba(0, 0, 0, 0.1);
    }

    * { transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease; }

    .main {
        background: var(--white);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        color: var(--black);
    }

    .stSidebar {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        border-right: 3px solid var(--blue);
    }

    .stSidebar [data-testid="stMarkdownContainer"] p,
    .stSidebar label { color: var(--black) !important; font-weight: 500; font-size: 0.9rem; }

    .stSidebar h1, .stSidebar h2, .stSidebar h3 { color: var(--blue) !important; }

    /* Metric Cards */
    [data-testid="stMetric"] {
        background: var(--white);
        padding: 1.75rem 1.5rem;
        border-radius: 16px;
        border: 2px solid var(--light-grey);
        box-shadow: 0 4px 12px var(--shadow), 0 1px 3px rgba(0,0,0,.05);
        position: relative;
        overflow: hidden;
    }
    [data-testid="stMetric"]::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--blue), var(--cyan));
        opacity: 0;
        transition: opacity 0.3s;
    }
    [data-testid="stMetric"]:hover::before { opacity: 1; }
    [data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        border-color: var(--blue);
        box-shadow: 0 8px 24px rgba(13,110,253,.15), 0 4px 8px var(--shadow);
    }
    [data-testid="stMetricLabel"] {
        color: var(--grey) !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="stMetricValue"] {
        font-size: 2.25rem !important;
        font-weight: 800 !important;
        font-family: 'Inter', sans-serif;
        color: var(--black);
    }
    [data-testid="stMetric"]:nth-child(1) [data-testid="stMetricValue"] {
        background: linear-gradient(135deg, var(--blue) 0%, var(--cyan) 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    }
    [data-testid="stMetric"]:nth-child(2) [data-testid="stMetricValue"] {
        background: linear-gradient(135deg, #0d6efd 0%, #6ea8fe 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    }
    [data-testid="stMetric"]:nth-child(3) [data-testid="stMetricValue"] {
        background: linear-gradient(135deg, var(--cyan) 0%, #6edff6 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    }
    [data-testid="stMetric"]:nth-child(4) [data-testid="stMetricValue"] {
        background: linear-gradient(135deg, #495057 0%, var(--grey) 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    }

    /* Typography */
    h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 2.75rem;
        color: var(--black);
        margin-bottom: 0.25rem;
        letter-spacing: -0.02em;
        border-bottom: 4px solid var(--blue);
        padding-bottom: 0.75rem;
    }
    h2 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 1.5rem;
        color: var(--blue);
        margin-top: 2.5rem;
        margin-bottom: 1rem;
        letter-spacing: -0.01em;
    }
    h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 1.1rem;
        color: var(--dark-grey);
        letter-spacing: 0.01em;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: var(--off-white);
        padding: 8px;
        border-radius: 12px;
        border: 2px solid var(--light-grey);
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background: var(--white);
        border-radius: 10px;
        color: var(--grey);
        font-weight: 600;
        font-size: 0.95rem;
        padding: 0 24px;
        border: 2px solid var(--light-grey);
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: var(--blue-light);
        color: var(--blue);
        border-color: var(--blue);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--blue), var(--cyan));
        color: white !important;
        border-color: var(--blue);
        box-shadow: 0 4px 12px rgba(13,110,253,.25);
    }

    /* Inputs */
    .stTextInput input, .stSelectbox select, .stTextArea textarea {
        background: var(--white) !important;
        border: 2px solid var(--light-grey) !important;
        border-radius: 10px;
        color: var(--black) !important;
        padding: 0.75rem 1rem;
        font-size: 0.95rem;
    }
    .stTextInput input:focus, .stSelectbox select:focus, .stTextArea textarea:focus {
        border-color: var(--blue) !important;
        box-shadow: 0 0 0 3px rgba(13,110,253,.1) !important;
        outline: none;
    }

    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, var(--blue) 0%, var(--cyan) 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.75rem;
        font-weight: 700;
        font-size: 0.95rem;
        box-shadow: 0 4px 12px rgba(13,110,253,.3);
        letter-spacing: 0.02em;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(13,110,253,.4);
        background: linear-gradient(135deg, #0b5ed7 0%, #0bb5d4 100%);
    }

    /* Alerts */
    .stAlert {
        background: var(--off-white);
        border: 2px solid var(--light-grey);
        border-left: 4px solid var(--blue);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        color: var(--black);
    }
    .stSuccess { border-left-color: #198754; background: #d1e7dd; }
    .stWarning { border-left-color: #ffc107; background: #fff3cd; }
    .stError   { border-left-color: #dc3545; background: #f8d7da; }
    .stInfo    { border-left-color: var(--cyan); background: #cff4fc; }

    /* Sliders */
    .stSlider { padding: 1rem 0; }
    .stSlider > label { color: var(--black) !important; font-weight: 600; }

    /* Selectbox */
    .stSelectbox label { color: var(--black); font-weight: 600; }

    /* Markdown */
    .stMarkdown { color: var(--dark-grey); }
    p, li, span { color: var(--dark-grey); }

    hr { border-color: var(--light-grey); margin: 2rem 0; }

    code {
        background: var(--off-white);
        color: var(--blue);
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        border: 1px solid var(--light-grey);
    }

    pre {
        background: var(--off-white);
        border: 1px solid var(--light-grey);
        border-radius: 8px;
    }

    .stDataFrame {
        background: var(--white);
        border: 2px solid var(--light-grey);
        border-radius: 12px;
    }

    .stExpander {
        background: var(--white);
        border: 1px solid var(--light-grey);
        border-radius: 10px;
    }

    .stFileUploader {
        background: var(--off-white);
        border: 2px dashed var(--light-grey);
        border-radius: 10px;
    }

    .stProgress > div > div {
        background: linear-gradient(90deg, var(--blue), var(--cyan));
    }

    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 2.5rem;
        max-width: 1400px;
    }
</style>
""", unsafe_allow_html=True)


# ── Cached data / model helpers ───────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_csv(data_key: str) -> pd.DataFrame | None:
    """Load a CSV data file by config key (``'india_csv'`` or ``'sample_csv'``)."""
    path = resolve_data_path(data_key)
    if path is None:
        return None
    logger.info("Loading CSV from %s", path)
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_resource(show_spinner=False)
def _get_model(use_csv: bool, data_source: str):
    """Load a cached model, or train one if no persisted file exists."""
    model_path = "model_csv.joblib" if use_csv else "model_synthetic.joblib"

    if os.path.exists(model_path):
        logger.info("Loading persisted model from %s", model_path)
        return load_model(model_path)

    with st.spinner(f"Training Random Forest model ({'CSV' if use_csv else 'Synthetic'})…"):
        try:
            if use_csv:
                data_key = "india_csv" if "Indian" in data_source else "sample_csv"
                df = _load_csv(data_key)
                if df is None:
                    st.error("Data file not found. Check config.yaml → data_paths.")
                    return None
                if "traffic" not in df.columns:
                    df["traffic"] = "High"
            else:
                df = generate_synthetic_data(n_samples=2000)

            df = create_features(df)
            model, metrics = train_model(df)
            save_model(model, model_path)
            logger.info("Model trained and saved: %s", metrics)
            return model
        except Exception as exc:
            st.error(f"Error during model training: {exc}")
            logger.exception("Model training failed.")
            return None


# ── Main application ──────────────────────────────────────────────────────────

def main() -> None:  # noqa: C901  (complexity tolerated for a single UI entry point)
    st.title("AI-Driven Dynamic Pricing System")
    st.markdown("### Intelligent Fare Optimization Engine")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Configuration")

        data_source = st.radio(
            "Training Data Source",
            ["Synthetic Simulation", "Real Data (CSV)", "Indian Market Data (CSV)"],
        )
        use_csv = data_source in ["Real Data (CSV)", "Indian Market Data (CSV)"]

        if use_csv:
            key = "india_csv" if "Indian" in data_source else "sample_csv"
            path = resolve_data_path(key)
            if path:
                st.success(f"Using `{path.name}`")
            else:
                st.warning("CSV file not found – check `config.yaml`.")

        st.divider()
        st.header("Simulation Control")
        st.subheader("Contextual Factors")

        time_of_day = st.slider("Time of Day (24h)", 0, 23, 18)
        weather = st.selectbox("Weather Condition", ["Clear", "Rainy", "Foggy"])
        event_status = st.selectbox(
            "Event Status", ["None", "Concert", "Sports", "Festival"]
        )

        st.subheader("Market Dynamics")
        n_requests = st.slider("Active Requests", 10, 300, 120)
        st.info("Adjust sliders to simulate different market conditions.")

    # ── Data preparation ──────────────────────────────────────────────────────
    current_date = datetime.now().replace(hour=time_of_day, minute=0, second=0)

    selected_city = "All"
    search_area = ""

    if data_source == "Indian Market Data (CSV)":
        df_full = _load_csv("india_csv")

        if df_full is None:
            st.error("Indian Market CSV not found. Falling back to synthetic data.")
            df_sim = generate_synthetic_data(n_samples=n_requests, start_date=current_date)
            df_sim["weather"] = weather
            df_sim["event"] = event_status
        else:
            with st.sidebar:
                st.subheader("Location Filter")
                selected_city = st.selectbox(
                    "Select City", ["All", "Delhi", "Mumbai", "Bangalore"]
                )
                search_area = st.text_input(
                    "Search Area", placeholder="e.g., Connaught Place"
                )

            df_sim, status_msg = filter_dataframe(
                df_full,
                city=selected_city,
                search_area=search_area,
                n_requests=n_requests,
            )

            if status_msg:
                if status_msg.startswith("⚠️"):
                    st.sidebar.warning(status_msg)
                else:
                    st.sidebar.success(status_msg)

            df_sim = impute_coordinates(df_sim)

    elif use_csv:
        df_full = _load_csv("sample_csv")
        if df_full is None:
            st.error("Sample CSV not found. Falling back to synthetic data.")
            df_sim = generate_synthetic_data(n_samples=n_requests, start_date=current_date)
            df_sim["weather"] = weather
            df_sim["event"] = event_status
        else:
            df_sim, _ = filter_dataframe(df_full, n_requests=n_requests)
            df_sim = impute_coordinates(df_sim)
    else:
        df_sim = generate_synthetic_data(n_samples=n_requests, start_date=current_date)
        df_sim["weather"] = weather
        df_sim["event"] = event_status

    # ── Feature engineering & inference ──────────────────────────────────────
    df_features = create_features(df_sim)

    model = _get_model(use_csv=use_csv, data_source=data_source)
    if model is None:
        st.error("Failed to load or train the model. Check the logs.")
        return

    city_center_rows = df_features[df_features["location_name"] == "City Center"]
    representative_row = (
        city_center_rows.iloc[0] if not city_center_rows.empty else df_features.iloc[0]
    )

    input_df = pd.DataFrame([representative_row])[FEATURES]
    predicted_multiplier: float = float(model.predict(input_df)[0])

    current_utilization = min(1.0, float(representative_row["demand_ratio"]))
    optimized_multiplier = optimize_price(predicted_multiplier, current_utilization)

    # ── Top metrics row ───────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        dr = representative_row["demand_ratio"]
        st.metric(
            "Avg Demand Ratio",
            f"{dr:.2f}",
            delta="High" if dr > 1.2 else "Normal",
        )
    with col2:
        st.metric(
            "Driver Utilization",
            f"{current_utilization * 100:.1f}%",
            delta_color="inverse",
        )
    with col3:
        st.metric("AI Predicted Surge", f"{predicted_multiplier:.2f}x")
    with col4:
        delta = optimized_multiplier - predicted_multiplier
        st.metric(
            "Optimized Surge",
            f"{optimized_multiplier:.2f}x",
            delta=f"{delta:.2f}",
            delta_color="normal",
        )

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["🗺️ Live Map", "📊 Analytics & Insights", "🤖 Model Performance"])

    # ── Tab 1: Live Map ───────────────────────────────────────────────────────
    with tab1:
        c1, c2 = st.columns([3, 1])

        with c1:
            map_style = st.selectbox(
                "Map Style", ["Light", "Dark", "Satellite", "Road"], index=0
            )
            map_styles = {
                "Light": "light",
                "Dark": "dark",
                "Satellite": "satellite",
                "Road": "road",
            }

            layer = pdk.Layer(
                "ScatterplotLayer",
                df_sim,
                get_position=["longitude", "latitude"],
                get_color=[200, 30, 0, 160],
                get_radius=100,
                pickable=True,
            )

            if not df_sim.empty and "latitude" in df_sim.columns:
                mid_lat = df_sim["latitude"].mean()
                mid_lon = df_sim["longitude"].mean()
                zoom = 14 if search_area else (12 if selected_city != "All" else 10)
            else:
                from data_access import get_default_coords
                mid_lat, mid_lon = get_default_coords()
                zoom = 10

            view_state = pdk.ViewState(
                latitude=mid_lat,
                longitude=mid_lon,
                zoom=zoom,
                pitch=50,
            )

            st.pydeck_chart(
                pdk.Deck(
                    map_style=map_styles[map_style],
                    layers=[layer],
                    initial_view_state=view_state,
                    tooltip={
                        "html": (
                            "<b>{location_name}</b><br/>"
                            "Requests: {requests}<br/>"
                            "Drivers: {drivers}<br/>"
                            "Weather: {weather}<br/>"
                            "Base Fare: {base_fare}<br/>"
                        ),
                        "style": {"backgroundColor": "steelblue", "color": "white"},
                    },
                )
            )

        with c2:
            st.markdown("#### Pricing Decision")

            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=optimized_multiplier,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Surge Multiplier"},
                    gauge={
                        "axis": {"range": [1, 5]},
                        "bar": {"color": "#FF4B4B"},
                        "steps": [
                            {"range": [1, 1.5], "color": "lightgreen"},
                            {"range": [1.5, 2.5], "color": "yellow"},
                            {"range": [2.5, 5], "color": "red"},
                        ],
                    },
                )
            )
            fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

            base_fare = representative_row.get("base_fare", 100)
            st.info(f"Base Fare: ₹{base_fare}")
            st.success(f"Final Fare: ₹{base_fare * optimized_multiplier:.2f}")

    # ── Tab 2: Analytics & Insights ───────────────────────────────────────────
    with tab2:
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Demand vs Supply Trend (Simulated)")
            hours = list(range(24))
            rng = np.random.default_rng()
            demand_trend = [
                50 + 20 * np.sin(h / 24 * 2 * np.pi) + rng.normal(0, 5)
                for h in hours
            ]
            supply_trend = [
                40 + 15 * np.sin(h / 24 * 2 * np.pi) + rng.normal(0, 5)
                for h in hours
            ]
            df_trend = pd.DataFrame(
                {"Hour": hours, "Demand": demand_trend, "Supply": supply_trend}
            )
            fig_trend = px.line(
                df_trend, x="Hour", y=["Demand", "Supply"], title="24H Market Trend"
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        with col_b:
            st.subheader("Price Sensitivity Analysis")
            ratios = np.linspace(0.5, 3.0, 50)
            prices = [1.0 + max(0, (r - 1.5) * 0.5) for r in ratios]
            df_sens = pd.DataFrame({"Demand Ratio": ratios, "Price Multiplier": prices})
            fig_sens = px.area(
                df_sens,
                x="Demand Ratio",
                y="Price Multiplier",
                title="Surge Logic Curve",
            )
            st.plotly_chart(fig_sens, use_container_width=True)

    # ── Tab 3: Model Performance ──────────────────────────────────────────────
    with tab3:
        st.subheader("Model Diagnostics")

        # The model might be a RandomForestRegressor (which has feature_importances_)
        # or a LinearRegression (which doesn't — it has coefficients instead, but
        # those aren't comparable to importance scores and would mislead stakeholders
        # if we displayed them as a bar chart). So we only show the chart when RF won.
        if hasattr(model, "feature_importances_"):
            feat_imp = (
                pd.DataFrame(
                    {"Feature": FEATURES, "Importance": model.feature_importances_}
                )
                .sort_values("Importance", ascending=False)
            )
            fig_imp = px.bar(
                feat_imp,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Feature Importance (Random Forest — Mean Decrease in Impurity)",
                color="Importance",
                color_continuous_scale="Blues",
            )
            fig_imp.update_layout(showlegend=False)
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info(
                "The Linear Regression baseline won this training run. "
                "Feature importances are not available for linear models "
                "(coefficients serve a different purpose and aren't shown here "
                "to avoid misleading comparisons). Retrain on a larger dataset "
                "to likely restore the Random Forest as winner."
            )

        st.markdown(
            """
            **Notes**
            - The model is a **Random Forest Regressor** (or LinearRegression
              baseline if RF underperformed on this dataset). Both are trained
              each session; the lower-RMSE model is used for inference.
            - The optimization layer uses a **soft-constraint Linear Program**
              (PuLP + CBC solver). Violations of utilization/retention targets
              incur a penalty cost rather than making the LP infeasible.
            - Penalty weights are configurable in `config.yaml` →
              `optimization` section — no redeploy needed.
            """
        )


if __name__ == "__main__":
    main()
