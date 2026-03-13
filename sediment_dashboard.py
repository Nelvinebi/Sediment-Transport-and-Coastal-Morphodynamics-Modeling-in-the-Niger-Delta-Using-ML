import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from sklearn.ensemble import RandomForestRegressor

# Page configuration
st.set_page_config(
    page_title="Niger Delta Sediment Transport Dashboard",
    layout="wide"
)

st.title("Sediment Transport & Coastal Morphodynamics Dashboard")

st.markdown("""
Machine learning analysis of sediment transport dynamics in the Niger Delta.
""")

# Load dataset
df = pd.read_excel("data.xlsx")

# Sidebar navigation
section = st.sidebar.radio(
    "Navigate",
    [
        "Project Overview",
        "Dataset Explorer",
        "Morphodynamic Analysis",
        "Feature Importance",
        "Prediction Tool"
    ]
)

# -------------------
# Overview
# -------------------
if section == "Project Overview":

    st.header("Project Overview")

    st.write("""
    This dashboard explores sediment transport dynamics along the Niger Delta
    using machine learning and synthetic coastal datasets.
    """)

    st.metric("Dataset Size", len(df))
    st.metric("Environmental Variables", len(df.columns)-2)

# -------------------
# Dataset Explorer
# -------------------
elif section == "Dataset Explorer":

    st.header("Dataset")

    st.dataframe(df)

    st.subheader("Statistical Summary")
    st.write(df.describe())

# -------------------
# Morphodynamic Analysis
# -------------------
elif section == "Morphodynamic Analysis":

    st.header("Morphodynamic Analysis")

    fig1 = px.scatter(
        df,
        x="wave_height_m",
        y="sediment_transport_rate",
        title="Wave Height vs Sediment Transport"
    )

    st.plotly_chart(fig1)

    fig2 = px.scatter(
        df,
        x="current_velocity_ms",
        y="sediment_transport_rate",
        title="Current Velocity vs Sediment Transport"
    )

    st.plotly_chart(fig2)

    fig3 = px.histogram(
        df,
        x="morphodynamic_index",
        title="Morphodynamic Index Distribution"
    )

    st.plotly_chart(fig3)

# -------------------
# Feature Importance
# -------------------
elif section == "Feature Importance":

    st.header("Key Drivers of Sediment Transport")

    X = df.drop(columns=["sediment_transport_rate"])
    y = df["sediment_transport_rate"]

    model = RandomForestRegressor()
    model.fit(X, y)

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    fig = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance"
    )

    st.plotly_chart(fig)

# -------------------
# Prediction Tool
# -------------------
elif section == "Prediction Tool":

    st.header("Sediment Transport Prediction")

    wave = st.slider("Wave Height (m)", 0.5, 5.0, 2.0)
    period = st.slider("Wave Period (s)", 4.0, 12.0, 7.0)
    velocity = st.slider("Current Velocity (m/s)", 0.1, 2.0, 0.5)
    tidal = st.slider("Tidal Range (m)", 0.1, 4.0, 1.5)
    grain = st.slider("Grain Size (mm)", 0.1, 2.0, 0.5)
    density = st.slider("Sediment Density (kg/m3)", 2500, 2800, 2650)
    slope = st.slider("Beach Slope", 1.0, 15.0, 5.0)
    orientation = st.slider("Shoreline Orientation", 0, 360, 180)

    X = df.drop(columns=["sediment_transport_rate"])
    y = df["sediment_transport_rate"]

    model = RandomForestRegressor()
    model.fit(X, y)

    input_data = [[
        wave,
        period,
        velocity,
        tidal,
        grain,
        density,
        slope,
        orientation,
        1
    ]]

    if st.button("Predict Transport Rate"):

        prediction = model.predict(input_data)

        st.success(f"Predicted Sediment Transport Rate: {prediction[0]:.3f}")
