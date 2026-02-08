
# ============================================================
# Sediment Transport and Coastal Morphodynamics Modeling
# Niger Delta (Synthetic Data + Machine Learning)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# ------------------------------------------------------------
# 1. Synthetic Dataset Generator
# ------------------------------------------------------------

def generate_sediment_data(samples=3500):
    np.random.seed(42)

    data = pd.DataFrame({
        "wave_height_m": np.random.normal(1.8, 0.6, samples).clip(0.3, 4.5),
        "wave_period_s": np.random.normal(7, 2, samples).clip(3, 15),
        "current_velocity_ms": np.random.normal(0.9, 0.3, samples).clip(0.1, 2.5),
        "tidal_range_m": np.random.normal(1.5, 0.5, samples).clip(0.4, 3.8),
        "grain_size_mm": np.random.normal(0.25, 0.1, samples).clip(0.05, 1.0),
        "sediment_density_kgm3": np.random.normal(2650, 100, samples),
        "beach_slope": np.random.uniform(0.01, 0.12, samples),
        "shoreline_orientation_deg": np.random.uniform(0, 360, samples),
    })

    data["sediment_transport_rate"] = (
        0.35 * data["wave_height_m"] +
        0.25 * data["current_velocity_ms"] +
        0.15 * data["tidal_range_m"] +
        0.10 * (1 / data["grain_size_mm"]) +
        0.10 * data["beach_slope"] +
        0.05 * (data["wave_period_s"] / 15)
    ) * np.random.uniform(0.8, 1.2, samples)

    data["morphodynamic_index"] = (
        0.6 * data["sediment_transport_rate"] +
        0.3 * data["current_velocity_ms"] -
        0.2 * data["grain_size_mm"]
    )

    return data

# Generate dataset
df = generate_sediment_data()

# ------------------------------------------------------------
# 2. Feature Selection and Scaling
# ------------------------------------------------------------

X = df.drop(["sediment_transport_rate", "morphodynamic_index"], axis=1)
y = df["sediment_transport_rate"]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42
)

# ------------------------------------------------------------
# 3. ML Model: Random Forest Regressor
# ------------------------------------------------------------

model = RandomForestRegressor(
    n_estimators=250,
    max_depth=18,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ------------------------------------------------------------
# 4. Model Evaluation
# ------------------------------------------------------------

y_pred = model.predict(X_test)

print("Model Performance:")
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
print("R² Score:", r2_score(y_test, y_pred))

# ------------------------------------------------------------
# 5. Feature Importance Visualization
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.barh(X.columns, model.feature_importances_)
plt.title("Sediment Transport Driver Importance")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 6. Morphodynamic Prediction Example
# ------------------------------------------------------------

sample = pd.DataFrame([{
    "wave_height_m": 2.8,
    "wave_period_s": 9,
    "current_velocity_ms": 1.6,
    "tidal_range_m": 2.2,
    "grain_size_mm": 0.18,
    "sediment_density_kgm3": 2650,
    "beach_slope": 0.06,
    "shoreline_orientation_deg": 145
}])

sample_scaled = scaler.transform(sample)
predicted_transport = model.predict(sample_scaled)[0]

print("\nPredicted Sediment Transport Rate:", round(predicted_transport, 3))
