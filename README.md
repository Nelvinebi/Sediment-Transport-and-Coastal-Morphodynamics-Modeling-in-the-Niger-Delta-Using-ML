# Sediment Transport and Coastal Morphodynamics Modeling in the Niger Delta Using Machine Learning


![alt text](https://github.com/Nelvinebi/Sediment-Transport-and-Coastal-Morphodynamics-Modeling-in-the-Niger-Delta-Using-ML/blob/main/3D%20Morphohodynamic%20Response%20Surface.png?raw=true)

## Live Dashboard

Interactive coastal analytics dashboard:

https://(https://zxhebax27f63qpmy2d7xta.streamlit.app/)

A data-driven coastal modeling project that investigates sediment transport dynamics and shoreline morphodynamics in the Niger Delta using machine learning and synthetic oceanographic data.

The project simulates realistic coastal conditions, predicts sediment transport rates, and analyzes morphodynamic behavior such as erosion and accretion processes, supporting coastal environmental analysis and spatial decision-making.


📌 Project Overview

The Niger Delta coastline is highly dynamic, with sediment movement controlled by interactions between waves, tidal processes, ocean currents, and sediment characteristics. These processes influence shoreline stability, ecosystem health, and coastal infrastructure.

This project applies machine learning techniques to simulate and model sediment transport behavior using physically consistent synthetic coastal datasets.

By integrating multiple oceanographic and sediment variables, the workflow predicts sediment transport rates and morphodynamic responses, demonstrating how data science and environmental modeling can support coastal research and management.

🎯 Project Objectives

The main goals of this project include:

• Simulating realistic coastal and oceanographic forcing conditions for the Niger Delta
• Predicting sediment transport rates using machine learning
• Analyzing coastal morphodynamic responses including erosion and accretion patterns
• Identifying key environmental drivers influencing sediment transport
• Generating analytical visualizations that explain coastal sediment dynamics

🌊 Coastal Factors Considered

The model evaluates several environmental variables that influence sediment movement.

Feature	Description
Wave Height (m)	Represents wave energy impacting sediment transport
Wave Period (s)	Controls wave-induced sediment motion
Current Velocity (m/s)	Drives horizontal sediment displacement
Tidal Range (m)	Influences sediment redistribution through tidal fluctuations
Sediment Grain Size (mm)	Determines sediment mobility
Sediment Density (kg/m³)	Influences settling behavior of particles
Beach Slope (degrees)	Affects coastal erosion potential
Shoreline Orientation (degrees)	Determines exposure to wave energy
🧮 Methodology

The project follows a simplified coastal morphodynamic modeling workflow.

1️⃣ Synthetic Coastal Data Generation

A synthetic dataset was generated representing realistic ranges of hydrodynamic and sediment characteristics typical of the Niger Delta coastal environment.

The dataset includes multiple environmental variables influencing sediment transport.

2️⃣ Data Preprocessing

The dataset is prepared using:

• Feature scaling with Min–Max normalization
• Validation of variable ranges and data structure
• Preparation of predictor variables for machine learning modeling

3️⃣ Machine Learning Modeling

A Random Forest regression model is applied to predict sediment transport rates based on environmental forcing variables.

Random Forest is used because it:

• Handles nonlinear environmental relationships
• Captures interactions between coastal variables
• Provides interpretable feature importance analysis

4️⃣ Model Evaluation

Model performance is assessed using:

• Root Mean Square Error (RMSE)
• Coefficient of Determination (R²)

These metrics help evaluate how accurately the model predicts sediment transport rates.

📊 Data Visualizations

Several visualizations are generated to analyze sediment transport behavior and interpret model results.

1️⃣ 3D Morphodynamic Response Surface

This three-dimensional surface plot illustrates how key coastal forcing variables interact to influence sediment transport dynamics.

The plot highlights how variations in environmental conditions such as wave height and current velocity affect morphodynamic responses along the coastline.

![alt text](https://github.com/Nelvinebi/Sediment-Transport-and-Coastal-Morphodynamics-Modeling-in-the-Niger-Delta-Using-ML/blob/main/3D%20Morphohodynamic%20Response%20Surface.png?raw=true)

2️⃣ Key Drivers of Sediment Transport (Niger Delta)

This bar chart displays the relative importance of environmental variables influencing sediment transport predictions.

The results reveal which factors—such as wave energy, current velocity, or sediment properties—play the most significant roles in sediment movement along the Niger Delta coast.

![alt text](https://github.com/Nelvinebi/Sediment-Transport-and-Coastal-Morphodynamics-Modeling-in-the-Niger-Delta-Using-ML/blob/1ea58fb67c29783dfa16f659d6f4ed12718b0feb/Key%20Drivers%20of%20Sediment%20Transport%20(Niger%20Delta).png)



3️⃣ Scattered Plot of Predicted vs Actual Sediment Transport

This scatter plot compares machine learning model predictions with actual simulated sediment transport values.

Points closer to the diagonal line indicate better model accuracy, demonstrating how effectively the model captures sediment transport dynamics.

![alt text](https://github.com/Nelvinebi/Sediment-Transport-and-Coastal-Morphodynamics-Modeling-in-the-Niger-Delta-Using-ML/blob/1ea58fb67c29783dfa16f659d6f4ed12718b0feb/Scattered%20plot%20of%20Predicted%20vs%20Actual%20Sediment%20Transport.png)

4️⃣ Scenario Prediction vs Historical Sediment Transport

This visualization compares model predictions under simulated environmental scenarios with historical sediment transport behavior.

The chart helps assess how well the model reproduces realistic coastal sediment transport patterns.

![alt text](https://github.com/Nelvinebi/Sediment-Transport-and-Coastal-Morphodynamics-Modeling-in-the-Niger-Delta-Using-ML/blob/1ea58fb67c29783dfa16f659d6f4ed12718b0feb/Scenario%20Prediction%20Vs%20Historical%20Sediment%20Transport.png)

5️⃣ Sensitivity of Sediment Transport to Wave Height

This sensitivity analysis explores how changes in wave height affect sediment transport rates.

The visualization highlights the strong influence of wave energy on sediment mobility along coastal systems.

![alt text](https://github.com/Nelvinebi/Sediment-Transport-and-Coastal-Morphodynamics-Modeling-in-the-Niger-Delta-Using-ML/blob/1ea58fb67c29783dfa16f659d6f4ed12718b0feb/Sensitivity%20of%20Sediment%20Transport%20to%20Wave%20height.png)

6️⃣ Storm vs Calm Condition Transport Comparison

This comparison chart illustrates differences in sediment transport under storm conditions versus calm ocean states.

The results demonstrate how extreme wave conditions can significantly increase sediment redistribution and coastal erosion risks.

![alt text](https://github.com/Nelvinebi/Sediment-Transport-and-Coastal-Morphodynamics-Modeling-in-the-Niger-Delta-Using-ML/blob/1ea58fb67c29783dfa16f659d6f4ed12718b0feb/Storm%20vs%20Calm%20Condition%20Transport%20Comparison.png)

📊 Dataset

The dataset used in this project is stored as:

sediment_transport_morphodynamics_niger_delta_dataset.xlsx

The dataset contains:

• Oceanographic forcing variables
• Sediment physical properties
• Simulated sediment transport rates
• Derived morphodynamic indicators

Although synthetic, the dataset maintains physical consistency with coastal sediment transport processes.

📁 Repository Structure
sediment-transport-morphodynamics-niger-delta
│
├── data
│   └── sediment_transport_morphodynamics_niger_delta_dataset.xlsx
│
├── gis_outputs
│   ├── niger_delta_erosion_accretion.tif
│   └── niger_delta_erosion_accretion_zones.shp
│
├── sediment_transport_morphodynamics_ml.py
├── README.md
🚀 How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/your-username/sediment-transport-morphodynamics.git
cd sediment-transport-morphodynamics
2️⃣ Install Required Libraries
pip install pandas numpy matplotlib scikit-learn geopandas rasterio openpyxl
3️⃣ Run the Analysis Script
python sediment_transport_morphodynamics_ml.py
📈 Output

Running the script generates:

• Predicted sediment transport rates
• Morphodynamic response indicators
• Multiple analytical visualizations
• Structured datasets for further coastal analysis

🌍 Potential Applications

This project workflow can support:

• Coastal erosion monitoring
• Shoreline change analysis
• Climate change impact assessments on deltaic coastlines
• Coastal infrastructure risk assessment
• Environmental research and teaching demonstrations

🔮 Future Improvements

Potential future extensions include:

• Integration with satellite-derived oceanographic datasets
• Time-series analysis of shoreline evolution
• Advanced machine learning models such as XGBoost or deep learning
• Interactive coastal vulnerability maps
• Integration with remote sensing and GIS shoreline monitoring

👤 Author

Agbozu Ebingiye Nelvin
Environmental Data Scientist | GIS | Remote Sensing | Machine Learning

📧 Email: nelvinebingiye@gmail.com

🔗 GitHub: https://github.com/nelvinebi

🔗 LinkedIn: https://www.linkedin.com/in/agbozu-ebi/

⚠️ Disclaimer

This project uses synthetic environmental data for research, demonstration, and educational purposes. While the dataset reflects realistic coastal process relationships, it does not replace field observations or operational coastal modeling systems.

