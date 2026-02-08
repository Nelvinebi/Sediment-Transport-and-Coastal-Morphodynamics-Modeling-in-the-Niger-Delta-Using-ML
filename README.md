Sediment Transport and Coastal Morphodynamics Modeling in the Niger Delta Using Machine Learning
Overview

This project models sediment transport dynamics and coastal morphodynamic behavior in the Niger Delta using machine learning and realistic synthetic oceanographic data. The study integrates wave, tidal, current, and sediment characteristics to predict sediment transport rates and spatial erosion–accretion patterns for coastal management applications.

Objectives

Simulate realistic coastal and oceanographic forcing conditions

Predict sediment transport rates using machine learning

Assess coastal morphodynamic behavior (erosion and accretion)

Generate GIS-ready outputs for spatial coastal analysis

Methodology

Synthetic datasets representing wave climate, tidal dynamics, sediment properties, and shoreline characteristics were generated based on established coastal process relationships. A Random Forest regression model was trained to predict sediment transport rates. Model outputs were spatially represented using GIS-compatible raster and vector formats.

Dataset

The dataset includes:

Wave height and period

Current velocity

Tidal range

Sediment grain size and density

Beach slope

Shoreline orientation

Sediment transport rate (target variable)

Morphodynamic index

Format: Excel (.xlsx)
Nature: Synthetic but physically consistent

Machine Learning Model

Algorithm: Random Forest Regressor

Feature scaling: Min–Max normalization

Evaluation metrics: RMSE, R² score

Feature importance analysis performed

GIS Outputs

GeoTIFF: Erosion–Accretion intensity raster

Shapefile: Classified erosion and accretion zones

Coordinate Reference System: WGS 84 (EPSG:4326)

Compatible with QGIS and ArcGIS

Project Structure
├── data/
│   └── sediment_transport_morphodynamics_niger_delta_dataset.xlsx
├── gis_outputs/
│   ├── niger_delta_erosion_accretion.tif
│   └── niger_delta_erosion_accretion_zones.shp
├── sediment_transport_morphodynamics_ml.py
├── README.md

Applications

Coastal erosion monitoring

Shoreline change analysis

Climate change impact assessment

Coastal infrastructure planning

Environmental management and policy support

Requirements

Python 3.9+

NumPy

Pandas

Scikit-learn

Matplotlib

Rasterio

GeoPandas

Author

AGBOZU EBINGIYE NELVIN

Disclaimer

This project uses synthetic data for research, demonstration, and educational purposes. It is designed to replicate realistic coastal processes but does not replace field observations or operational coastal models.
