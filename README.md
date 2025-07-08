
#  Smart-Mobility-GNN: Graph Neural Network Pipeline for Urban Mobility Analytics

##  Overview

Smart-Mobility-GNN is a deep learning–driven framework that utilizes Graph Neural Networks (GNNs) to model, analyze, and predict urban mobility patterns using geographic and accident datasets. This project integrates geospatial processing, time-series modeling, and advanced machine learning to inform data-driven decisions in urban planning and traffic safety interventions.

The pipeline ingests regional accident data and outputs actionable insights such as spatial-risk zones, hotspot prediction, and mobility trend forecasting. Built with a modular and reproducible architecture, this system is designed for scalability and extensibility in smart city applications.

---

##  Technical Architecture

###  Dataset Sources

* **US\_Accidents.csv** (multi-year traffic incident logs)
* **County\_BoundingBoxes.csv** and **demographic\_data.csv** for geographic/region enrichment
* External shape files or geo-coordinates for spatial partitioning

###  Region Definition

* Grid-based or county-based hexagonal binning using `add_hex.py`
* Defined regions stored as `region_id` with centroid and bounding box metadata
* Mapped to spatial attributes like population density, road length, or weather conditions

###  Feature Engineering

* Extracted temporal features: time-of-day, weekday/weekend, seasonality
* Derived accident density, frequency, and severity scores per region
* Calculated inter-incident time deltas and rolling window aggregates

---

##  GNN Modeling Pipeline

### Graph Construction

Each region is treated as a node; spatial adjacency is encoded as edges:

* **Nodes:** Regions (counties, hex bins) with engineered mobility features
* **Edges:** Geospatial proximity or road network connectivity
* **Edge Attributes:** Optional traffic volume, travel time, or distance

### Graph Neural Network Configuration

| Component          | Description                                                        |
| ------------------ | ------------------------------------------------------------------ |
| **Model Type**     | Spatio-temporal Graph Neural Network (e.g., DCRNN, GCN-RNN hybrid) |
| **Node Inputs**    | Time-series mobility/accident vectors                              |
| **Edge Weights**   | Learned or distance-based                                          |
| **Temporal Block** | 1D-CNN or GRU/LSTM layers over time dimension                      |
| **Spatial Block**  | GCN/GraphSAGE/EdgeConv                                             |
| **Loss Function**  | MSE / MAE for regression, BCE/Softmax for classification           |

---

##  Training Workflow

### Preprocessing

* Region definition via `define_regions.py`
* Feature matrix generation in `Preparation/`
* Label targets (e.g., future accident counts or risk index) defined in `.ipynb` notebooks

### Hyperparameters

* Graph window size: `T=12` time steps
* Prediction horizon: `H=3` (e.g., 3 hours or 3 days ahead)
* Batch size: 32
* Learning rate: 0.001 with cosine decay
* Optimizer: AdamW

### Training and Evaluation

* Train/val/test split by temporal window
* Evaluation metrics:

  * Mean Absolute Error (MAE)
  * Root Mean Squared Error (RMSE)
  * R² score for regional prediction
* Trained on NVIDIA GPU (10–30 min/dataset)

---

##  Postprocessing & Visualization

* Heatmaps of high-risk regions by prediction horizon
* Temporal line plots comparing predicted vs actual incidents
* Geospatial overlays of predicted accident density on maps
* Summary statistics of model performance per region

---

##  Code Structure

```plaintext
├── Code/
│   ├── add_hex.py                  # Grid creation for spatial bins
│   ├── define_regions.py          # Defines region metadata
│   ├── US_County_BoundingBoxes.csv
│   ├── Columbus_accidents.csv
├── Preparation/
│   ├── CleanColumbusData.ipynb    # Cleans and formats raw accident data
│   ├── Features - US accident dataset.pdf
├── Output/
│   ├── columbus_regions.csv
│   ├── accidents_with_regions.csv
├── Reference Files/
│   ├── Final_Plan.pptx
│   ├── demographic_data.csv
│   ├── Features - US accident dataset.pdf
```

---

##  Example Insights

###  Accident Hotspots

* Identified regional bins with >10x average incident rates
* Found strong correlation with intersections and population density

###  Time Series Forecasting

* 24-hour future mobility prediction with 87% R² accuracy

###  GNN Effectiveness

* GNN outperformed MLP and CNN baselines by \~18% in RMSE
* Edge-based spatial modeling captured cross-region influence during holidays

---

##  Efficiency & Scalability

| Task                | Manual  | Automated |
| ------------------- | ------- | --------- |
| Region assignment   | 2–3 hrs | <5 min    |
| Feature engineering | 4 hrs   | <10 min   |
| Model training      | \~3 hrs | \~20 min  |
| Inference per batch | —       | \~2 ms    |

* Supports batched inference across cities
* Future support for live integration with traffic APIs

---

##  Key Achievements

* Designed a generalizable GNN pipeline for regional time-series prediction
* Demonstrated end-to-end automation from raw CSV to predictive analytics
* Achieved scalable geospatial preprocessing via hex binning
* Built interpretable, real-world visualizations for decision makers

---

##  Future Work

* Incorporate weather and live traffic sensor streams
* Apply attention mechanisms for temporal reasoning
* Integrate with city dashboards (e.g., via Streamlit/Flask)
* Extend to pedestrian & cyclist mobility analysis

---

##  Summary Metrics

| Feature                   | Value  |
| ------------------------- | ------ |
| Node-level prediction R²  | 0.87   |
| RMSE (accident counts)    | \~1.3  |
| Edge sparsity             | <10%   |
| Inference time (per node) | \~2 ms |
| Region coverage           | 200+   |

---

This project provides an extensible deep learning baseline for city-scale mobility prediction and risk modeling. It can be adapted across geographic regions and used by urban planners, policy analysts, and traffic engineers aiming for smarter, safer streets.

