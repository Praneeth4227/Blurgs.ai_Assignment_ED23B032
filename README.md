# Where Will This Vessel Be in 4–6 Hours?
### AIS Vessel Trajectory Forecasting over Extended Horizons
**Candidate:** Praneeth (ED23B032)  
**Take-Home Assignment:** Blurgs.ai  

---

## Project Overview

When navigating open waters and coastal channels, vessels follow paths governed by recent kinematics, navigational corridors, traffic separation schemes, and operational goals. This project tackles a practical marine tracking problem:

> **Problem Statement:** Given approximately 4 hours of historical AIS observations for a vessel up to time $T$, predict its geographic position at:
> - $T + 4\text{ hours}$
> - $T + 6\text{ hours}$

Forecasting vessel coordinates 4 to 6 hours into the future using only standard kinematic and trajectory extrapolation methods is inherently challenging because vessels alter engine throttle, execute course changes around waypoints, and encounter tidal currents. This project implements a disciplined progression of interpretable, physics-based, and state-space models evaluated on real-world NOAA AIS data.

---

## Modelling Progression

Rather than jumping to complex black-box deep learning models that require enormous compute and obscure physics, the project systematically explores:

1. **Linear Extrapolation Baseline:** Estimates velocity over the recent history window and projects the vessel forward linearly.
2. **Polynomial Baseline:** Fits a degree-2 polynomial in local metric coordinates $(x, y)$ over time to capture gentle curvature or acceleration.
3. **Kinematic (SOG / COG) Model:** Decomposes speed-over-ground (SOG) and course-over-ground (COG) into metric velocity components $(v_x, v_y)$ on a local East-North tangent plane, with optional rolling smoothing.
4. **Discrete Kalman Filter (Constant Velocity):** Implements a 4D state-space model $[x, y, v_x, v_y]^T$ updating on irregular GPS observations, estimating velocity and covariance, and propagating state and uncertainty to $T+4\text{h}$ and $T+6\text{h}$.

---

## Repository Structure

```
├── README.md                           <- Project overview, methodology, results, reproduction
├── requirements.txt                    <- Python dependencies
├── .gitignore                          <- Git ignore rules for data and system artifacts
├── LICENSE                             <- MIT License
│
├── docs/
│   └── implementation_plan.md          <- Detailed research plan and technical decisions
│
├── configs/
│   └── config.yaml                     <- Experiment parameters, thresholds, and region bounds
│
├── src/                                <- Reusable Python modules
│   ├── __init__.py
│   ├── data_loader.py                  <- Streaming/loading raw AIS parquet and extracting subsets
│   ├── preprocessing.py                <- Deduplication, boundary cleaning, and anomaly filtering
│   ├── trajectories.py                 <- Trajectory segmentation and sliding window extraction
│   ├── coordinates.py                  <- WGS84 geodesic & local East-North metric conversions
│   ├── baselines.py                    <- Linear and polynomial trajectory extrapolation
│   ├── kinematic.py                    <- SOG/COG kinematic propagation
│   ├── kalman.py                       <- 4D state-space Kalman filter with uncertainty projection
│   ├── evaluation.py                   <- Geodesic error metrics (Mean, Median, RMSE, P90)
│   └── visualization.py                <- Spatial plotting, trajectory overlays, and error distributions
│
├── notebooks/                          <- Step-by-step reproducible analysis
│   ├── 01_data_exploration.ipynb       <- Raw AIS inspection, schema verification, temporal extent
│   ├── 02_data_cleaning.ipynb          <- Filtering noise, invalid coordinates, speed anomalies
│   ├── 03_trajectory_construction.ipynb<- 4h history windows and +4h/+6h target formulation
│   ├── 04_baseline_models.ipynb        <- Linear, polynomial, and kinematic baselines
│   ├── 05_kalman_model.ipynb           <- Kalman filter implementation and covariance tracking
│   └── 06_evaluation_and_analysis.ipynb<- Quantitative benchmark, vessel-type breakdown, failures
│
├── data/
│   ├── raw/                            <- Raw NOAA MarineCadastre data
│   └── processed/                      <- Cleaned trajectories and evaluation windows
│
└── results/
    ├── figures/                        <- Trajectory plots, error CDFs, failure cases
    ├── metrics/                        <- CSV/JSON evaluation metric summaries
    └── predictions/                    <- Model forecast outputs for test trajectories
```

---

## Dataset & Geographic Region

We use the official **NOAA MarineCadastre 2024 GeoParquet** dataset (Jan 1, 2024). The dataset records broadcast points across US coastal and offshore waters.

- **Selected Corridor:** Southeast US Atlantic Coast & Straits of Florida (Lat: $24.0^\circ\text{N} - 32.0^\circ\text{N}$, Lon: $-85.0^\circ\text{W} - -78.0^\circ\text{W}$).
- **Why this region?** This maritime corridor hosts major shipping lanes connecting Gulf Coast ports (e.g., Houston, New Orleans, Tampa) to East Coast hubs (Miami, Savannah, Charleston). It provides abundant commercial traffic (Cargo, Tankers, Tugs) with long, continuous underway voyages spanning 10+ hours, ideal for evaluating 4-hour history with 4-to-6 hour forecasting horizons.

---

## Current Status: Stage 1 Complete

- [x] Cloned and verified GitHub repository structure.
- [x] Inspected raw NOAA MarineCadastre GeoParquet schema, row groups, and WKB coordinate encoding.
- [x] Defined reproducible configuration and data hygiene thresholds in `configs/config.yaml`.
- [x] Authored complete research and implementation plan in `docs/implementation_plan.md`.
- [ ] Stage 2: Data discovery and exploratory inspection notebook.
- [ ] Stage 3: Data cleaning pipeline.
- [ ] Stage 4: Trajectory construction & window extraction.
- [ ] Stage 5: Chronological train/test split.
- [ ] Stage 6: Linear and polynomial baselines.
- [ ] Stage 7: Kinematic model.
- [ ] Stage 8: Kalman filter model.
- [ ] Stage 9: Comprehensive quantitative evaluation.
- [ ] Stage 10: Failure analysis and uncertainty visualization.
- [ ] Stage 11: Final documentation and audit.

*(Numerical metrics and benchmark tables will be updated directly from model test runs in subsequent stages without fabrication).*
