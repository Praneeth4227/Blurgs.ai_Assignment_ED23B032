# Where Will This Vessel Be in 4–6 Hours?
### AIS Vessel Trajectory Forecasting over Extended Horizons
**Candidate:** Praneeth (ED23B032)  
**Take-Home Assignment:** Blurgs.ai  
**Repository:** [Blurgs.ai_Assignment_ED23B032](https://github.com/Praneeth4227/Blurgs.ai_Assignment_ED23B032)  

---

## 1. Problem Formulation & Objective

Vessel movement is physically constrained by recent momentum, vessel dimensions, navigational corridors, bathymetry, traffic separation schemes, and operational destinations. In this project, we address the following real-world marine tracking task:

> **Goal:** Given approximately **4 hours of historical AIS observations** for a vessel up to origin time $T$, predict its geographic coordinate $(\text{latitude}, \text{longitude})$ at:
> - **Horizon 1:** $+4\text{ hours}$ ($T + 4\text{h}$)
> - **Horizon 2:** $+6\text{ hours}$ ($T + 6\text{h}$)

Predicting vessel coordinates 4 to 6 hours into the future using historical telemetry alone is challenging: vessels frequently alter course around waypoints, adjust engine throttle, and maneuver into coastal fairways. Rather than building opaque neural networks with excessive parameters, this project follows an interpretable engineering progression:
1. Data understanding & schema inspection
2. Quality cleaning & anomaly filtering
3. Trajectory segmentation & gap-splitting
4. Leakage-safe chronological formulation
5. Linear velocity extrapolation baseline
6. Quadratic polynomial extrapolation baseline
7. Kinematic SOG/COG dead-reckoning baseline
8. 4D constant-velocity Discrete Kalman Filter
9. Scientifically valid geodesic evaluation on the WGS-84 ellipsoid
10. Failure analysis & spatial uncertainty quantification

---

## 2. Dataset & Maritime Corridor Selection

We use official **NOAA MarineCadastre 2024 GeoParquet** data (`ais-2024-01-01`).

### Dataset Discovery & Schema:
- **Daily Scale:** 7,293,408 broadcast points across the United States offshore and coastal zones.
- **Key Columns:** `mmsi` (vessel identifier), `base_date_time` (UTC timestamp), `sog` (knots), `cog` (degrees clockwise from North), `heading` (degrees), `vessel_type` (ITU categorization), `status` (navigational status), `geometry` (Point encoded in 21-byte Well-Known Binary in WGS-84).
- **Sampling Nature:** Irregular temporal intervals with a median spacing of **178 seconds (~3 minutes)**, with long communication gaps when vessels sail out of coastal receiver line-of-sight.

### Geographic Corridor:
We selected the **Southeast US Atlantic Coast & Straits of Florida**:
- **Bounding Box:** Latitude $24.0^\circ\text{N} - 32.0^\circ\text{N}$, Longitude $-85.0^\circ\text{W} - -78.0^\circ\text{W}$.
- **Why this region?** This maritime corridor connects Gulf of Mexico ports (Houston, New Orleans, Tampa) to North Atlantic shipping channels. It features heavy commercial traffic (Cargo container ships, Tankers, Ocean-going Tugs) with long open-water transit legs extending beyond 12–24 continuous hours.

![Spatial Distribution](results/figures/eda_spatial_distribution.png)

---

## 3. Data Cleaning & Trajectory Construction Pipeline

Raw AIS broadcasts contain transponder glitches, duplicated seconds, and stationary clutter. We apply a strict multi-stage filtering pipeline:

1. **Coordinate Verification:** Require $-90 \le \text{lat} \le 90$ and $-180 \le \text{lon} \le 180$.
2. **Deduplication:** Dropped 15 identical duplicate transmission packets matching `(mmsi, timestamp)`.
3. **Speed Filtering:** Retained underway vessels ($0.5 \le \text{SOG} \le 40.0\text{ knots}$), eliminating 636,520 stationary moored/anchored points and unrealistic GPS jump anomalies ($>40\text{ kn}$).
4. **Course Filtering:** Verified $0^\circ \le \text{COG} \le 360^\circ$.
5. **Observation Gap Handling:** **We do not interpolate across large gaps.** If consecutive reports for a vessel exceed **1.0 hour**, the trajectory is split into independent sub-tracks.
6. **Usable Trajectory Criterion:** A trajectory must span at least **10.0 continuous hours** with $\ge 20$ observations to support a 4-hour historical window plus a 6-hour forecast window ($4 + 6 = 10\text{h}$).

### Preprocessing Waterfall:
```
Raw Records in Corridor:       816,653 (100.0%)
  -> Valid Coordinates:        816,653 (100.0%)
  -> Deduplicated:             816,638 (99.9%)
  -> Speed Filtered (>=0.5kn): 180,118 (22.1%)
  -> Course Filtered:          175,037 (21.4%)
  -> Usable Trajectory Points:  60,976 (7.5%) across 101 continuous voyages (>= 10h)
```

![Preprocessing Funnel](results/figures/preprocessing_funnel.png)

---

## 4. Leakage-Safe Formulation & Local Metric Coordinates

### Zero-Leakage Chronological Split
To prevent lookahead leakage, windows are partitioned strictly chronologically based on forecast origin timestamp $T$:
- **Train Set (70%):** 295 windows ($T \in [04:00, 13:00\text{ UTC}]$)
- **Validation Set (15%):** 63 windows ($T \in (13:00, 15:15\text{ UTC}]$)
- **Test Set (15%):** 64 windows ($T \in [15:23, 17:57\text{ UTC}]$)

The test set represents the actual chronological future and remained completely sequestered until the final quantitative evaluation.

![Window Split Timeline](results/figures/trajectory_window_split.png)

### Local Metric Tangent Plane
To avoid spherical degree distortion ($1^\circ$ longitude varies with latitude), all modelling is executed on a local East-North tangent plane $(x, y)$ in metres centered at each trajectory origin $(\text{lat}_0, \text{lon}_0)$. Predictions are analytically inverted back to $(\widehat{\text{lat}}, \widehat{\text{lon}})$ with sub-millimeter precision.

---

## 5. Models Evaluated

1. **Linear Velocity Baseline:** Estimates recent velocity $(v_x, v_y)$ via linear regression over the recent 30-minute history and propagates constant velocity forward.
2. **Polynomial Baseline (Degree 2):** Fits quadratic polynomials $x(t)$ and $y(t)$ centered at origin $T$ over the 4-hour window. Degree is strictly limited to 2 to prevent Runge's phenomenon.
3. **Kinematic Baseline (SOG/COG):** Decomposes onboard transponder Speed Over Ground (knots converted to m/s) and Course Over Ground (bearing clockwise from North) into metric velocity components, with circular mean smoothing over recent observations.
4. **4D Discrete Kalman Filter:** State vector $\mathbf{x} = [x, y, v_x, v_y]^T$ under constant-velocity kinematics and continuous white-noise acceleration ($\sigma_a = 0.05\text{ m/s}^2, \sigma_{\text{pos}} = 20\text{ m}$). Sequentially filters irregular historical GPS measurements and propagates state and analytical covariance ellipses forward to $+4\text{h}$ and $+6\text{h}$.

---

## 6. Final Benchmark Results

All models were evaluated on the **64 unseen test trajectories** using true geodesic distances (Vincenty / Karney WGS-84 formula) in kilometres.

### Overall Benchmark Table:

| Model | Horizon | Mean (km) | Median (km) | RMSE (km) | P90 (km) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear** | **+4h** | **17.18** | **8.42** | **29.35** | **48.85** |
| **Linear** | **+6h** | **30.70** | **15.03** | **50.86** | **78.49** |
| **Polynomial (d=2)** | +4h | 30.43 | 15.52 | 51.57 | 70.31 |
| **Polynomial (d=2)** | +6h | 57.32 | 31.38 | 98.12 | 141.02 |
| **Kinematic (SOG/COG)** | +4h | 22.54 | 18.04 | 30.41 | 58.20 |
| **Kinematic (SOG/COG)** | +6h | 38.41 | 30.59 | 51.41 | 94.94 |
| **Kalman Filter** | **+4h** | **20.44** | **16.47** | **29.17** | **56.36** |
| **Kalman Filter** | **+6h** | **35.37** | **29.10** | **49.76** | **95.66** |

![Model Comparison Horizons](results/figures/model_comparison_horizons.png)
![Error CDF Distributions](results/figures/error_distributions_ecdf.png)

### Key Performance Insights:
- **Linear Extrapolation** achieved the lowest median error (**8.42 km at +4h**, **15.03 km at +6h**). Estimating velocity over recent 30-minute position differences provides a clean, robust velocity vector.
- **The Kalman Filter** produced the lowest overall RMSE (**29.17 km at +4h**, **49.76 km at +6h**), outperforming the raw kinematic transponder baseline because it filters out transponder heading jitter.
- **Polynomial Extrapolation** performed significantly worse (**RMSE 51.57 km at +4h, 98.12 km at +6h**). Quadratic terms accelerate outward rapidly during 4–6 hour extrapolation, amplifying small curvatures in history into large overshoot errors.

---

## 7. Performance Stratified by Vessel Category

| Vessel Category | Model | Horizon | Mean (km) | Median (km) | RMSE (km) | P90 (km) | Samples |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cargo** | Linear | +4h | 17.09 | **7.18** | 25.17 | 40.12 | 15 |
| **Cargo** | Kalman Filter | +4h | 18.72 | **7.20** | 29.21 | 40.35 | 15 |
| **Tug / Towing** | Linear | +4h | 8.81 | **5.78** | 11.74 | 18.32 | 5 |
| **Tug / Towing** | Kalman Filter | +4h | 10.16 | **5.72** | 13.62 | 21.35 | 5 |
| **Tanker** | Linear | +4h | 35.15 | 34.04 | 41.22 | 62.77 | 9 |
| **Tanker** | Kalman Filter | +4h | 33.75 | 34.39 | 39.90 | 59.08 | 9 |

![Vessel Type Performance](results/figures/vessel_type_performance.png)

- **Cargo ships and Tugs:** Exhibit highly predictable, low-error tracks (median error ~5.7–7.2 km at +4h). Cargo ships cruise at fixed throttle down deep-water lanes, while tugs operate at low speed (6–10 kn), reducing spatial displacement.
- **Tankers:** Produced higher median errors (~34 km at +4h) because tankers in this corridor navigate coastal approaches into petroleum terminals (Port Everglades, Miami, Tampa), executing course and speed alterations.

---

## 8. Failure Analysis: Why Do Models Fail?

Examining large-error predictions ($>80\text{ km}$) reveals three primary root causes:
1. **Waypoint Navigational Turns:** Vessels in the Florida Straits follow marine corridors that bend around the Florida Keys. Constant-velocity extrapolation projects straight into open water, leading to divergence once the ship turns $60^\circ - 90^\circ$.
2. **Port Approach Deceleration:** Commercial vessels entering harbour limits reduce throttle from 18 knots to 4 knots to await pilots, causing dead-reckoning models to overshoot.
3. **Polynomial Acceleration Divergence:** Quadratic polynomials amplify even subtle historical accelerations, leading to runaway divergence at $+6\text{h}$.

![Success and Failure Case Studies](results/figures/case_studies_success_failure.png)

---

## 9. Spatial Uncertainty & Probabilistic Extensions

Because vessel operational intent is unobserved in historical AIS data, point forecasts are incomplete. The Kalman filter provides an analytical covariance matrix $\mathbf{P}_{T+\tau}$, defining a 95% confidence ellipse that expands naturally as uncertainty accumulates over time.

![Kalman Uncertainty Ellipses](results/figures/kalman_uncertainty_ellipse.png)

### Recommended Next Steps for Research:
1. **Route-Conditioned Priors (Historical Trajectory Clustering):** Cluster historical vessel routes using Fréchet distance. Given 4 hours of history, match the track to the most probable maritime corridor prior, constraining predictions to navigational channels.
2. **Conformal Prediction:** Use validation set residuals to generate distribution-free spatial prediction sets with exact finite-sample coverage guarantees.
3. **Gaussian Mixture Trajectory Models (GMM):** Model multi-modal branch points where shipping lanes diverge (e.g. northbound Atlantic vs eastbound Caribbean).

---

## 10. How to Reproduce

### 1. Prerequisites
- Python 3.10+
- Git

### 2. Setup Environment
```bash
git clone https://github.com/Praneeth4227/Blurgs.ai_Assignment_ED23B032.git
cd Blurgs.ai_Assignment_ED23B032
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Step-by-Step Notebook Execution
Open and run the notebooks in sequential order:
```bash
jupyter lab
```
1. `notebooks/01_data_exploration.ipynb`: Inspects raw NOAA AIS data schema and distributions.
2. `notebooks/02_data_cleaning.ipynb`: Filters stationary clutter, anomalies, and splits on gaps.
3. `notebooks/03_trajectory_construction.ipynb`: Formulates 4h history windows and +4h/+6h targets.
4. `notebooks/04_baseline_models.ipynb`: Evaluates linear, polynomial, and kinematic baselines.
5. `notebooks/05_kalman_model.ipynb`: Demonstrates 4D Kalman filtering and covariance ellipses.
6. `notebooks/06_evaluation_and_analysis.ipynb`: Runs benchmark on the test set, generates tables, figures, and failure case studies.
