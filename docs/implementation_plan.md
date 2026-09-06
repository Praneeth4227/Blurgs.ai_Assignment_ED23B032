# Implementation Plan: 4-6 Hour Vessel Trajectory Forecasting

**Project:** Where Will This Vessel Be in 4–6 Hours?   
**Repository:** `https://github.com/Praneeth4227/Blurgs.ai_Assignment_ED23B032`  

---

## 1. Problem Formulation
Given approximately 4 hours of historical AIS observations for an individual vessel up to forecast origin time $T$, our objective is to predict its geographic position (latitude and longitude on the WGS-84 ellipsoid) at two distinct future horizons:
- $T_1 = T + 4\text{ hours}$
- $T_2 = T + 6\text{ hours}$

Mathematically, let an input trajectory sample be represented as:
$$\mathcal{H}(T) = \left\{ (t_i, \text{lat}_i, \text{lon}_i, \text{sog}_i, \text{cog}_i) \mid T - 4\text{h} \le t_i \le T \right\}$$
where timestamps $t_i$ are irregularly spaced. The models must map $\mathcal{H}(T)$ to predicted coordinates:
$$\hat{\mathbf{p}}(T + 4\text{h}) = (\widehat{\text{lat}}_{T+4}, \widehat{\text{lon}}_{T+4}), \quad \hat{\mathbf{p}}(T + 6\text{h}) = (\widehat{\text{lat}}_{T+6}, \widehat{\text{lon}}_{T+6})$$
using strictly causal information ($t \le T$).

---

## 2. Dataset Selection
We select the official **NOAA MarineCadastre 2024 GeoParquet** dataset.
- **Source:** Bureau of Ocean Energy Management (BOEM) and NOAA MarineCadastre AIS archive.
- **Format:** Cloud-optimized GeoParquet with Apache Parquet column-chunk compression and WKB (Well-Known Binary) Point geometry encoding in WGS-84 (`EPSG:4326`).
- **Inspection Findings:**
  - Column schema: `mmsi` (INT32), `base_date_time` (TIMESTAMP/INT64), `sog` (FLOAT), `cog` (FLOAT), `heading` (INT32), `vessel_name` (STRING), `vessel_type` (INT32), `status` (INT32), `length` (FLOAT), `width` (INT32), `draft` (FLOAT), `geometry` (WKB Point).
  - High observation density: A single daily file contains over 7.2 million records across 7 row groups.
  - Columnar access: Specific column chunks (`mmsi`, `base_date_time`, `sog`, `cog`, `vessel_type`, `geometry`) can be accessed efficiently without reading unused metadata.

---

## 3. Geographic Region Selection
Rather than attempting to forecast globally or across arbitrary inshore channels where vessels stop every 20 minutes, we select the **Southeast US Coastal & Florida Straits Shipping Corridor**:
- **Bounding Box:**
  - Latitude: $24.0^\circ\text{N} \le \text{lat} \le 32.0^\circ\text{N}$
  - Longitude: $-85.0^\circ\text{W} \le \text{lon} \le -78.0^\circ\text{W}$
- **Rationale:**
  1. High volume of commercial traffic (container ships, bulk cargo carriers, crude/chemical tankers, ocean-going tugs).
  2. Long, continuous open-water transit legs extending far beyond 10 consecutive hours.
  3. Real operational challenges: Vessels transition from the Gulf of Mexico through the Florida Straits into the open Atlantic, navigating constrained corridors and Gulf Stream currents.

---

## 4. Time Period Selection
- **Selected Day:** January 1, 2024 (`ais-2024-01-01`).
- **Temporal Extent:** 00:00:00 UTC to 23:59:59 UTC (24 hours).
- **Adequacy:** A 24-hour observation window provides ample duration to establish:
  - 4 hours of continuous historical observations $[T-4\text{h}, T]$
  - 6 hours of future forecast horizon $[T, T+6\text{h}]$
  - Total minimum continuous tracking requirement per trajectory: $4 + 6 = 10\text{ hours}$.
  Multiple origin timestamps $T$ can be extracted between 04:00 UTC and 18:00 UTC.

---

## 5. Data Cleaning Pipeline
Raw AIS broadcasts are known to contain telemetry noise, GPS multipath errors, and stationary vessel clutter. We apply a strict multi-stage filter:
1. **Coordinate Validity:** Ensure $-90 \le \text{lat} \le 90$ and $-180 \le \text{lon} \le 180$, rejecting null or out-of-range coordinates.
2. **Deduplication:** Remove identical AIS transmissions where `(mmsi, timestamp)` are duplicates.
3. **Speed Filtering:** Filter out stationary or moored vessels ($\text{SOG} < 0.5\text{ knots}$) and physically unrealistic vessel speeds ($\text{SOG} > 40\text{ knots}$).
4. **Course Filtering:** Verify $0^\circ \le \text{COG} \le 360^\circ$.
5. **Temporal Gap Splitting:** If the interval between consecutive AIS reports for a given vessel exceeds 60 minutes, the trajectory is split into independent sub-trajectories rather than interpolated.

---

## 6. Trajectory Construction
- Vessels are grouped by unique `mmsi`.
- Records are sorted strictly chronologically by `base_date_time`.
- Each continuous track without gaps $> 60\text{ min}$ forms a valid trajectory.
- Minimum trajectory duration requirement: A trajectory must span at least 10 continuous hours to qualify for 4h-history and 6h-forecast evaluation.

---

## 7. Forecast-Window Construction
For each qualified vessel trajectory:
1. Choose a forecast origin timestamp $T$ such that:
   - Historical segment: Observations exist spanning $[T - 4\text{h}, T]$.
   - Minimum observations: At least 10 AIS messages in the history window to support fitting.
2. Ground truth targets:
   - Find the closest observation to $T + 4\text{h}$ within a tolerance of $\pm 15\text{ minutes}$.
   - Find the closest observation to $T + 6\text{h}$ within a tolerance of $\pm 15\text{ minutes}$.
3. Strict Causal Isolation: Feature extraction, velocity calculation, and Kalman filtering operate exclusively on data where $t \le T$. Future records are completely sequestered.

---

## 8. Train / Validation / Test Split & Leakage Prevention
- **Splitting Strategy:** Chronological split based on forecast origin time $T$:
  - Train: First 70% of the active origin window.
  - Validation: Next 15%.
  - Test: Final 15%.
- **Zero Leakage Rules:**
  - No random observation sampling (which would leak future trajectory points into past training).
  - No vessel-trajectory overlap between train and test evaluation windows.
  - Test set remains completely unseen until final benchmark generation.

---

## 9. Coordinate System & Metric Projections
Motion modeling directly in latitude/longitude degrees introduces severe latitudinal distortion:
$$1^\circ\text{ longitude} = 111.320 \times \cos(\text{lat})\text{ km}, \quad 1^\circ\text{ latitude} \approx 110.574\text{ km}$$
- **Local Metric Conversion:** For each trajectory window around origin $(\text{lat}_0, \text{lon}_0)$, we project geographic coordinates to a local East-North tangent plane $(x, y)$ in metres:
  $$x = R \cdot (\text{lon} - \text{lon}_0) \cdot \cos(\text{lat}_0) \cdot \frac{\pi}{180}$$
  $$y = R \cdot (\text{lat} - \text{lat}_0) \cdot \frac{\pi}{180}$$
  where $R \approx 6,371,000\text{ m}$.
- **Geodesic Inversion:** Model predictions $(\hat{x}, \hat{y})$ are inverted back to $(\widehat{\text{lat}}, \widehat{\text{lon}})$.
- **Evaluation:** Distance between predicted coordinates and true ground truth coordinates is computed using the exact geodesic distance on the WGS-84 ellipsoid (via `geopy` / Vincenty-Karney formula).

---

## 10. Baseline 1: Linear Extrapolation
- Computes empirical velocity over the recent history window (e.g., last 30–60 minutes of history up to $T$):
  $$v_x = \frac{x(T) - x(T - \Delta t)}{\Delta t}, \quad v_y = \frac{y(T) - y(T - \Delta t)}{\Delta t}$$
- Projects position linearly:
  $$\hat{x}(T + \tau) = x(T) + v_x \cdot \tau, \quad \hat{y}(T + \tau) = y(T) + v_y \cdot \tau$$
  for $\tau \in \{4\text{h}, 6\text{h}\}$.

---

## 11. Baseline 1 (cont.): Polynomial Extrapolation
- Fits independent degree-2 polynomials for $x(t)$ and $y(t)$ over the 4-hour history window:
  $$x(t) \approx a_x t^2 + b_x t + c_x, \quad y(t) \approx a_y t^2 + b_y t + c_y$$
- Evaluates at $t = T + 4\text{h}$ and $t = T + 6\text{h}$.
- We constrain the degree strictly to 2 to prevent catastrophic Runge-type polynomial explosion over 4–6 hour horizons.

---

## 12. Baseline 2: Kinematic Model (SOG / COG)
- Uses onboard Speed Over Ground ($\text{SOG}$, converted from knots to $\text{m/s}$: $1\text{ kn} \approx 0.514444\text{ m/s}$) and Course Over Ground ($\text{COG}$, nautical bearing clockwise from North):
  $$v_x = \text{SOG} \cdot \sin\left(\text{COG} \cdot \frac{\pi}{180}\right)$$
  $$v_y = \text{SOG} \cdot \cos\left(\text{COG} \cdot \frac{\pi}{180}\right)$$
- To mitigate GPS sensor noise, SOG and COG are optionally smoothed using a short rolling window over the latest observations before $T$.
- Propagates constant velocity forward to $T+4\text{h}$ and $T+6\text{h}$.

---

## 13. Additional Approach: 4D Discrete Kalman Filter
- **State Vector:** $\mathbf{x}_k = [x_k, y_k, v_{x,k}, v_{y,k}]^T$ in local metric coordinates.
- **Kinematic Transition:**
  $$\mathbf{F}_k = \begin{bmatrix} 1 & 0 & \Delta t_k & 0 \\ 0 & 1 & 0 & \Delta t_k \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}, \quad \mathbf{Q}_k = \text{Continuous white-noise acceleration discrete process noise}$$
- **Measurement Model:** GPS positions provide measurements $\mathbf{z}_k = [x_{\text{obs}, k}, y_{\text{obs}, k}]^T$:
  $$\mathbf{H} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix}, \quad \mathbf{R} = \sigma_{\text{pos}}^2 \mathbf{I}_2$$
- **Filtering & Extrapolation:** The filter sequentially updates on the irregular observations up to time $T$. At time $T$, the posterior state $\hat{\mathbf{x}}_{T|T}$ and error covariance $\mathbf{P}_{T|T}$ are propagated forward analytically without measurement updates:
  $$\hat{\mathbf{x}}_{T+\tau} = \mathbf{F}(\tau) \hat{\mathbf{x}}_{T|T}$$
  $$\mathbf{P}_{T+\tau} = \mathbf{F}(\tau) \mathbf{P}_{T|T} \mathbf{F}(\tau)^T + \mathbf{Q}(\tau)$$
- Provides both the point prediction and an analytic uncertainty covariance ellipse.

---

## 14. Evaluation Protocol & Metrics
All models are evaluated on the exact same test trajectories at horizons $\tau = +4\text{h}$ and $\tau = +6\text{h}$.
For each prediction $j$, we compute the geodesic distance $e_j = d_{\text{geodesic}}(\mathbf{p}_j, \hat{\mathbf{p}}_j)$ in kilometres.
We report:
- **Mean Error (km)**
- **Median Error (km)** (robust to outlier maneuvers)
- **RMSE (km)**
- **90th Percentile Error (P90, km)**
- **Subgroup Breakdown:** Performance stratified by `vessel_type` (e.g., Cargo vs. Tanker vs. Tug).

---

## 15. Visualization Suite
1. **Regional Maritime Overview:** Spatial density map of the chosen corridor.
2. **Trajectory Predictions:** Historical 4h track, forward ground truth, and competing model predictions (+4h, +6h).
3. **Comparative Horizon Plots:** Error boxplots and empirical cumulative distributions (CDFs) at +4h vs +6h.
4. **Kalman Uncertainty Ellipses:** 2-sigma (95%) confidence ellipses projected at +4h and +6h.
5. **Success and Failure Case Studies:** Side-by-side plots of representative straight-line cruises vs sharp waypoint turns.

---

## 16. Failure Analysis
We will analyze why large forecast errors occur:
- **Course Alterations:** Vessels following maritime navigation channels (e.g., rounding the Florida Keys) deviate from constant velocity.
- **Throttle Reductions & Pilot Stations:** Vessels slowing down on approach to ports (e.g., Port of Miami).
- **Sensor Jitter / Erratic AIS:** Outlier SOG/COG reports distorting kinematic baselines.

---

## 17. Uncertainty & Probabilistic Extensions
- Discussion of how Gaussian predictive distributions from the Kalman filter provide calibrated spatial error bounds.
- Comparison of constant-velocity extrapolation against route-conditioned priors, conformal prediction, and historical trajectory clustering.

---

## 18. Reproducibility Guarantee
- All code formatted in clean, modular Python modules under `src/` and reproducible step-by-step notebooks under `notebooks/`.
- Fixed random seeds and strict parameter tracking via `configs/config.yaml`.
- Absolutely no synthetic or fabricated numbers; all metrics generated directly from the execution pipeline on real AIS data.
