"""
kalman.py
Constant-Velocity Discrete Kalman Filter for AIS Trajectory Smoothing and Forward Extrapolation.
State vector: [x, y, vx, vy]^T in local metric tangent plane (metres and metres/second).
Provides forward point predictions and 2D spatial covariance matrices for uncertainty quantification.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from src.coordinates import wgs84_to_local_enu, local_enu_to_wgs84


class KalmanFilterTrajectoryModel:
    """
    4D Constant-Velocity Kalman Filter for irregular AIS observations.
    
    State vector:
      x = [x, y, vx, vy]^T
      x, y: East and North displacements from origin (metres)
      vx, vy: East and North velocity components (m/s)
    """
    def __init__(
        self,
        process_noise_accel: float = 0.05,  # Acceleration spectral density sigma_a (m/s^2)
        measurement_noise_pos: float = 20.0  # GPS measurement error sigma_pos (metres)
    ):
        self.q_var = process_noise_accel ** 2
        self.r_var = measurement_noise_pos ** 2
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float64)
        self.R = np.eye(2, dtype=np.float64) * self.r_var

    def _get_transition_matrix(self, dt: float) -> np.ndarray:
        return np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)

    def _get_process_noise_matrix(self, dt: float) -> np.ndarray:
        # Discrete continuous-white-noise acceleration covariance
        dt3 = (dt ** 3) / 3.0
        dt2 = (dt ** 2) / 2.0
        q = self.q_var
        return q * np.array([
            [dt3, 0.0, dt2, 0.0],
            [0.0, dt3, 0.0, dt2],
            [dt2, 0.0, dt,  0.0],
            [0.0, dt2, 0.0, dt ]
        ], dtype=np.float64)

    def fit_and_predict(
        self,
        history_df: pd.DataFrame,
        forecast_horizons_hours: list = [4.0, 6.0]
    ) -> Tuple[Dict[float, Tuple[float, float]], Dict[float, np.ndarray], Dict]:
        """
        Filters the irregular historical observations up to origin T, then extrapolates
        forward to forecast horizons.
        
        Returns:
          - predictions: {horizon_hours: (predicted_lat, predicted_lon)}
          - covariances: {horizon_hours: 2x2 position covariance matrix in metres^2}
          - filter_state: dict with final posterior state and velocity estimate
        """
        hist = history_df.sort_values(by="base_date_time").copy()
        lat0 = float(hist["latitude"].iloc[-1])
        lon0 = float(hist["longitude"].iloc[-1])
        t_origin = hist["base_date_time"].iloc[-1]

        # Convert to local metric tangent plane
        x_obs, y_obs = wgs84_to_local_enu(
            hist["latitude"].values, hist["longitude"].values, lat0, lon0
        )
        t_sec = (hist["base_date_time"] - t_origin).dt.total_seconds().values

        # Initial state estimation from first two observations (or first observation + SOG/COG)
        x_init = x_obs[0]
        y_init = y_obs[0]
        if len(t_sec) > 1 and (t_sec[1] - t_sec[0]) > 0:
            vx_init = (x_obs[1] - x_obs[0]) / (t_sec[1] - t_sec[0])
            vy_init = (y_obs[1] - y_obs[0]) / (t_sec[1] - t_sec[0])
        else:
            vx_init, vy_init = 0.0, 0.0

        state = np.array([x_init, y_init, vx_init, vy_init], dtype=np.float64)
        P = np.diag([self.r_var, self.r_var, 5.0**2, 5.0**2]).astype(np.float64)

        # Sequential Kalman Filter Update over history points
        for k in range(1, len(t_sec)):
            dt = t_sec[k] - t_sec[k - 1]
            if dt <= 0:
                continue

            # Predict
            F = self._get_transition_matrix(dt)
            Q = self._get_process_noise_matrix(dt)
            state_pred = F @ state
            P_pred = F @ P @ F.T + Q

            # Update
            z = np.array([x_obs[k], y_obs[k]], dtype=np.float64)
            y = z - (self.H @ state_pred)
            S = self.H @ P_pred @ self.H.T + self.R
            K = P_pred @ self.H.T @ np.linalg.inv(S)

            state = state_pred + (K @ y)
            I = np.eye(4, dtype=np.float64)
            P = (I - K @ self.H) @ P_pred

        # Final posterior state at origin T
        final_state = state.copy()
        final_P = P.copy()

        # Extrapolate forward
        predictions = {}
        covariances = {}

        for h in forecast_horizons_hours:
            tau_sec = h * 3600.0
            F_ext = self._get_transition_matrix(tau_sec)
            Q_ext = self._get_process_noise_matrix(tau_sec)

            state_ext = F_ext @ final_state
            P_ext = F_ext @ final_P @ F_ext.T + Q_ext

            lat_pred, lon_pred = local_enu_to_wgs84(state_ext[0], state_ext[1], lat0, lon0)
            predictions[h] = (float(lat_pred), float(lon_pred))
            covariances[h] = P_ext[0:2, 0:2]  # Position sub-covariance (East-North)

        filter_info = {
            "posterior_pos_m": (final_state[0], final_state[1]),
            "posterior_vel_mps": (final_state[2], final_state[3]),
            "posterior_speed_knots": np.hypot(final_state[2], final_state[3]) / 0.514444,
            "posterior_cov": final_P
        }

        return predictions, covariances, filter_info
