"""
baselines.py
Baseline 1: Linear extrapolation and Polynomial extrapolation models.
Operates on local East-North tangent plane metric coordinates (metres)
and inverts predictions back to WGS-84 (lat/lon).
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from src.coordinates import wgs84_to_local_enu, local_enu_to_wgs84


class LinearExtrapolationModel:
    """
    Linear Extrapolation Baseline.
    Estimates recent velocity over a configurable history window (e.g., recent 30-45 minutes)
    and propagates constant velocity forward.
    """
    def __init__(self, recent_window_minutes: float = 30.0):
        self.recent_window_minutes = recent_window_minutes

    def predict(
        self,
        history_df: pd.DataFrame,
        forecast_horizons_hours: list = [4.0, 6.0]
    ) -> Dict[float, Tuple[float, float]]:
        """
        Fits linear velocity on recent history points and extrapolates.
        Returns dict: {horizon_hours: (predicted_lat, predicted_lon)}
        """
        hist = history_df.sort_values(by="base_date_time").copy()
        t_origin = hist["base_date_time"].iloc[-1]
        lat0 = float(hist["latitude"].iloc[-1])
        lon0 = float(hist["longitude"].iloc[-1])

        # Metric projection on local tangent plane
        x, y = wgs84_to_local_enu(
            hist["latitude"].values, hist["longitude"].values, lat0, lon0
        )
        t_seconds = (hist["base_date_time"] - t_origin).dt.total_seconds().values

        # Filter recent observations for velocity estimation
        recent_mask = t_seconds >= (-self.recent_window_minutes * 60.0)
        # Ensure at least 2 points for regression; if not, use last 5 points or all points
        if recent_mask.sum() < 2:
            recent_mask = np.zeros(len(t_seconds), dtype=bool)
            recent_mask[-min(5, len(t_seconds)):] = True

        t_rec = t_seconds[recent_mask]
        x_rec = x[recent_mask]
        y_rec = y[recent_mask]

        # Robust linear velocity estimation via 1st degree polynomial / OLS
        # x(t) = vx * t + x0, y(t) = vy * t + y0
        poly_x = np.polyfit(t_rec, x_rec, deg=1)
        poly_y = np.polyfit(t_rec, y_rec, deg=1)

        vx, x0 = poly_x[0], poly_x[1]
        vy, y0 = poly_y[0], poly_y[1]

        predictions = {}
        for h in forecast_horizons_hours:
            tau_sec = h * 3600.0
            x_pred = vx * tau_sec + x0
            y_pred = vy * tau_sec + y0
            lat_pred, lon_pred = local_enu_to_wgs84(x_pred, y_pred, lat0, lon0)
            predictions[h] = (float(lat_pred), float(lon_pred))

        return predictions


class PolynomialExtrapolationModel:
    """
    Polynomial Extrapolation Baseline.
    Fits a low-degree polynomial (degree 2) to capture gentle trajectory curvature.
    Higher-degree polynomials are avoided due to the instability of higher-order polynomial extrapolation outside the observed interval.
    """
    def __init__(self, degree: int = 2):
        self.degree = min(degree, 2)  # Strictly enforce degree <= 2

    def predict(
        self,
        history_df: pd.DataFrame,
        forecast_horizons_hours: list = [4.0, 6.0]
    ) -> Dict[float, Tuple[float, float]]:
        """
        Fits quadratic polynomial x(t) and y(t) centered at origin T.
        Returns dict: {horizon_hours: (predicted_lat, predicted_lon)}
        """
        hist = history_df.sort_values(by="base_date_time").copy()
        t_origin = hist["base_date_time"].iloc[-1]
        lat0 = float(hist["latitude"].iloc[-1])
        lon0 = float(hist["longitude"].iloc[-1])

        x, y = wgs84_to_local_enu(
            hist["latitude"].values, hist["longitude"].values, lat0, lon0
        )
        # Time normalized to hours relative to origin: t in [-4.0, 0.0]
        t_hours = (hist["base_date_time"] - t_origin).dt.total_seconds().values / 3600.0

        # Fit degree-2 polynomial: p(t) = c2 * t^2 + c1 * t + c0
        deg = 2 if len(t_hours) >= 3 else 1
        poly_x = np.polyfit(t_hours, x, deg=deg)
        poly_y = np.polyfit(t_hours, y, deg=deg)

        predictions = {}
        for h in forecast_horizons_hours:
            x_pred = np.polyval(poly_x, h)
            y_pred = np.polyval(poly_y, h)
            lat_pred, lon_pred = local_enu_to_wgs84(x_pred, y_pred, lat0, lon0)
            predictions[h] = (float(lat_pred), float(lon_pred))

        return predictions
