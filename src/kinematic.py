"""
kinematic.py
Baseline 2: Kinematic motion model utilizing reported Speed Over Ground (SOG)
and Course Over Ground (COG) telemetry.
Handles knots-to-m/s conversion and clockwise-from-North nautical bearing conventions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from src.coordinates import wgs84_to_local_enu, local_enu_to_wgs84

# Exact nautical conversion: 1 international knot = 1852 m / 3600 s
KNOTS_TO_METRES_PER_SECOND = 1852.0 / 3600.0  # 0.51444444...


class KinematicModel:
    """
    Kinematic Dead-Reckoning Model.
    Decomposes reported vessel Speed Over Ground (SOG) and Course Over Ground (COG)
    into local East-North velocity components.
    
    Coordinate Convention:
      - COG is clockwise from True North (0° = North, 90° = East, 180° = South, 270° = West)
      - Easting velocity:  vx = speed_mps * sin(radians(COG))
      - Northing velocity: vy = speed_mps * cos(radians(COG))
    """
    def __init__(self, smooth_window_points: int = 3):
        self.smooth_window_points = smooth_window_points

    def predict(
        self,
        history_df: pd.DataFrame,
        forecast_horizons_hours: list = [4.0, 6.0]
    ) -> Dict[float, Tuple[float, float]]:
        """
        Extracts recent SOG and COG, computes metric velocity components, and propagates forward.
        Returns dict: {horizon_hours: (predicted_lat, predicted_lon)}
        """
        hist = history_df.sort_values(by="base_date_time").copy()
        lat0 = float(hist["latitude"].iloc[-1])
        lon0 = float(hist["longitude"].iloc[-1])

        # Select recent observations for optional smoothing
        k = min(self.smooth_window_points, len(hist))
        recent_pts = hist.iloc[-k:]

        # Average recent SOG and COG (handling circular angle averaging for COG)
        sog_knots = float(recent_pts["sog"].mean())
        
        # Circular mean for Course Over Ground
        cog_rad = np.radians(recent_pts["cog"].values)
        sin_mean = np.mean(np.sin(cog_rad))
        cos_mean = np.mean(np.cos(cog_rad))
        mean_cog_rad = np.arctan2(sin_mean, cos_mean)
        if mean_cog_rad < 0:
            mean_cog_rad += 2 * np.pi

        # Convert speed from knots to metres/second
        speed_mps = sog_knots * KNOTS_TO_METRES_PER_SECOND

        # Decompose into East and North metric velocity (m/s)
        vx = speed_mps * np.sin(mean_cog_rad)
        vy = speed_mps * np.cos(mean_cog_rad)

        predictions = {}
        for h in forecast_horizons_hours:
            tau_sec = h * 3600.0
            x_pred = vx * tau_sec
            y_pred = vy * tau_sec
            lat_pred, lon_pred = local_enu_to_wgs84(x_pred, y_pred, lat0, lon0)
            predictions[h] = (float(lat_pred), float(lon_pred))

        return predictions
