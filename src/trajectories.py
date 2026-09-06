"""
trajectories.py
Trajectory window formulation and leakage-safe chronological dataset splitting.
Extracts 4-hour historical input observations and links actual positions at +4h and +6h.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from typing import List, Dict, Tuple


class ForecastingWindow:
    """Encapsulates a single causal forecasting sample."""
    def __init__(
        self,
        sample_id: str,
        trajectory_id: str,
        mmsi: int,
        vessel_type: int,
        origin_time: pd.Timestamp,
        history_df: pd.DataFrame,
        target_4h: Dict,
        target_6h: Dict
    ):
        self.sample_id = sample_id
        self.trajectory_id = trajectory_id
        self.mmsi = mmsi
        self.vessel_type = vessel_type
        self.origin_time = origin_time
        self.history_df = history_df.copy()
        self.target_4h = target_4h
        self.target_6h = target_6h

    @property
    def origin_lat(self) -> float:
        return float(self.history_df.iloc[-1]["latitude"])

    @property
    def origin_lon(self) -> float:
        return float(self.history_df.iloc[-1]["longitude"])

    @property
    def origin_sog(self) -> float:
        return float(self.history_df.iloc[-1]["sog"])

    @property
    def origin_cog(self) -> float:
        return float(self.history_df.iloc[-1]["cog"])

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "trajectory_id": self.trajectory_id,
            "mmsi": self.mmsi,
            "vessel_type": self.vessel_type,
            "origin_time": self.origin_time,
            "origin_lat": self.origin_lat,
            "origin_lon": self.origin_lon,
            "origin_sog": self.origin_sog,
            "origin_cog": self.origin_cog,
            "history_num_points": len(self.history_df),
            "target_4h_time": self.target_4h["time"],
            "target_4h_lat": self.target_4h["lat"],
            "target_4h_lon": self.target_4h["lon"],
            "target_4h_dt_min": self.target_4h["dt_minutes"],
            "target_6h_time": self.target_6h["time"],
            "target_6h_lat": self.target_6h["lat"],
            "target_6h_lon": self.target_6h["lon"],
            "target_6h_dt_min": self.target_6h["dt_minutes"]
        }


def extract_forecasting_windows(
    df: pd.DataFrame,
    history_hours: float = 4.0,
    forecast_horizons: List[float] = [4.0, 6.0],
    min_history_points: int = 10,
    history_span_tolerance_hours: float = 0.5,
    target_tolerance_minutes: float = 15.0,
    origin_step_hours: float = 1.5
) -> List[ForecastingWindow]:
    """
    Extracts strictly causal forecasting windows [T - history_hours, T] and future targets
    at T + 4h and T + 6h.
    """
    windows = []
    df = df.sort_values(by=["trajectory_id", "base_date_time"]).copy()
    trajectories = df.groupby("trajectory_id")

    sample_counter = 0

    for traj_id, traj in trajectories:
        t_min = traj["base_date_time"].min()
        t_max = traj["base_date_time"].max()
        max_horizon = max(forecast_horizons)
        
        total_required_hours = history_hours + max_horizon
        if (t_max - t_min).total_seconds() / 3600.0 < total_required_hours:
            continue

        earliest_origin = t_min + timedelta(hours=history_hours)
        latest_origin = t_max - timedelta(hours=max_horizon)

        current_origin = earliest_origin
        while current_origin <= latest_origin:
            t_hist_start = current_origin - timedelta(hours=history_hours)
            hist = traj[
                (traj["base_date_time"] >= t_hist_start) &
                (traj["base_date_time"] <= current_origin)
            ].copy()

            if len(hist) < min_history_points:
                current_origin += timedelta(hours=origin_step_hours)
                continue

            actual_hist_span = (hist["base_date_time"].max() - hist["base_date_time"].min()).total_seconds() / 3600.0
            if actual_hist_span < (history_hours - history_span_tolerance_hours):
                current_origin += timedelta(hours=origin_step_hours)
                continue

            last_hist_gap_min = (current_origin - hist["base_date_time"].max()).total_seconds() / 60.0
            if last_hist_gap_min > target_tolerance_minutes:
                current_origin += timedelta(hours=origin_step_hours)
                continue

            targets = {}
            target_valid = True

            for h in forecast_horizons:
                nominal_t = current_origin + timedelta(hours=h)
                time_diffs = (traj["base_date_time"] - nominal_t).abs()
                min_idx = time_diffs.idxmin()
                best_match = traj.loc[min_idx]
                dt_min = time_diffs.loc[min_idx].total_seconds() / 60.0

                if dt_min > target_tolerance_minutes:
                    target_valid = False
                    break

                targets[h] = {
                    "time": best_match["base_date_time"],
                    "lat": float(best_match["latitude"]),
                    "lon": float(best_match["longitude"]),
                    "dt_minutes": float(dt_min)
                }

            if not target_valid:
                current_origin += timedelta(hours=origin_step_hours)
                continue

            sample_counter += 1
            sample_id = f"WIN_{sample_counter:04d}"
            mmsi = int(hist["mmsi"].iloc[0])
            vessel_type = int(hist["vessel_type"].iloc[0]) if pd.notnull(hist["vessel_type"].iloc[0]) else 0

            window = ForecastingWindow(
                sample_id=sample_id,
                trajectory_id=traj_id,
                mmsi=mmsi,
                vessel_type=vessel_type,
                origin_time=current_origin,
                history_df=hist,
                target_4h=targets[4.0],
                target_6h=targets[6.0]
            )
            windows.append(window)

            current_origin += timedelta(hours=origin_step_hours)

    return windows


def chronological_dataset_split(
    windows: List[ForecastingWindow],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[List[ForecastingWindow], List[ForecastingWindow], List[ForecastingWindow], Dict]:
    """
    Splits forecasting windows chronologically by forecast origin time T.
    """
    sorted_windows = sorted(windows, key=lambda w: w.origin_time)
    n = len(sorted_windows)
    
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    
    train_windows = sorted_windows[:n_train]
    val_windows = sorted_windows[n_train:n_train + n_val]
    test_windows = sorted_windows[n_train + n_val:]

    split_summary = {
        "total_windows": n,
        "train_count": len(train_windows),
        "train_pct": len(train_windows) / n * 100 if n > 0 else 0,
        "train_t_min": str(train_windows[0].origin_time) if train_windows else None,
        "train_t_max": str(train_windows[-1].origin_time) if train_windows else None,
        "val_count": len(val_windows),
        "val_pct": len(val_windows) / n * 100 if n > 0 else 0,
        "val_t_min": str(val_windows[0].origin_time) if val_windows else None,
        "val_t_max": str(val_windows[-1].origin_time) if val_windows else None,
        "test_count": len(test_windows),
        "test_pct": len(test_windows) / n * 100 if n > 0 else 0,
        "test_t_min": str(test_windows[0].origin_time) if test_windows else None,
        "test_t_max": str(test_windows[-1].origin_time) if test_windows else None,
    }

    return train_windows, val_windows, test_windows, split_summary
