"""
preprocessing.py
Data cleaning, anomaly filtering, deduplication, and trajectory splitting for AIS data.

Missing Value and Quality Filtering Methodology:
  - Latitude/Longitude: Records with null or out-of-bounds coordinates are dropped.
  - SOG (Speed Over Ground): Records with null SOG are dropped because reported speed
    is required to validate against physical limits and verify underway state.
  - COG (Course Over Ground): Null/NaN COG is explicitly permitted when a vessel is
    stationary (SOG <= 0.5 knots), where Doppler course is physically undefined.
    For moving vessels (SOG > 0.5 knots), out-of-range compass values are dropped.
  - Unreasonable Speeds: Speeds above max_sog (45.0 knots) are dropped as physical
    GPS transmission spikes, while legitimate slow/stationary craft (>= 0.0 knots) are retained.
  - Observation Gaps: When consecutive reports for a vessel exceed max_gap_hours (1.0 hour),
    the track is split into distinct continuous trajectory segments rather than interpolated.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict


def filter_invalid_coordinates(
    df: pd.DataFrame,
    min_lat: float = -90.0,
    max_lat: float = 90.0,
    min_lon: float = -180.0,
    max_lon: float = 180.0
) -> pd.DataFrame:
    """Removes rows with null or out-of-bounds geographic coordinates."""
    mask = (
        df["latitude"].notnull() &
        df["longitude"].notnull() &
        (df["latitude"] >= min_lat) &
        (df["latitude"] <= max_lat) &
        (df["longitude"] >= min_lon) &
        (df["longitude"] <= max_lon)
    )
    return df[mask].copy()


def remove_duplicate_records(
    df: pd.DataFrame,
    subset: list = ["mmsi", "base_date_time"]
) -> pd.DataFrame:
    """Removes duplicate AIS messages for the same vessel and timestamp, keeping first."""
    return df.drop_duplicates(subset=subset, keep="first").copy()


def filter_speeds(
    df: pd.DataFrame,
    min_sog: float = 0.0,
    max_sog: float = 45.0
) -> pd.DataFrame:
    """
    Filters Speed Over Ground (SOG):
      - Retains legitimate speeds from 0.0 knots up to max_sog (45.0 knots).
      - Drops null SOG values because speed verification is required.
      - Drops SOG > max_sog: physically unrealistic for commercial displacement vessels,
        reflecting GPS jump anomalies or transceiver transmission glitches.
    """
    mask = (
        df["sog"].notnull() &
        (df["sog"] >= min_sog) &
        (df["sog"] <= max_sog)
    )
    return df[mask].copy()


def filter_courses(
    df: pd.DataFrame,
    min_cog: float = 0.0,
    max_cog: float = 360.0
) -> pd.DataFrame:
    """
    Filters Course Over Ground (COG):
      - Valid compass range is [0.0, 360.0] degrees clockwise from North.
      - Allows null/NaN COG when vessel is stationary (where Doppler heading is undefined),
        but removes out-of-range COG values when present.
    """
    valid_range = (df["cog"] >= min_cog) & (df["cog"] <= max_cog)
    is_stationary = (df["sog"] <= 0.5) & df["cog"].isnull()
    mask = valid_range | is_stationary
    return df[mask].copy()


def split_trajectories_on_gaps(
    df: pd.DataFrame,
    max_gap_hours: float = 1.0
) -> pd.DataFrame:
    """
    Splits vessel tracks into distinct sub-trajectories whenever the observation gap
    between consecutive AIS reports exceeds max_gap_hours.
    Assigns a unique 'trajectory_id' (e.g., '{mmsi}_T001').
    """
    df = df.sort_values(by=["mmsi", "base_date_time"]).copy()
    
    # Calculate time difference between consecutive points for same vessel
    time_diff = df.groupby("mmsi")["base_date_time"].diff().dt.total_seconds() / 3600.0
    
    # Flag points that start a new trajectory (first point of vessel or gap > max_gap_hours)
    is_new_traj = (time_diff.isna()) | (time_diff > max_gap_hours)
    
    # Cumulative sum generates distinct segment indices
    df["segment_idx"] = is_new_traj.groupby(df["mmsi"]).cumsum()
    df["trajectory_id"] = df["mmsi"].astype(str) + "_T" + df["segment_idx"].astype(str).str.zfill(3)
    df.drop(columns=["segment_idx"], inplace=True)
    
    return df


def clean_ais_pipeline(
    df: pd.DataFrame,
    min_lat: float = 24.0,
    max_lat: float = 32.0,
    min_lon: float = -85.0,
    max_lon: float = -78.0,
    min_sog: float = 0.0,
    max_sog: float = 45.0,
    max_gap_hours: float = 1.0,
    min_points_per_trajectory: int = 15,
    min_trajectory_hours: float = 10.0
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Executes complete end-to-end cleaning pipeline and tracks record counts:
      Raw -> Valid Coords -> Deduplicated -> Speed Filtered -> Course Filtered
      -> Split on Gaps -> Trajectory Duration Filter
    """
    report = {}
    report["01_raw_records"] = len(df)
    
    # 1. Valid coordinates
    df1 = filter_invalid_coordinates(df, min_lat, max_lat, min_lon, max_lon)
    report["02_valid_coordinates"] = len(df1)
    
    # 2. Deduplication
    df2 = remove_duplicate_records(df1)
    report["03_deduplicated"] = len(df2)
    
    # 3. Speed filtering
    df3 = filter_speeds(df2, min_sog, max_sog)
    report["04_speed_filtered"] = len(df3)
    
    # 4. Course filtering
    df4 = filter_courses(df3)
    report["05_course_filtered"] = len(df4)
    
    # 5. Split on gaps
    df5 = split_trajectories_on_gaps(df4, max_gap_hours)
    report["06_total_split_segments"] = df5["trajectory_id"].nunique()
    
    # 6. Keep usable trajectories: duration >= min_trajectory_hours & points >= min_points
    traj_stats = df5.groupby("trajectory_id").agg(
        pts=("base_date_time", "count"),
        t_start=("base_date_time", "min"),
        t_end=("base_date_time", "max")
    )
    traj_stats["duration_hours"] = (traj_stats["t_end"] - traj_stats["t_start"]).dt.total_seconds() / 3600.0
    
    valid_trajs = traj_stats[
        (traj_stats["duration_hours"] >= min_trajectory_hours) &
        (traj_stats["pts"] >= min_points_per_trajectory)
    ].index
    
    df_clean = df5[df5["trajectory_id"].isin(valid_trajs)].copy()
    report["07_usable_trajectories"] = len(valid_trajs)
    report["08_usable_trajectory_records"] = len(df_clean)
    
    df_clean.sort_values(by=["trajectory_id", "base_date_time"], inplace=True)
    df_clean.reset_index(drop=True, inplace=True)
    
    return df_clean, report
