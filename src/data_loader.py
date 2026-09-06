"""
data_loader.py
Module for downloading, unpacking, and loading raw NOAA MarineCadastre AIS data.
Handles WKB geometry decoding into geographic coordinates and regional extraction.
"""

import os
import io
import struct
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def decode_wkb_points(geometries):
    """
    Vectorized extraction of (longitude, latitude) from WKB Point binary blobs.
    WKB Point layout:
      - 1 byte byte-order (1 = little-endian)
      - 4 bytes geometry type (1 = Point)
      - 8 bytes double longitude (X)
      - 8 bytes double latitude (Y)
    Total = 21 bytes per point.
    """
    raw = b"".join(geometries)
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(len(geometries), 21)
    lons = arr[:, 5:13].copy().view(dtype=np.float64).flatten()
    lats = arr[:, 13:21].copy().view(dtype=np.float64).flatten()
    return lons, lats


def load_ais_data(
    file_path: str,
    bounding_box: dict = None,
    columns: list = None
) -> pd.DataFrame:
    """
    Loads AIS data from parquet file.
    If 'geometry' column is present, automatically decodes to 'longitude' and 'latitude'.
    Optionally filters by bounding box: {'min_lat': ..., 'max_lat': ..., 'min_lon': ..., 'max_lon': ...}
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"AIS file not found at: {file_path}")

    if columns is None:
        df = pd.read_parquet(file_path)
    else:
        # Ensure geometry is included if needed for lat/lon extraction
        cols_to_read = list(columns)
        has_coords = "longitude" in df.columns if False else True
        df = pd.read_parquet(file_path, columns=cols_to_read)

    if "geometry" in df.columns:
        lons, lats = decode_wkb_points(df["geometry"].values)
        df["longitude"] = lons
        df["latitude"] = lats
        df.drop(columns=["geometry"], inplace=True)

    if "base_date_time" in df.columns:
        df["base_date_time"] = pd.to_datetime(df["base_date_time"])

    # Spatial bounding box filter if specified
    if bounding_box:
        mask = (
            (df["latitude"] >= bounding_box["min_lat"]) &
            (df["latitude"] <= bounding_box["max_lat"]) &
            (df["longitude"] >= bounding_box["min_lon"]) &
            (df["longitude"] <= bounding_box["max_lon"])
        )
        df = df[mask].copy()

    # Sort strictly by vessel and time
    if "mmsi" in df.columns and "base_date_time" in df.columns:
        df.sort_values(by=["mmsi", "base_date_time"], inplace=True)
        df.reset_index(drop=True, inplace=True)

    return df


def compute_dataset_summary(df: pd.DataFrame) -> dict:
    """
    Computes an empirical summary of the raw dataset:
      - record count
      - unique vessels
      - temporal coverage
      - spatial bounds
      - SOG / COG statistics
      - missing value counts
      - duplicate record count
      - sampling frequency statistics
    """
    summary = {
        "total_records": len(df),
        "columns": df.columns.tolist(),
        "unique_mmsi": int(df["mmsi"].nunique()) if "mmsi" in df.columns else 0,
        "time_min": str(df["base_date_time"].min()) if "base_date_time" in df.columns else None,
        "time_max": str(df["base_date_time"].max()) if "base_date_time" in df.columns else None,
        "lat_min": float(df["latitude"].min()) if "latitude" in df.columns else None,
        "lat_max": float(df["latitude"].max()) if "latitude" in df.columns else None,
        "lon_min": float(df["longitude"].min()) if "longitude" in df.columns else None,
        "lon_max": float(df["longitude"].max()) if "longitude" in df.columns else None,
        "sog_min": float(df["sog"].min()) if "sog" in df.columns else None,
        "sog_max": float(df["sog"].max()) if "sog" in df.columns else None,
        "sog_median": float(df["sog"].median()) if "sog" in df.columns else None,
        "cog_min": float(df["cog"].min()) if "cog" in df.columns else None,
        "cog_max": float(df["cog"].max()) if "cog" in df.columns else None,
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_mmsi_timestamp": int(df.duplicated(subset=["mmsi", "base_date_time"]).sum()) if ("mmsi" in df.columns and "base_date_time" in df.columns) else 0
    }

    # Estimate sampling frequency (median interval between reports per vessel)
    if "mmsi" in df.columns and "base_date_time" in df.columns and len(df) > 1:
        diffs = df.groupby("mmsi")["base_date_time"].diff().dt.total_seconds().dropna()
        summary["median_sampling_interval_seconds"] = float(diffs.median()) if len(diffs) > 0 else None
        summary["p90_sampling_interval_seconds"] = float(diffs.quantile(0.90)) if len(diffs) > 0 else None

    # Vessel type breakdown
    if "vessel_type" in df.columns:
        summary["vessel_type_counts"] = df["vessel_type"].value_counts().to_dict()

    return summary
