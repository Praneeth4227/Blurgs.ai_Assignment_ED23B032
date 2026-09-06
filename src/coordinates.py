"""
coordinates.py
Geospatial coordinate conversions and geodesic distance computations.
Converts between WGS-84 (lat/lon in degrees) and local metric East-North tangent planes (x, y in metres).
"""

import numpy as np
from geopy.distance import geodesic

# WGS-84 Earth equatorial radius in metres
EARTH_RADIUS_METRES = 6378137.0


def wgs84_to_local_enu(
    lat: np.ndarray,
    lon: np.ndarray,
    lat_ref: float,
    lon_ref: float
) -> tuple:
    """
    Projects WGS-84 (lat, lon in degrees) to a local flat East-North tangent plane (x, y in metres)
    centered at (lat_ref, lon_ref).
    
    x: Easting displacement (positive East) in metres
    y: Northing displacement (positive North) in metres
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    lat_ref_rad = np.radians(lat_ref)
    lon_ref_rad = np.radians(lon_ref)

    dlat = lat_rad - lat_ref_rad
    dlon = lon_rad - lon_ref_rad

    # Metric projection on local tangent plane
    x = EARTH_RADIUS_METRES * dlon * np.cos(lat_ref_rad)
    y = EARTH_RADIUS_METRES * dlat

    return x, y


def local_enu_to_wgs84(
    x: np.ndarray,
    y: np.ndarray,
    lat_ref: float,
    lon_ref: float
) -> tuple:
    """
    Inverts local flat tangent coordinates (x, y in metres) back to WGS-84 (lat, lon in degrees)
    relative to reference point (lat_ref, lon_ref).
    """
    lat_ref_rad = np.radians(lat_ref)
    lon_ref_rad = np.radians(lon_ref)

    dlat = y / EARTH_RADIUS_METRES
    dlon = x / (EARTH_RADIUS_METRES * np.cos(lat_ref_rad))

    lat_pred = np.degrees(lat_ref_rad + dlat)
    lon_pred = np.degrees(lon_ref_rad + dlon)

    return lat_pred, lon_pred


def compute_geodesic_distance_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float
) -> float:
    """
    Computes exact geodesic distance on the WGS-84 ellipsoid in kilometres.
    Uses Karney's geodesic method via geopy.
    """
    return geodesic((lat1, lon1), (lat2, lon2)).kilometers


def batch_geodesic_distance_km(
    lats_true: np.ndarray,
    lons_true: np.ndarray,
    lats_pred: np.ndarray,
    lons_pred: np.ndarray
) -> np.ndarray:
    """
    Vectorized computation of geodesic error in kilometres.
    Computes exact geopy geodesic for each pair.
    """
    errors = np.zeros(len(lats_true), dtype=np.float64)
    for i in range(len(lats_true)):
        errors[i] = geodesic(
            (lats_true[i], lons_true[i]),
            (lats_pred[i], lons_pred[i])
        ).kilometers
    return errors
