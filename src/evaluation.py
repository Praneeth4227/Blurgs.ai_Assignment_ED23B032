"""
evaluation.py
Comprehensive evaluation pipeline for vessel trajectory forecasting models.
Calculates geodesic forecast errors (Mean, Median, RMSE, P90) on WGS-84 in kilometres.
Produces overall benchmarks and vessel-type stratified evaluations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from src.coordinates import compute_geodesic_distance_km


def compute_geodesic_metrics(
    true_lats: np.ndarray,
    true_lons: np.ndarray,
    pred_lats: np.ndarray,
    pred_lons: np.ndarray
) -> Dict[str, float]:
    """
    Computes exact geodesic errors in kilometres on the WGS-84 ellipsoid.
    Returns: {mean, median, rmse, p90, std}
    """
    n = len(true_lats)
    errors = np.zeros(n, dtype=np.float64)
    for i in range(n):
        errors[i] = compute_geodesic_distance_km(
            true_lats[i], true_lons[i],
            pred_lats[i], pred_lons[i]
        )

    return {
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "p90": float(np.percentile(errors, 90)),
        "std": float(np.std(errors)),
        "errors": errors
    }


def evaluate_models_on_dataset(
    windows: list,
    models: dict,
    horizons: List[float] = [4.0, 6.0]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Evaluates a dictionary of models across forecasting windows.
    
    Returns:
      - predictions_df: sample-level prediction records and geodesic errors
      - benchmark_summary: overall benchmark table (Model | Horizon | Mean | Median | RMSE | P90)
      - vessel_breakdown: stratified benchmark table by vessel category
    """
    records = []

    for w in windows:
        hist = w.history_df
        t4_true = w.target_4h
        t6_true = w.target_6h

        for model_name, model in models.items():
            # Fit/predict model
            if hasattr(model, "fit_and_predict"):
                preds, _, _ = model.fit_and_predict(hist, horizons)
            else:
                preds = model.predict(hist, horizons)

            p4 = preds[4.0]
            p6 = preds[6.0]

            err_4h = compute_geodesic_distance_km(t4_true["lat"], t4_true["lon"], p4[0], p4[1])
            err_6h = compute_geodesic_distance_km(t6_true["lat"], t6_true["lon"], p6[0], p6[1])

            records.append({
                "sample_id": w.sample_id,
                "trajectory_id": w.trajectory_id,
                "mmsi": w.mmsi,
                "vessel_type": w.vessel_type,
                "model": model_name,
                "origin_time": w.origin_time,
                "origin_lat": w.origin_lat,
                "origin_lon": w.origin_lon,
                "origin_sog": w.origin_sog,
                "origin_cog": w.origin_cog,
                "mean_sog": getattr(w, "mean_sog", w.origin_sog),
                "is_underway": getattr(w, "is_underway", True),
                "pred_4h_lat": p4[0],
                "pred_4h_lon": p4[1],
                "true_4h_lat": t4_true["lat"],
                "true_4h_lon": t4_true["lon"],
                "error_4h_km": err_4h,
                "pred_6h_lat": p6[0],
                "pred_6h_lon": p6[1],
                "true_6h_lat": t6_true["lat"],
                "true_6h_lon": t6_true["lon"],
                "error_6h_km": err_6h
            })

    pred_df = pd.DataFrame(records)

    # Map vessel types to categories
    def map_cat(code):
        if pd.isna(code): return "Unknown"
        c = int(code)
        if 70 <= c <= 79: return "Cargo"
        elif 80 <= c <= 89: return "Tanker"
        elif c == 52: return "Tug / Towing"
        elif 60 <= c <= 69: return "Passenger"
        elif c in [30, 31, 32]: return "Fishing"
        else: return "Other"

    pred_df["vessel_category"] = pred_df["vessel_type"].apply(map_cat)

    # Overall benchmark summary
    summary_rows = []
    for model_name in models.keys():
        sub = pred_df[pred_df["model"] == model_name]
        for h, col in [(4.0, "error_4h_km"), (6.0, "error_6h_km")]:
            errs = sub[col].values
            summary_rows.append({
                "Model": model_name,
                "Horizon": f"+{int(h)}h",
                "Mean": float(np.mean(errs)),
                "Median": float(np.median(errs)),
                "RMSE": float(np.sqrt(np.mean(errs ** 2))),
                "P90": float(np.percentile(errs, 90)),
                "Std": float(np.std(errs)),
                "Samples": len(errs)
            })

    benchmark_df = pd.DataFrame(summary_rows)

    # Vessel breakdown summary
    vessel_rows = []
    top_categories = ["Cargo", "Tanker", "Tug / Towing", "Other"]
    for model_name in models.keys():
        for cat in top_categories:
            sub = pred_df[(pred_df["model"] == model_name) & (pred_df["vessel_category"] == cat)]
            if len(sub) == 0:
                continue
            for h, col in [(4.0, "error_4h_km"), (6.0, "error_6h_km")]:
                errs = sub[col].values
                vessel_rows.append({
                    "Model": model_name,
                    "Category": cat,
                    "Horizon": f"+{int(h)}h",
                    "Mean": float(np.mean(errs)),
                    "Median": float(np.median(errs)),
                    "RMSE": float(np.sqrt(np.mean(errs ** 2))),
                    "P90": float(np.percentile(errs, 90)),
                    "Samples": len(errs)
                })

    vessel_df = pd.DataFrame(vessel_rows)

    return pred_df, benchmark_df, vessel_df
