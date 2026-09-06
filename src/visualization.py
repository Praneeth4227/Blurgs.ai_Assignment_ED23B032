"""
visualization.py
Visualization suite for vessel trajectory forecasting.
Generates:
  - Horizon comparative error bar charts (+4h and +6h)
  - Empirical Cumulative Distribution Function (ECDF) curves
  - Performance by vessel category
  - Representative success and failure case trajectory maps
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_model_comparison_horizons(
    benchmark_df: pd.DataFrame,
    output_path: str
):
    """
    Side-by-side grouped bar chart comparing Mean and Median geodesic errors
    across models at +4h and +6h.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    df_4h = benchmark_df[benchmark_df["Horizon"] == "+4h"].sort_values("Median")
    df_6h = benchmark_df[benchmark_df["Horizon"] == "+6h"].sort_values("Median")

    # Horizon +4h
    x = np.arange(len(df_4h))
    width = 0.35

    ax1.bar(x - width/2, df_4h["Median"], width, label="Median Error (km)", color="#2b5c8f", edgecolor="black")
    ax1.bar(x + width/2, df_4h["Mean"], width, label="Mean Error (km)", color="#e26d5c", edgecolor="black")
    ax1.set_title("Forecast Accuracy at +4 Hours Horizon", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_4h["Model"], rotation=15)
    ax1.set_ylabel("Geodesic Error (km)")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Annotate numbers
    for i, (_, row) in enumerate(df_4h.iterrows()):
        ax1.text(i - width/2, row["Median"] + 1, f"{row['Median']:.1f}", ha="center", va="bottom", fontsize=8.5)
        ax1.text(i + width/2, row["Mean"] + 1, f"{row['Mean']:.1f}", ha="center", va="bottom", fontsize=8.5)

    # Horizon +6h
    x6 = np.arange(len(df_6h))
    ax2.bar(x6 - width/2, df_6h["Median"], width, label="Median Error (km)", color="#2b5c8f", edgecolor="black")
    ax2.bar(x6 + width/2, df_6h["Mean"], width, label="Mean Error (km)", color="#e26d5c", edgecolor="black")
    ax2.set_title("Forecast Accuracy at +6 Hours Horizon", fontsize=12)
    ax2.set_xticks(x6)
    ax2.set_xticklabels(df_6h["Model"], rotation=15)
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    for i, (_, row) in enumerate(df_6h.iterrows()):
        ax2.text(i - width/2, row["Median"] + 1, f"{row['Median']:.1f}", ha="center", va="bottom", fontsize=8.5)
        ax2.text(i + width/2, row["Mean"] + 1, f"{row['Mean']:.1f}", ha="center", va="bottom", fontsize=8.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_error_distributions(
    pred_df: pd.DataFrame,
    output_path: str
):
    """
    Plots Empirical Cumulative Distribution Function (ECDF) curves of forecast errors
    for all 4 models at +4h and +6h.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    models = pred_df["model"].unique()
    colors = {"Linear": "#1f77b4", "Polynomial (d=2)": "#9467bd", "Kinematic (SOG/COG)": "#2ca02c", "Kalman Filter": "#d62728"}

    for name in models:
        sub = pred_df[pred_df["model"] == name]
        
        # +4h ECDF
        e4 = np.sort(sub["error_4h_km"].values)
        cdf4 = np.arange(1, len(e4) + 1) / len(e4)
        ax1.plot(e4, cdf4, label=name, color=colors.get(name, "black"), lw=2)

        # +6h ECDF
        e6 = np.sort(sub["error_6h_km"].values)
        cdf6 = np.arange(1, len(e6) + 1) / len(e6)
        ax2.plot(e6, cdf6, label=name, color=colors.get(name, "black"), lw=2)

    ax1.set_title("Empirical CDF of Forecast Error (+4h)", fontsize=12)
    ax1.set_xlabel("Geodesic Error (km)")
    ax1.set_ylabel("Cumulative Fraction")
    ax1.set_xlim(0, 150)
    ax1.axhline(0.5, color="gray", linestyle=":", label="Median (50%)")
    ax1.axhline(0.9, color="orange", linestyle=":", label="P90 (90%)")
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.set_title("Empirical CDF of Forecast Error (+6h)", fontsize=12)
    ax2.set_xlabel("Geodesic Error (km)")
    ax2.set_ylabel("Cumulative Fraction")
    ax2.set_xlim(0, 200)
    ax2.axhline(0.5, color="gray", linestyle=":", label="Median (50%)")
    ax2.axhline(0.9, color="orange", linestyle=":", label="P90 (90%)")
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_vessel_type_performance(
    vessel_df: pd.DataFrame,
    output_path: str
):
    """
    Bar chart of forecast error stratified by vessel category (Cargo vs Tanker vs Tug).
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    df_plot = vessel_df[vessel_df["Horizon"] == "+4h"].sort_values("Median")

    categories = df_plot["Category"].unique()
    models = df_plot["Model"].unique()

    x = np.arange(len(categories))
    width = 0.2

    for i, m in enumerate(models):
        sub = df_plot[df_plot["Model"] == m]
        medians = [sub[sub["Category"] == c]["Median"].values[0] if len(sub[sub["Category"] == c]) > 0 else 0 for c in categories]
        ax.bar(x + (i - 1.5) * width, medians, width, label=m, edgecolor="black", alpha=0.85)

    ax.set_title("Median Forecast Error by Vessel Type (+4 Hours Horizon)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Median Geodesic Error (km)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_case_studies(
    success_window,
    failure_window,
    pred_df: pd.DataFrame,
    output_path: str
):
    """
    Generates side-by-side maps of:
      1. Representative Success Case (Steady transit cruising)
      2. Representative Failure Case (Vessel executing major maneuver/course change)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    cases = [(ax1, success_window, "Success Case: Open-Water Cruise (Constant Velocity)"),
             (ax2, failure_window, "Failure Case: Severe Course Alteration / Turn")]

    for ax, w, title in cases:
        hist = w.history_df
        ax.plot(hist["longitude"], hist["latitude"], "b.-", lw=1.5, label="4h History")
        ax.scatter([w.origin_lon], [w.origin_lat], color="blue", s=80, zorder=5, label="Origin T")

        # Ground truth
        ax.scatter([w.target_4h["lon"]], [w.target_4h["lat"]], color="black", s=90, marker="*", zorder=5, label="Actual +4h")
        ax.scatter([w.target_6h["lon"]], [w.target_6h["lat"]], color="black", s=110, marker="X", zorder=5, label="Actual +6h")

        # Extract predictions for this sample
        sub_preds = pred_df[pred_df["sample_id"] == w.sample_id]
        colors = {"Linear": "crimson", "Polynomial (d=2)": "purple", "Kinematic (SOG/COG)": "teal", "Kalman Filter": "green"}

        for _, row in sub_preds.iterrows():
            m = row["model"]
            ax.plot([w.origin_lon, row["pred_4h_lon"], row["pred_6h_lon"]],
                    [w.origin_lat, row["pred_4h_lat"], row["pred_6h_lat"]],
                    linestyle="--", marker="o", color=colors.get(m, "gray"), alpha=0.8,
                    label=f"{m} (Err: {row['error_4h_km']:.1f} km)")

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Longitude (°W)")
        ax.set_ylabel("Latitude (°N)")
        ax.legend(loc="best", fontsize=8.5)
        ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
