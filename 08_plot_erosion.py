# -*- coding: utf-8 -*-
"""
08_plot_erosion.py

Calculate hollow-scale erosion rates from modeled recurrence interval results,
critical failure volume, and contributing-area zonal statistics. Produces the
slope-erosion plot used for manuscript figures.

Run after 02_extract_and_calculate_RI.py and after zonal statistic CSVs are
available.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit

from config import WorkflowConfig
from plot_helpers import (
    add_critical_area_and_volume,
    clean_ri_dataframe,
    get_figure_dir,
    read_ri_results,
)


# =============================================================================
# USER SETTINGS
# =============================================================================
TARGET_COHESION = 3600
SATURATION_FOR_VOLUME = 1.0
MIN_SLOPE_DEG = 20.0

ZONAL_9_CSV = "zonal_9_1.csv"


SAVE_FIGURE = True
SAVE_EROSION_TABLE = True


plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10,
    "legend.title_fontsize": 11,
})


def nonlinear_transport(x, a, s_c):
    """Roering et al. (1999) nonlinear hillslope transport model.

    E = a / (1 - (S / S_c)^2)

    Parameters
    ----------
    x : array-like
        Slope in degrees.
    a : float
        Low-slope erosion rate coefficient.
    s_c : float
        Critical slope (degrees) at which flux diverges.
    """
    return a / (1 - (x / s_c) ** 2)


def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


def load_zonal_table(path, required_column):
    """Read zonal-statistics CSV and normalize extent names."""
    df = pd.read_csv(path)
    if "Extent" not in df.columns:
        raise ValueError(f"{path} is missing required column: Extent")
    if required_column not in df.columns:
        raise ValueError(f"{path} is missing required column: {required_column}")

    df["Extent"] = df["Extent"].astype(str).str.replace(r"_\d+$", "", regex=True)
    return df[["Point_ID", "Extent", required_column]]


def main():
    cfg = WorkflowConfig()
    fig_dir = get_figure_dir(cfg)

    ri_df = read_ri_results(cfg.results_dir)

    
    ri_df = clean_ri_dataframe(
        ri_df,
        min_slope=20.0,
        drop_points=[
            ("ext1", 4),
            ("ext1", 3),
            ("ext1", 2),
            ("ext1", 1),
            ("ext16", 1)
        ],
    )

    ri_df = ri_df[ri_df["Cohesion"] == TARGET_COHESION].copy()
    ri_df = ri_df[ri_df["m"] == 1.0].copy()
    
    if ri_df.empty:
        raise ValueError(f"No RI results found for Cohesion = {TARGET_COHESION} Pa")

    ri_df = add_critical_area_and_volume(ri_df, cfg, saturation=SATURATION_FOR_VOLUME)

    zonal_9_path = cfg.base_dir / ZONAL_9_CSV
    

    zonal_9 = load_zonal_table(zonal_9_path, "max_9")
    

    merged_ero = (
        ri_df
        .merge(zonal_9, on=["Point_ID", "Extent"], how="inner")
        
    )

    merged_ero["Erosion_9"] = (merged_ero["Volume"] / (merged_ero["Year"] * merged_ero["max_9"])) * 1000.0
    

    # Remove anomalously low erosion-rate point near 42.7 degrees if present.
    plot_ero = merged_ero[
        ~(
            (merged_ero["Avg_Slope_deg"] > 42.5)
            & (merged_ero["Avg_Slope_deg"] < 43.0)
            & (merged_ero["Erosion_9"] < 0.05)
        )
    ].copy()

    plot_ero = plot_ero.dropna(subset=["Avg_Slope_deg", "Erosion_9"])
    plot_ero = plot_ero[plot_ero["Erosion_9"] > 0]
    plot_ero = plot_ero.drop([21])
    # plot_ero = plot_ero.drop([12, 20])
    
    if SAVE_EROSION_TABLE:
        out_csv = cfg.results_dir / f"erosion_results_C{TARGET_COHESION}.csv"
        merged_ero.to_csv(out_csv, index=False)
        print(f"Saved erosion table: {out_csv}")

    x = plot_ero["Avg_Slope_deg"].to_numpy()
    y = plot_ero["Erosion_9"].to_numpy()

    params, _ = curve_fit(
        nonlinear_transport,
        x,
        y,
        p0=[0.01, 60.0],
        bounds=([0, x.max() + 0.01], [np.inf, 200]),
        maxfev=20000,
    )
    r2 = r_squared(y, nonlinear_transport(x, *params))

    slope_range = np.linspace(x.min(), 45.0, 500)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_facecolor("#f0f0f0")
    ax.set_ylim(1e-2, 5e-1)

    ax.scatter(
        x,
        y,
        c="black",
        s=70,
        edgecolors="#111",
        linewidths=0.5,
        alpha=0.9,
        zorder=3,
    )

    ax.plot(
        slope_range,
        nonlinear_transport(slope_range, *params),
        c="red",
        linestyle="--",
        linewidth=2.2,
        zorder=2,
    )

    ax.set_xlabel(r"Hollow slope, $\theta_H$ (°)", fontweight="bold")
    ax.set_ylabel(r"Erosion rate, $E$ (mm yr$^{-1}$)", fontweight="bold")

    xticks = np.arange(
        np.floor(plot_ero["Avg_Slope_deg"].min() / 3) * 3,
        plot_ero["Avg_Slope_deg"].max() + 3,
        3,
    )
    ax.set_xticks(xticks)
    ax.grid(True, which="both", linestyle="--", alpha=0.40)

    ax.set_xlim(right=45.0)

    eq_label = (
        fr"$E = {params[0]:.5f}\,/\,"
        fr"(1 - (\theta_H / {params[1]:.1f})^2)$"
    )
    r2_label = fr"$R^2 = {r2:.3f}$"

    legend_handles = [
        Line2D([], [], color="red", linestyle="--", lw=2.2, label=eq_label),
        Line2D([], [], linestyle="None", label=r2_label),
    ]

    legend = ax.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="black",
    )
    legend.get_texts()[1].set_color("red")

    fig.tight_layout()

    if SAVE_FIGURE:
        fig.savefig(fig_dir / "erosion_slope_final.png", dpi=450, bbox_inches="tight")

    plt.show()
    return plot_ero


if __name__ == "__main__":
    plot_ero = main()
