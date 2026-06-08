# -*- coding: utf-8 -*-
"""
04_plot_soil_depth.py

Plot soil depth at failure as a function of hollow slope.
Run this after 02_extract_and_calculate_RI.py.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from config import WorkflowConfig
from plot_helpers import add_critical_area_and_volume, clean_ri_dataframe, get_figure_dir, read_ri_results


plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "legend.title_fontsize": 11,
})

def main():
    cfg = WorkflowConfig()
    fig_dir = get_figure_dir(cfg)


    ri_df = read_ri_results(cfg.results_dir)
    ri_df = clean_ri_dataframe(
        ri_df,
        min_slope=25.0,
        #drop_indices=[55, 57, 65],
    )

    target_m = 1.0
    ri_df = ri_df[ri_df["m"] == target_m].copy()

    ri_df = add_critical_area_and_volume(
        ri_df,
        cfg,
        saturation=target_m,
    )

    markers = ["o", "s", "^", "D", "v", "P"]

    ac_min = ri_df["Ac"].min()
    ac_max = ri_df["Ac"].max()
    norm = mcolors.Normalize(vmin=ac_min, vmax=ac_max)
    cmap = cm.viridis

    grouped = sorted(
        ri_df.groupby("Cohesion"),
        key=lambda item: item[0],
        reverse=True,
    )

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.grid(True, alpha=0.4)

    for i, (cohesion, group) in enumerate(grouped):
        ax.scatter(
            group["Avg_Slope_deg"],
            group["Avg_Soil_Depth_m"],
            c=group["Ac"],
            cmap=cmap,
            norm=norm,
            marker=markers[i % len(markers)],
            s=70,
            alpha=0.85,
            edgecolors="grey",
            linewidths=0.4,
            label=f"{int(cohesion)} Pa",
        )

    cbar = plt.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        pad=0.01,
    )
    cbar.set_label(
        r"Critical area ($m^2$)",
        fontsize=10,
        fontweight="bold",
    )

    ax.set_xlabel(r"Hollow slope, $\theta_H$ (°)", fontweight="bold")
    ax.set_ylabel("Critical soil depth,\n$z_c$ (m)", fontweight="bold")

    legend_handles = [
        Line2D(
            [0], [0],
            marker=markers[i % len(markers)],
            color="w",
            markerfacecolor="gray",
            markeredgecolor="grey",
            markersize=8,
            label=f"{int(cohesion)} Pa",
        )
        for i, (cohesion, _) in enumerate(grouped)
    ]

    ax.legend(
        handles=legend_handles,
        title="Cohesion",
        loc="upper left",
        frameon=True,
    )

    ax.set_xlim(
        ri_df["Avg_Slope_deg"].min() - 1,
        ri_df["Avg_Slope_deg"].max() + 1.5,
    )

    xticks = np.arange(
        np.floor(ri_df["Avg_Slope_deg"].min() / 3) * 3,
        ri_df["Avg_Slope_deg"].max() + 1.5,
        3,
    )
    ax.set_xticks(xticks)

    fig.tight_layout()

    out_path = fig_dir / "soildepth_Ac_color.png"
    fig.savefig(out_path, dpi=450, bbox_inches="tight")
    plt.show()

    print(f"Saved: {out_path}")

    return ri_df

if __name__ == "__main__":
    main()


