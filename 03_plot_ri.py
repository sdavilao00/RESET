# -*- coding: utf-8 -*-
"""
03_plot_ri.py

Plot modeled recurrence interval (RI) as a function of hollow slope.
Run this after 02_extract_and_calculate_RI.py.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext
from scipy.optimize import curve_fit

from config import WorkflowConfig
from plot_helpers import (
    clean_ri_dataframe,
    get_figure_dir,
    inverse_model_log,
    read_ri_results,
)


plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
    "legend.title_fontsize": 16,
})


CBF_COLORS = ["#E69F00", "#56B4E9", "#009E73", "#D55E00", "#CC79A7", "#0072B2"]

COLOR_MAP = {
    6400: "#E69F00",
    3600: "#56B4E9",
    1920: "#009E73",
    760: "#D55E00",
}


def setup_ri_axis(ax):
    ax.set_axisbelow(True)
    ax.minorticks_on()

    ax.grid(True, which="major", axis="y", color="#d0d0d0", alpha=0.7)
    ax.grid(True, which="minor", axis="y", color="#d0d0d0", alpha=0.3)
    ax.grid(True, which="major", axis="x", color="#bdbdbd", linewidth=0.8)
    ax.grid(False, which="minor", axis="x")

    ax.set_yscale("log")
    ax.set_ylim(50, 1e4)
    ax.set_yticks([100, 1000, 10000])
    ax.yaxis.set_major_formatter(LogFormatterMathtext())

    ax.set_xlabel(r"Hollow slope, $\theta_H$ (°)", fontweight="bold")
    ax.set_ylabel("Recurrence interval, RI (years)", fontweight="bold")


def plot_inverse_fit(ax, x, y, color, linestyle="--"):
    if len(x) < 3:
        return

    popt, _ = curve_fit(
        inverse_model_log,
        x,
        np.log10(y),
        p0=[3.0, -25.0],
        maxfev=10000,
    )

    loga_fit, b_fit = popt
    a_fit = 10 ** loga_fit

    x_fit = np.linspace(x.min() - 0.5, x.max() + 0.5, 300)
    y_fit = a_fit / (x_fit + b_fit)

    ax.plot(
        x_fit,
        y_fit,
        color=color,
        lw=2.5,
        linestyle=linestyle,
    )


def plot_first_ri_figure(ri_df, fig_dir):
    """Plot m = 1 RI curves for all cohesion values."""

    target_m = 1.0
    plot_df = ri_df[ri_df["m"] == target_m].copy()

    cohesions = sorted(plot_df["Cohesion"].unique(), reverse=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    setup_ri_axis(ax)

    for i, cohesion in enumerate(cohesions):
        group = plot_df[plot_df["Cohesion"] == cohesion].copy()

        x = group["Avg_Slope_deg"].to_numpy()
        y = group["Year"].to_numpy()
        color = CBF_COLORS[i % len(CBF_COLORS)]

        ax.scatter(x, y, color=color, s=60, alpha=0.75)
        plot_inverse_fit(ax, x, y, color=color, linestyle="--")

    xticks = np.arange(
        np.floor(plot_df["Avg_Slope_deg"].min() / 3) * 3,
        np.ceil(plot_df["Avg_Slope_deg"].max() / 3) * 3 + 3,
        3,
    )
    ax.set_xticks(xticks)

    cohesion_handles = [
        Line2D(
            [], [],
            marker="o",
            linestyle="None",
            color=CBF_COLORS[i % len(CBF_COLORS)],
            markersize=8,
            label=f"{int(cohesion)} Pa",
        )
        for i, cohesion in enumerate(cohesions)
    ]

    ax.legend(
        handles=cohesion_handles,
        title="Cohesion",
        loc="upper right",
        frameon=True,
    )

    fig.tight_layout()

    out_path = fig_dir / "RI_3D_final.png"
    fig.savefig(out_path, dpi=450, bbox_inches = 'tight')
    plt.show()

    print(f"Saved: {out_path}")

    return plot_df


def plot_m_comparison_figure(ri_df, fig_dir):
    """Plot m = 1 fit lines and m = 0.85 points/fit lines for 760 and 1920 Pa."""

    plot_df = ri_df[
        (ri_df["Cohesion"].isin([760, 1920])) &
        (ri_df["m"].isin([1.0, 0.85]))
    ].copy()

    cohesions = sorted(plot_df["Cohesion"].unique(), reverse=True)
    m_values = [1.0, 0.85]

    fig, ax = plt.subplots(figsize=(9, 6))
    setup_ri_axis(ax)

    for cohesion in cohesions:
        color = COLOR_MAP[cohesion]

        for m_value in m_values:
            group = plot_df[
                (plot_df["Cohesion"] == cohesion) &
                (plot_df["m"] == m_value)
            ].copy()

            if group.empty:
                continue

            x = group["Avg_Slope_deg"].to_numpy()
            y = group["Year"].to_numpy()

            if m_value == 1.0:
                linestyle = "-"
            else:
                linestyle = "--"

                ax.scatter(
                    x,
                    y,
                    color=color,
                    marker="^",
                    s=60,
                    alpha=0.75,
                )

            plot_inverse_fit(
                ax,
                x,
                y,
                color=color,
                linestyle=linestyle,
            )

    xticks = np.arange(
        np.floor(plot_df["Avg_Slope_deg"].min() / 3) * 3,
        plot_df["Avg_Slope_deg"].max() + 1,
        3,
    )
    ax.set_xticks(xticks)

    cohesion_handles = [
        Line2D(
            [], [],
            marker="o",
            linestyle="None",
            color=COLOR_MAP[cohesion],
            markersize=8,
            label=f"{int(cohesion)} Pa",
        )
        for cohesion in cohesions
    ]

    legend1 = ax.legend(
        handles=cohesion_handles,
        title="Cohesion",
        loc="upper right",
        frameon=True,
    )

    ax.add_artist(legend1)

    saturation_handles = [
        Line2D(
            [], [],
            linestyle="-",
            color="black",
            lw=2.5,
            label="m = 1.0",
        ),
        Line2D(
            [], [],
            marker="^",
            linestyle="--",
            color="black",
            markersize=8,
            lw=2.5,
            label="m = 0.85",
        ),
    ]

    ax.legend(
        handles=saturation_handles,
        title="Saturation",
        loc="lower left",
        frameon=True,
    )

    fig.tight_layout()

    out_path = fig_dir / "RI_3D_m_comparison_760_1920.png"
    fig.savefig(out_path, dpi=450, bbox_inches = 'tight')
    plt.show()

    print(f"Saved: {out_path}")

    return plot_df


def main():
    cfg = WorkflowConfig()
    fig_dir = get_figure_dir(cfg)

    ri_df = read_ri_results(cfg.results_dir)

    ri_df = clean_ri_dataframe(
        ri_df,
        min_slope=25.0,
        drop_indices=[55, 57, 65],
    )

    ri_df.to_csv(fig_dir / "cleaned_RI_dataframe.csv", index=False)

    ri_df_m1 = plot_first_ri_figure(ri_df, fig_dir)
    ri_df_mcompare = plot_m_comparison_figure(ri_df, fig_dir)

    return ri_df, ri_df_m1, ri_df_mcompare


if __name__ == "__main__":
    ri_df, ri_df_m1, ri_df_mcompare = main()