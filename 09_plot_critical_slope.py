# -*- coding: utf-8 -*-
"""
09_plot_critical_slope.py

Calculate and plot critical hollow slope as a function of saturation and cohesion.
The critical slope is defined as the smallest slope where min_z FS <= 1
for a given cohesion and saturation.

Run after config.py and plot_helpers.py are available.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from config import WorkflowConfig
from plot_helpers import get_figure_dir


plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 11,
    "legend.title_fontsize": 12,
})


# =============================================================================
# USER SETTINGS
# =============================================================================

PW = 1000.0
PS = 1600.0
G = 9.81

PHI_DEG = 41.0
J = 0.8

# Representative failure geometry
LENGTH_M = 23.9
WIDTH_M = 15.9

# Depth range searched for min_z FS
ZMIN = 0.01
ZMAX = 5.0
NZ = 1600

# Parameter ranges
COHESIONS = [6400.0, 3600.0, 1920.0, 760.0]
M_VALUES = np.round(np.linspace(0.5, 1.0, 51), 2)

# Colors match cohesion order above
COLOR_MAP = {
    6400: "#E69F00",
    3600: "#56B4E9",
    1920: "#009E73",
    760: "#D55E00",
}


# =============================================================================
# MODEL SETUP
# =============================================================================

YW = G * PW
YS = G * PS

PHI = np.deg2rad(PHI_DEG)
TAN_PHI = np.tan(PHI)

KP = np.tan(np.deg2rad(45.0) + PHI / 2.0) ** 2
KA = np.tan(np.deg2rad(45.0) - PHI / 2.0) ** 2

ZS = np.linspace(ZMIN, ZMAX, NZ)


def calculate_min_fs(theta_deg, m, cohesion, length_m=LENGTH_M, width_m=WIDTH_M):
    """
    Calculate minimum factor of safety over soil depths ZS for a given
    slope, saturation, cohesion, and failure geometry.
    """

    theta = np.deg2rad(theta_deg)

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    exp_term = np.exp(-ZS * J)

    Crb = cohesion * exp_term
    Crl = (cohesion / (J * ZS)) * (1.0 - exp_term)

    K0 = 1.0 - sin_theta

    Frb = (
        Crb
        + (cos_theta ** 2) * ZS * (YS - YW * m) * TAN_PHI
    ) * length_m * width_m

    Frc = (
        Crl
        + K0 * 0.5 * ZS * (YS - YW * (m ** 2)) * TAN_PHI
    ) * (cos_theta * ZS * length_m * 2.0)

    Frddu = (
        (KP - KA)
        * 0.5
        * (ZS ** 2)
        * (YS - YW * (m ** 2))
        * width_m
    )

    Fdc = (
        sin_theta
        * cos_theta
        * ZS
        * YS
        * length_m
        * width_m
    )

    FS = (Frb + Frc + Frddu) / Fdc

    return float(np.min(FS))


def calculate_theta_crit(
    m,
    cohesion,
    theta_min=10.0,
    theta_max=80.0,
    coarse_step=1.0,
    tol=0.05,
):
    """
    Calculate critical slope for one saturation and cohesion value.

    theta_crit is the smallest slope where min_z FS <= 1.
    """

    thetas = np.arange(theta_min, theta_max + coarse_step, coarse_step)
    fvals = np.array([
        calculate_min_fs(theta, m, cohesion) - 1.0
        for theta in thetas
    ])

    if np.min(fvals) > 0:
        return np.nan

    crossing_index = None

    for i in range(len(thetas) - 1):
        if (fvals[i] > 0) and (fvals[i + 1] <= 0):
            crossing_index = i
            break

    if crossing_index is None:
        return float(theta_min)

    lo = float(thetas[crossing_index])
    hi = float(thetas[crossing_index + 1])

    while (hi - lo) > tol:
        mid = 0.5 * (lo + hi)

        if calculate_min_fs(mid, m, cohesion) <= 1.0:
            hi = mid
        else:
            lo = mid

    return 0.5 * (lo + hi)


def build_critical_slope_dataframe():
    """Calculate theta_crit for all cohesion and saturation combinations."""

    records = []

    for cohesion in COHESIONS:
        for m in M_VALUES:
            theta_crit = calculate_theta_crit(m, cohesion)

            records.append({
                "Cohesion": int(cohesion),
                "m": float(m),
                "theta_crit_deg": theta_crit,
                "length_m": LENGTH_M,
                "width_m": WIDTH_M,
                "zmin_m": ZMIN,
                "zmax_m": ZMAX,
            })

    return pd.DataFrame(records)


def setup_critical_slope_axis(ax):
    """Apply shared formatting to the critical slope plot."""

    ax.set_axisbelow(True)
    # ax.set_facecolor("#f0f0f0")
    ax.grid(True, which="major", color="#d0d0d0", alpha=0.7)

    ax.set_xlabel("Saturation, m", fontweight="bold")
    ax.set_ylabel(r"Critical slope, $\theta_{\mathrm{crit}}$ (°)", fontweight="bold")


def plot_critical_slope_figure(crit_df, fig_dir):
    """Plot theta_crit vs saturation for all cohesion values."""

    fig, ax = plt.subplots(figsize=(9, 6))
    setup_critical_slope_axis(ax)

    for cohesion in COHESIONS:
        group = crit_df[crit_df["Cohesion"] == int(cohesion)].copy()
        color = COLOR_MAP[int(cohesion)]

        ax.plot(
            group["m"],
            group["theta_crit_deg"],
            color=color,
            linewidth=2.5,
            label=f"{int(cohesion)} Pa",
        )

    # Cohesion handles
    cohesion_handles = [
        Line2D(
            [], [],
            color=COLOR_MAP[int(cohesion)],
            lw=2.5,
            label=f"{int(cohesion)} Pa",
        )
        for cohesion in COHESIONS
    ]
    
    # Reference saturation handles
    reference_handles = [
        Line2D(
            [], [],
            color="black",
            linestyle="--",
            lw=1.5,
            label="m = 0.85",
        ),
        Line2D(
            [], [],
            color="black",
            linestyle=":",
            lw=1.5,
            label="m = 1.0",
        ),
    ]
    
    # Combined legend
    ax.legend(
        handles=cohesion_handles + reference_handles,
        title="Cohesion / Saturation",
        loc="lower left",
        frameon=True,
    )
    
    # Reference saturation values
    ax.axvline(
        0.85,
        color="black",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
    )
    
    ax.axvline(
        1.0,
        color="black",
        linestyle=":",
        linewidth=1.5,
        alpha=0.8,
    )


    fig.tight_layout()

    out_path = fig_dir / "critical_slope_vs_saturation_l23p9_w15p9.png"
    fig.savefig(out_path, dpi=450, bbox_inches="tight")
    plt.show()

    print(f"Saved: {out_path}")

    return out_path


def write_critical_slope_summary(crit_df, fig_dir):
    """Save CSV and print key m = 0.85 and m = 1.0 values."""

    csv_path = fig_dir / "critical_slope_dataframe_l23p9_w15p9.csv"
    crit_df.to_csv(csv_path, index=False)

    print(f"Saved critical slope dataframe: {csv_path}")

    key_values = (
        crit_df[crit_df["m"].isin([0.85, 1.0])]
        .pivot(index="Cohesion", columns="m", values="theta_crit_deg")
    )

    print("\nKey theta_crit values:")
    print(key_values)

    txt_path = fig_dir / "critical_slope_key_values_l23p9_w15p9.txt"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Critical slope key values\n")
        f.write("=========================\n")
        f.write(f"length_m = {LENGTH_M}\n")
        f.write(f"width_m = {WIDTH_M}\n")
        f.write(f"z range = {ZMIN} to {ZMAX} m\n\n")
        f.write(key_values.to_string())

    print(f"Saved key values: {txt_path}")

    return csv_path, txt_path


def main():
    cfg = WorkflowConfig()
    fig_dir = get_figure_dir(cfg)

    crit_df = build_critical_slope_dataframe()

    write_critical_slope_summary(crit_df, fig_dir)
    plot_critical_slope_figure(crit_df, fig_dir)

    return crit_df


if __name__ == "__main__":
    crit_df = main()