# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 13:41:36 2026

@author: sdavilao
"""

# -*- coding: utf-8 -*-
"""
10_plot_cumulative_soil_depth.py

Plot cumulative:
    1. Soil production
    2. Soil deposition
    3. Total soil depth

inside a buffered hollow through time.

Requires GeoTIFF outputs from the soil transport model.
"""

from pathlib import Path
import re

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.features import geometry_mask

from config import WorkflowConfig
from plot_helpers import get_figure_dir


# -----------------------------
# Plot formatting
# -----------------------------
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 10,
    "legend.title_fontsize": 12,
})

SAVE_FIGURE = True

def build_point_buffer_mask(
    dem_path,
    point_shapefile,
    target_point_id,
    buffer_distance,
    id_field="id",
):
    """
    Create a boolean mask for one buffered hollow point.
    """

    with rasterio.open(dem_path) as src:
        transform = src.transform
        out_shape = src.read(1).shape
        dem_crs = src.crs

    gdf = gpd.read_file(point_shapefile)

    if id_field not in gdf.columns:
        raise ValueError(
            f"Field '{id_field}' not found.\n"
            f"Available fields: {list(gdf.columns)}"
        )

    point = gdf[gdf[id_field] == target_point_id]

    if point.empty:
        raise ValueError(
            f"{id_field}={target_point_id} not found in shapefile."
        )

    if point.crs != dem_crs:
        point = point.to_crs(dem_crs)

    buffered_geom = point.geometry.buffer(buffer_distance)

    mask_raw = geometry_mask(
        buffered_geom,
        transform=transform,
        invert=True,
        out_shape=out_shape,
    )

    # Match np.flipud() used in saved rasters
    return np.flipud(mask_raw)


def load_masked_tif(path, buffer_mask):
    """
    Load raster and apply hollow mask.
    """

    with rasterio.open(path) as src:
        arr = src.read(1).astype(float)
        nodata = src.nodata

    arr = np.flipud(arr)

    if nodata is not None:
        arr[arr == nodata] = np.nan

    return np.where(buffer_mask, arr, np.nan)


def collect_mean_timeseries(cfg, buffer_mask):
    """
    Collect mean values inside hollow buffer through time.
    """

    base = cfg.basename

    pattern_prod = re.compile(
        rf"{base}_production_rate_(\d+)yrs"
    )

    pattern_dz = re.compile(
        rf"{base}_change_in_elevation_(\d+)yrs"
    )

    pattern_depth = re.compile(
        rf"{base}_total_soil_depth_(\d+)yrs"
    )

    mean_prod = {}
    mean_dz = {}
    mean_depth = {}

    for tif_path in Path(cfg.tif_dir).glob("*.tif"):

        fname = tif_path.name

        match = pattern_prod.search(fname)
        if match:
            year = int(match.group(1))
            mean_prod[year] = np.nanmean(
                load_masked_tif(tif_path, buffer_mask)
            )
            continue

        match = pattern_dz.search(fname)
        if match:
            year = int(match.group(1))
            mean_dz[year] = np.nanmean(
                load_masked_tif(tif_path, buffer_mask)
            )
            continue

        match = pattern_depth.search(fname)
        if match:
            year = int(match.group(1))
            mean_depth[year] = np.nanmean(
                load_masked_tif(tif_path, buffer_mask)
            )

    return mean_prod, mean_dz, mean_depth


def plot_cumulative_soil_depth(
    times,
    cumulative_production,
    cumulative_deposition,
    soil_depth,
    fig_dir,
    save_figure=True,
):

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.set_xlim(0, 2500)
    ax.set_ylim(0, 1.2)

    ax.plot(
        times,
        cumulative_production,
        marker="o",
        label="Cumulative soil produced (m)",
    )

    ax.plot(
        times,
        cumulative_deposition,
        marker="o",
        label="Cumulative soil deposited (m)",
    )

    ax.plot(
        times,
        soil_depth,
        marker="o",
        label="Total soil depth (m)",
    )

    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Soil depth (m)\ninside hollow")

    ax.grid(True, linestyle="--", alpha=0.5)

    ax.legend()

    fig.tight_layout()

    if save_figure:
        fig.savefig(
            fig_dir / "c_soil.png",
            dpi=450,
            bbox_inches="tight",
        )

    if save_figure:
        fig.savefig(fig_dir / "c_soil.png", dpi=450, bbox_inches="tight")

    plt.show()

def main():

    cfg = WorkflowConfig()
    cfg.make_dirs()

    # -----------------------------
    # User settings
    # -----------------------------
    target_point_id = 1
    id_field = "id"

    # -----------------------------
    # Build hollow mask
    # -----------------------------
    buffer_mask = build_point_buffer_mask(
        dem_path=cfg.input_tiff_path,
        point_shapefile=cfg.points_shp_path,
        target_point_id=target_point_id,
        buffer_distance=cfg.hollow_buffer_distance,
        id_field=id_field,
    )

    # -----------------------------
    # Load timeseries
    # -----------------------------
    mean_prod, mean_dz, mean_depth = collect_mean_timeseries(
        cfg,
        buffer_mask,
    )

    times = sorted(
        set(mean_prod.keys())
        | set(mean_dz.keys())
        | set(mean_depth.keys())
    )

    times_arr = np.array(times, dtype=float)

    prod_ts = np.array(
        [mean_prod.get(t, np.nan) for t in times]
    )

    dz_ts = np.array(
        [mean_dz.get(t, np.nan) for t in times]
    )

    depth_ts = np.array(
        [mean_depth.get(t, np.nan) for t in times]
    )

    cumulative_prod = np.nancumsum(prod_ts)
    cumulative_dz = np.nancumsum(dz_ts)

    # Start curves at t = 0
    times_plot = np.insert(times_arr, 0, 0.0)

    cumulative_prod_plot = np.insert(
        cumulative_prod,
        0,
        0.0,
    )

    cumulative_dz_plot = np.insert(
        cumulative_dz,
        0,
        0.0,
    )

    depth_plot = np.insert(
        depth_ts,
        0,
        0.0,
    )

    # -----------------------------
    # Save figure
    # -----------------------------
    SAVE_FIGURE = True
    fig_dir = get_figure_dir(cfg)
    
    plot_cumulative_soil_depth(
        times=times_plot,
        cumulative_production=cumulative_prod_plot,
        cumulative_deposition=cumulative_dz_plot,
        soil_depth=depth_plot,
        fig_dir=fig_dir,
        save_figure=SAVE_FIGURE,
    )

if __name__ == "__main__":
    main()