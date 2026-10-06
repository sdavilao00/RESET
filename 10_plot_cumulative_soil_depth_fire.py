# -*- coding: utf-8 -*-
"""Plot hollow soil-depth budgets from the FIRE companion driver's outputs.

Keep beside config.py and plot_helpers.py. Run after the fire simulation.
Uses native-CRS TIFFs under cfg.tif_dir/fire_scenario, not reprojected TIFFs.
The fire driver writes interval totals (meters); sum these directly, without
multiplying by dt. Older background-driver outputs have different semantics.
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

TARGET_POINT_ID = 1
ID_FIELD = 'id'
SAVE_FIGURE = True
# Set None to show the entire simulation, or choose a zoom such as (40, 65).
X_LIMITS = (0, 500)
FIRST_FIRE = 50.0
FIRE_INTERVAL = 100.0
FIRE_DURATION = 3.0

plt.rcParams.update({
    'font.size': 16, 'axes.titlesize': 16, 'axes.labelsize': 16,
    'xtick.labelsize': 16, 'ytick.labelsize': 16, 'legend.fontsize': 10,
})


def build_point_buffer_mask(dem_path, point_shapefile, target_point_id,
                            buffer_distance, id_field='id'):
    with rasterio.open(dem_path) as src:
        if src.crs is None or not src.crs.is_projected:
            raise ValueError('DEM requires a projected CRS.')
        crs, transform, shape = src.crs, src.transform, src.shape
        _, horizontal_to_m = crs.linear_units_factor
    points = gpd.read_file(point_shapefile)
    if points.crs is None:
        raise ValueError('Point shapefile requires CRS metadata.')
    if id_field not in points.columns:
        raise ValueError(f'{id_field!r} missing; available fields: {list(points.columns)}')
    point = points.loc[points[id_field] == target_point_id]
    if len(point) != 1:
        raise ValueError(f'Expected one point for {id_field}={target_point_id}; found {len(point)}.')
    point = point.to_crs(crs)
    polygons = point.geometry.buffer(buffer_distance / horizontal_to_m)
    mask = geometry_mask(list(polygons), transform=transform,
                         out_shape=shape, invert=True)
    if not mask.any():
        raise ValueError('Hollow buffer contains no DEM cells.')
    # Both TIFF data and geometry mask are already north-up. Do not flip either.
    return mask, (crs, transform, shape)


def load_masked_mean(path, buffer_mask, reference):
    with rasterio.open(path) as src:
        if (src.crs, src.transform, src.shape) != reference:
            raise ValueError(f'Raster grid differs from DEM: {path}')
        values = src.read(1, masked=True).astype(float).filled(np.nan)
    selected = values[buffer_mask]
    if not np.isfinite(selected).all():
        raise ValueError(f'Hollow contains nodata/nonfinite cells in {path}; use a valid buffer.')
    return float(selected.mean())


def collect_mean_timeseries(tif_dir, base, buffer_mask, reference):
    patterns = {
        field: re.compile(rf'^{re.escape(base)}_{field}_([0-9]+(?:\.[0-9]+)?)yrs\.tif$')
        for field in ('production_rate', 'change_in_elevation', 'total_soil_depth')
    }
    means = {field: {} for field in patterns}
    for path in sorted(Path(tif_dir).glob('*.tif')):
        for field, pattern in patterns.items():
            match = pattern.fullmatch(path.name)
            if match:
                year = float(match.group(1))
                if year in means[field]:
                    raise ValueError(f'Duplicate {field} output at {year:g} yr.')
                means[field][year] = load_masked_mean(path, buffer_mask, reference)
                break
    sets = [set(series) for series in means.values()]
    if not sets[0]:
        raise FileNotFoundError(f'No matching fire output TIFFs for {base!r} in {tif_dir}')
    if not all(years == sets[0] for years in sets):
        raise ValueError('Production, elevation-change and depth output years do not match. '
                         'Finish the simulation before plotting; do not silently skip missing files.')
    if 0.0 not in sets[0]:
        raise ValueError('Missing year-0 state; rerun the updated fire driver.')
    times = np.array(sorted(sets[0]))
    produced = np.array([means['production_rate'][t] for t in times])
    deposited = np.array([means['change_in_elevation'][t] for t in times])
    depth = np.array([means['total_soil_depth'][t] for t in times])
    if not np.allclose([produced[0], deposited[0]], 0):
        raise ValueError('Year-0 increments must be zero.')
    cumulative_prod = np.cumsum(produced)
    cumulative_dz = np.cumsum(deposited)
    # Soil budget: H(t) = H(0) + summed production + summed net transport.
    if not np.allclose(depth, depth[0] + cumulative_prod + cumulative_dz,
                       rtol=2e-5, atol=2e-6):
        raise ValueError('Soil-depth budget does not close. Check for stale/mixed outputs '
                         'or missing interval-total rasters.')
    return times, cumulative_prod, cumulative_dz, depth


def plot_cumulative_soil_depth(times, cumulative_production,
                               cumulative_deposition, soil_depth, fig_dir,
                               save_figure=True):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor('#f0f0f0')
    curves = (
        (cumulative_production, '#E69F00', 'Cumulative soil produced'),
        (cumulative_deposition, '#56B4E9', 'Cumulative net deposition/erosion'),
        (soil_depth, '#009E73', 'Total soil depth'),
    )
    for values, color, label in curves:
        ax.plot(times, values, color=color, lw=1.8, marker='o',
                markersize=2.5, label=label, zorder=3)
    limits = X_LIMITS if X_LIMITS is not None else (0, times[-1])
    if limits[1] <= limits[0]:
        raise ValueError('X_LIMITS must have increasing bounds.')
    ax.set_xlim(*limits)
    ax.set_xlim(*limits)
    ax.set_ylim(0, 0.6)  # Minimum and maximum thickness in meters
    first_visible = max(0, int(np.floor((limits[0] - FIRST_FIRE) / FIRE_INTERVAL)))
    label_added = False
    for start in np.arange(FIRST_FIRE + first_visible * FIRE_INTERVAL,
                           min(limits[1], times[-1]) + 1e-9, FIRE_INTERVAL):
        if start + FIRE_DURATION < limits[0]:
            continue
        ax.axvspan(start, min(start + FIRE_DURATION, times[-1]),
                   color='#D55E00', alpha=0.18, linewidth=0,
                   label='Fire transport period (3 yr)' if not label_added else None)
        label_added = True
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Mean thickness inside hollow (m)')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend()
    fig.tight_layout()
    if save_figure:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
        output = fig_dir / f'c_soil_fire_point_{TARGET_POINT_ID}.png'
        fig.savefig(output, dpi=450, bbox_inches='tight')
        print(f'Saved: {output}')
    plt.show()


def main():
    cfg = WorkflowConfig()
    cfg.make_dirs()
    tif_dir = Path(cfg.tif_dir) / 'fire_scenario'
    # This matches the fire driver's basename exactly.
    base = Path(cfg.input_tiff_path).stem
    mask, reference = build_point_buffer_mask(
        cfg.input_tiff_path, cfg.points_shp_path, TARGET_POINT_ID,
        cfg.hollow_buffer_distance, ID_FIELD)
    times, production, deposition, depth = collect_mean_timeseries(
        tif_dir, base, mask, reference)
    expected = {start + offset for start in np.arange(FIRST_FIRE, times[-1], FIRE_INTERVAL)
                for offset in (0, 1, 2, 3) if start + offset <= times[-1]}
    missing = sorted(expected - set(times))
    if missing:
        raise ValueError(f'Missing fire-year outputs: {missing[:12]}. '
                         'Rerun the latest fire driver before plotting.')
    print(f'Loaded {len(times)} times, including fire-year outputs; '
          f'initial mean depth = {depth[0]:.4f} m.')
    fig_dir = Path(get_figure_dir(cfg)) / 'fire_scenario'
    plot_cumulative_soil_depth(times, production, deposition, depth,
                               fig_dir, SAVE_FIGURE)


if __name__ == '__main__':
    main()
