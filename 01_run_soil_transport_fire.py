# -*- coding: utf-8 -*-
"""Fire-enabled companion to 01_run_soil_transport.py.

Keep this file beside the original driver and config.py; run THIS file.
Uses the original driver's reprojection and plotting helpers.
All model coordinates, elevations and soil depths are in meters internally.
GeoTIFFs retain the input raster's horizontal CRS; elevation values are meters.
Fire alters transport parameters only (no soil evacuation or production change).
Change/production rasters contain totals since the previous saved output;
production_rate retains the legacy name but contains produced thickness (m).
"""
import csv
import importlib.util
from pathlib import Path

BACKGROUND_K = 0.0042  # m2/yr
BACKGROUND_SC = 1.25  # dimensionless slope gradient
FIRE_K = 0.13  # m2/yr
FIRE_SC = 1.03
FIRST_FIRE = 50.0  # years
FIRE_INTERVAL = 100.0
FIRE_DURATION = 3.0
FIRE_MAX_DT = 1.0  # production/transport coupling steps during fire (yr)
# None assumes vertical units match horizontal units, as in the OCR DEMs.
# Set to 1.0 for meter elevations in a feet-based horizontal CRS.
ELEVATION_TO_METERS = None
RUN_TRANSPORT = True
RUN_REPROJECTION = True


class ConfigOverrides:
    """Delegate config reads, keeping overrides off read-only properties."""

    def __init__(self, original):
        self._original = original

    def __getattr__(self, name):
        return getattr(self._original, name)


def scheduled_steps(target_time, max_dt, output_interval):
    """Yield (start, end, fire_active, save_output), aligned to every event.

    Fire windows are half-open: [50, 53), [150, 153), etc.
    Save at fire starts and every completed fire step, plus regular outputs.
    max_dt is a maximum, not a requirement; the final partial step is retained.
    """
    import math
    values = (target_time, max_dt, output_interval, FIRE_INTERVAL,
              FIRE_DURATION, FIRE_MAX_DT, FIRST_FIRE)
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Time settings must be finite.')
    if min(target_time, max_dt, output_interval, FIRE_MAX_DT) <= 0:
        raise ValueError('Target time and timestep/output settings must be positive.')
    if FIRST_FIRE < 0 or not 0 < FIRE_DURATION < FIRE_INTERVAL:
        raise ValueError('Require first fire >= 0 and 0 < duration < recurrence interval.')
    t = 0.0
    output_index = 1
    while t < target_time:
        if t < FIRST_FIRE:
            active = False
            event = FIRST_FIRE
        else:
            cycle = math.floor((t - FIRST_FIRE) / FIRE_INTERVAL)
            start = FIRST_FIRE + cycle * FIRE_INTERVAL
            active = t < start + FIRE_DURATION
            event = start + FIRE_DURATION if active else start + FIRE_INTERVAL
        next_output = output_index * output_interval
        end = min(target_time, event, next_output,
                  t + min(max_dt, FIRE_MAX_DT) if active else t + max_dt)
        if end <= t:
            raise RuntimeError('Schedule failed to advance.')
        save = active or end == event or end == next_output or end == target_time
        yield t, end, active, save
        if end == next_output:
            output_index += 1
        t = end


def load_original_driver():
    path = Path(__file__).resolve().with_name('01_run_soil_transport.py')
    if not path.exists():
        raise FileNotFoundError(f'Keep this companion beside the original driver: {path}')
    spec = importlib.util.spec_from_file_location('original_soil_transport', path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    return driver


def run_soil_transport_simulation(cfg, driver):
    import numpy as np
    import geopandas as gpd
    import rasterio
    from rasterio.features import geometry_mask
    from landlab import RasterModelGrid
    from landlab.components import TaylorNonLinearDiffuser
    from tqdm import tqdm

    cfg.make_dirs()
    # Avoid overwriting background-only model results.
    for attr in ('tif_dir', 'png_dir', 'asc_dir', 'reproj_tif_dir'):
        directory = Path(getattr(cfg, attr)) / 'fire_scenario'
        directory.mkdir(parents=True, exist_ok=True)
        setattr(cfg, attr, directory)
    cfg.K = BACKGROUND_K
    cfg.Sc = BACKGROUND_SC
    if min(cfg.pr, cfg.ps, cfg.h0) <= 0 or cfg.P0 < 0:
        raise ValueError('Require positive densities/h0 and nonnegative P0.')

    with rasterio.open(cfg.input_tiff_path) as src:
        if src.crs is None or not src.crs.is_projected:
            raise ValueError('DEM must have a projected CRS.')
        transform = src.transform
        if transform.b != 0 or transform.d != 0 or transform.a <= 0 or transform.e >= 0:
            raise ValueError('DEM must be north-up without rotation.')
        _, horizontal_to_m = src.crs.linear_units_factor
        vertical_to_m = (horizontal_to_m if ELEVATION_TO_METERS is None
                         else ELEVATION_TO_METERS)
        if vertical_to_m <= 0:
            raise ValueError('Elevation conversion factor must be positive.')
        dem = src.read(1, masked=True)
        valid_raster = ~np.ma.getmaskarray(dem) & np.isfinite(dem.data)
        meta = src.meta.copy()
        dem_crs = src.crs
        spacing = (abs(transform.e) * horizontal_to_m,
                   transform.a * horizontal_to_m)
    if not valid_raster.any():
        raise ValueError('DEM contains no valid cells.')
    grid = RasterModelGrid(dem.shape, xy_spacing=spacing)
    z_raster = np.where(valid_raster, dem.data * vertical_to_m, 0.0)
    z = grid.add_field('topographic__elevation', np.flipud(z_raster).ravel().copy(),
                       at='node', clobber=True)
    valid = np.flipud(valid_raster).ravel()
    grid.set_closed_boundaries_at_grid_edges(False, False, False, False)
    grid.status_at_node[~valid] = grid.BC_NODE_IS_CLOSED

    points = gpd.read_file(cfg.points_shp_path)
    if points.crs is None:
        raise ValueError('Hollow points require CRS metadata.')
    if points.empty or points.geometry.is_empty.any() or points.geometry.isna().any():
        raise ValueError('Hollow points must contain valid, nonempty geometries.')
    points = points.to_crs(dem_crs)
    # Buffer points ONCE. Config distance is interpreted as meters.
    polygons = points.geometry.buffer(cfg.hollow_buffer_distance / horizontal_to_m)
    buffered = points.copy()
    buffered.geometry = polygons
    buffered.to_file(Path(cfg.buffer_shp_path).with_name(
        Path(cfg.buffer_shp_path).stem + '_fire.shp'))
    hollow_raster = geometry_mask(list(polygons), out_shape=dem.shape,
                                 transform=transform, invert=True)
    hollow = np.flipud(hollow_raster).ravel() & valid
    h = grid.add_field('soil__depth', np.where(valid, 0.5, 0.0),
                       at='node', clobber=True)
    h[hollow] = 0.0
    print(f'Initialized {hollow.sum()} hollow cells at 0 m soil depth.')

    diffuser = None
    previous_mode = None
    base = Path(cfg.input_tiff_path).stem
    accum_z = np.zeros_like(h)
    accum_h = np.zeros_like(h)
    accum_p = np.zeros_like(h)
    meta.update(dtype='float32', count=1, compress='deflate', nodata=-9999.0)

    def save(values, name):
        raster = np.flipud(values.reshape(grid.shape)).astype('float32')
        raster[~valid_raster] = meta['nodata']
        with rasterio.open(Path(cfg.tif_dir) / name, 'w', **meta) as dst:
            dst.write(raster, 1)
            dst.update_tags(value_units='m', scenario='periodic_fire')

    def save_state(time):
        label = f'{time:g}'
        for values, field in ((accum_z, 'change_in_elevation'),
                              (accum_h, 'change_in_soil_depth'),
                              (h, 'total_soil_depth'),
                              (accum_p, 'production_rate')):
            save(values, f'{base}_{field}_{label}yrs.tif')
        save(z, f'{base}_{label}yrs_K{BACKGROUND_K}.tif')
        # Plot/ASCII helpers operate on the SI grid; no ASCII-to-TIFF parsing.
        driver.save_elevation_outputs(grid, base, label, cfg)

    save_state(0.0)
    history_path = Path(cfg.tif_dir) / f'{base}_fire_transport_history.csv'
    with history_path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['start_yr', 'end_yr', 'dt_yr', 'fire_active', 'K_m2_yr', 'Sc'])
        with tqdm(total=cfg.target_time, unit='yr', desc='Fire transport') as progress:
            for start, end, active, save_output in scheduled_steps(
                    float(cfg.target_time), float(cfg.dt), float(cfg.output_interval)):
                step_dt = end - start
                k, sc = (FIRE_K, FIRE_SC) if active else (BACKGROUND_K, BACKGROUND_SC)
                if active != previous_mode:
                    # Reconstruct using the public constructor; grid fields persist.
                    diffuser = TaylorNonLinearDiffuser(
                        grid, linear_diffusivity=k, slope_crit=sc,
                        dynamic_dt=True, nterms=2, if_unstable='raise')
                    previous_mode = active
                    print(f't={start:g} yr: K={k:g}, Sc={sc:g}, fire={active}')
                old_z = z.copy()
                old_h = h.copy()
                produced = (cfg.pr / cfg.ps) * cfg.P0 * np.exp(-old_h / cfg.h0) * step_dt
                produced[~valid] = 0.0
                diffuser.run_one_step(step_dt)
                if not np.isfinite(z[valid]).all():
                    raise RuntimeError(f'Nonfinite elevations at t={end:g} yr.')
                dz = z - old_z
                # Limit erosion using CURRENT available soil, including bare hollows.
                # This retains the driver's local clipping approach, not a
                # sediment-conserving, supply-limited flux formulation.
                dz[valid] = np.maximum(dz[valid], -old_h[valid])
                dz[~valid] = 0.0
                z[:] = old_z + dz
                h[:] = old_h + dz + produced
                accum_z += dz
                accum_h += h - old_h
                accum_p += produced
                writer.writerow([start, end, step_dt, int(active), k, sc])
                if save_output:
                    save_state(end)
                    accum_z.fill(0.0)
                    accum_h.fill(0.0)
                    accum_p.fill(0.0)
                progress.update(step_dt)
    print(f'Completed fire simulation. Parameter history: {history_path}')


def main():
    driver = load_original_driver()
    cfg = ConfigOverrides(driver.WorkflowConfig())
    if RUN_TRANSPORT:
        run_soil_transport_simulation(cfg, driver)
    else:
        cfg.tif_dir = Path(cfg.tif_dir) / 'fire_scenario'
        cfg.reproj_tif_dir = Path(cfg.reproj_tif_dir) / 'fire_scenario'
    if RUN_REPROJECTION:
        driver.reproject_all_tifs(cfg.tif_dir, cfg.reproj_tif_dir,
                                 target_crs=cfg.target_crs)
        driver.reproject_shapefiles_safe(
            cfg.base_dir, cfg.reproj_shp_dir, target_crs=cfg.target_crs,
            assumed_source_epsg=cfg.source_shp_epsg)


if __name__ == '__main__':
    main()
