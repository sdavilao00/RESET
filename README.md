# **RESET**: REcurring Soil Evacuation in Topographic Hollows

This repository contains a two-stage workflow for modeling hollow soil depth evolution and estimating hollow failure recurrence intervals. The workflow was designed for characteristic hollow DEM snippets from the Oregon Coast Range, where smaller DEM subsets (`extX`) were clipped around representative convergent hollows to reduce computational cost while preserving local hillslope and hollow geometry.

## Main Files

- `config.py`  
  Stores the project folder, extent name, CRS settings, model parameters, cohesion values, saturation values, and output folders. Edit this file first.

- `01_run_soil_transport.py`  
  Runs the soil transport/soil production model, writes time-stepped GeoTIFFs, and reprojects raster and shapefile outputs.

- `02_extract_and_calculate_RI.py`  
  Reads the reprojected soil-depth rasters, extracts soil depth by hollow and candidate buffer size, calculates factor of safety (FS), interpolates the first FS = 1 crossing, and saves optima buffer recurrence-interval outputs.

- `03_plot_ri.py`  
  Plots recurrence interval data and determines best-fit.

- `environment.yml`  
  Defines the conda environment needed to run the workflow.

  Information regarding necessary inputs and format can be found on the wiki page of this repository. In this wiki page, you can also find information on how to plot various outputs from this model.

  Example data and additional data can be found in example_data
