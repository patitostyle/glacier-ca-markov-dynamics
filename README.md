# glacier-ca-markov-dynamics

Cellular Automata + Markov chain (CA-Markov) simulation of glacier
retreat in the Peruvian Andes, built from two time-separated
glacier-extent snapshots (RGI 1998 vs. INAIGEM 2020). Implemented from
scratch in numpy/scipy -- no GIS desktop software (IDRISI, MOLUSCE)
required.

## Why this project

This portfolio already covers time series forecasting (SARIMA/LSTM),
supervised ML/DL benchmarking, and remote-sensing feature extraction.
Cellular automata add a different modeling paradigm: instead of
learning from many past observations of one variable, CA-Markov models
**spatial transition dynamics** from as few as two classified
snapshots -- the standard approach for land-use/land-cover change and
glacier-extent projection in the remote sensing literature.

## Data

**Older snapshot (1998): Randolph Glacier Inventory (RGI) 7.0, region
16 "Low Latitudes"**, fetched via the Earth Engine community dataset
`projects/sat-io/open-datasets/RGI/RGI_VECTOR_MERGED_V7`, filtered to
Peru. Reproduce with this script in the Earth Engine Code Editor
(https://code.earthengine.google.com):

var rgi = ee.FeatureCollection('projects/sat-io/open-datasets/RGI/RGI_VECTOR_MERGED_V7');
var peru = ee.FeatureCollection('FAO/GAUL/2015/level0')
  .filter(ee.Filter.eq('ADM0_NAME', 'Peru'));

var rgiPeru = rgi.filterBounds(peru.geometry());

// RGI includes a few 'nominal' glaciers mapped as lines instead of
// polygons -- Shapefile export requires one geometry type, so filter
// them out first.
var rgiPeruPolygons = rgiPeru
  .map(function(f) { return f.set('geom_type', f.geometry().type()); })
  .filter(ee.Filter.eq('geom_type', 'Polygon'));

Export.table.toDrive({
  collection: rgiPeruPolygons,
  description: 'rgi_peru_glaciers_1998',
  fileFormat: 'SHP'
});

Run it, go to **Tasks** -> **Run**, download the resulting shapefile
(6 files: .shp/.shx/.dbf/.prj/.cpg/.fix) from Google Drive into
`data/raw/`.

**Newer snapshot (2020): INAIGEM National Glacier Inventory** -- Peru's
Instituto Nacional de Investigación en Glaciares y Ecosistemas de
Montaña distributes this as an Esri File Geodatabase via ArcGIS Online:
<https://www.arcgis.com/home/item.html?id=ea055f40a9084e05b2fc7fd0544468fc>
Download and place the `.gdb` folder in `data/raw/`. It has two
layers -- `Glaciares_2020` (used here) and `Lagunas_2020` (glacial
lakes, not used).

A ~22-year gap between snapshots (1998 vs. 2020) gives the Markov
transition matrix a large, honest signal to estimate from, rather than
two dates close enough that observed change is mostly noise.

## Methodology

1. `01_explore_data.py` -- lists whatever's under `data/raw/`, and for
   `.gdb`/`.shp`/`.geojson` files prints layer names, columns, and CRS.
2. `02_rasterize_snapshots.py` -- reprojects both snapshots to a shared
   UTM CRS, flattens INAIGEM's 3D (Z) geometries, and rasterizes both
   onto **the same grid** (union of both extents, 100m cells by
   default) as a binary glacier/non-glacier categorical array.
3. `03_calibrate_and_simulate.py` -- estimates the Markov transition
   matrix `P[i,j]` = probability a cell in class `i` at t0 is in class
   `j` at t1, directly from the two observed snapshots. Then simulates
   forward N future steps: at each step, transition probability for a
   cell = Markov probability **scaled by local neighborhood
   suitability** (fraction of its 3x3 neighbors already in the target
   class) -- the classic CA rule that transitions cluster spatially
   rather than happening independently per pixel.
4. `04_visualize_results.py` -- map panels (before / current /
   simulated future) and a glacier-area-over-time plot.

## Core engine (`scripts/ca_markov.py`)

Reusable, dependency-light (numpy + scipy only) functions:
`transition_matrix()`, `neighborhood_suitability()`, `step()`,
`simulate()`, `class_areas()`.

## Results (real data: RGI 1998 vs. INAIGEM 2020, 100m grid, 10325x9396 cells)

Observed change: glacier extent fell from **136,763 to 115,837 cells**
(~15.3% loss) over 22 years. Estimated transition matrix:

|  | -> non-glacier | -> glacier |
|---|---|---|
| glacier (t0) | 27.0% | 73.0% |
| non-glacier (t0) | 99.98% | 0.02% |

Projected forward 5 steps (~22 years each) at this same historical
transition rate:

| Step | Glacier cells |
|---|---|
| -1 (1998) | 136,763 |
| 0 (2020) | 115,837 |
| +1 | 103,619 |
| +2 | 92,473 |
| +3 | 82,292 |
| +4 | 73,134 |
| +5 | 65,074 |

See `reports/figures/` for the map panels and area-over-time plot.

## Honest note on this repo's state

Ran end-to-end against the real RGI 1998 + INAIGEM 2020 snapshots (see
Results above). `tests/run_smoke_test.py` remains in the repo as a
fast synthetic-data regression check on the core `ca_markov.py` logic.

## Setup

pip install -r requirements.txt

## Limitations

- A 2-class (glacier / non-glacier) transition matrix from only two
  snapshots is a coarse approximation -- no elevation, slope, or
  aspect covariates in the suitability term (only spatial
  neighborhood).
- The 5-step projection assumes the 1998-2020 transition rate stays
  constant into the future, a standard CA-Markov simplification, not a
  physical ice-flow/mass-balance model -- real glacier retreat is
  driven by climate forcing that can accelerate or plateau, which this
  model cannot anticipate.