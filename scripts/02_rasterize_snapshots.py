"""
Rasterize two glacier-extent vector snapshots (an older reference and
the current INAIGEM inventory) onto a common categorical grid, so the
CA-Markov engine (ca_markov.py) has two aligned arrays to compute a
transition matrix from.

EDIT the paths/column names below once you've run 01_explore_data.py
and know what the real files/layers/attributes are actually called.
"""
from pathlib import Path
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw"
OUT = BASE / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# --- EDIT THESE once you know the real file/layer names ---
OLDER_SNAPSHOT_PATH = RAW / "rgi_peru_glaciers_1998.shp"   # RGI 7.0, region 16, Peru subset (1998 imagery)
NEWER_SNAPSHOT_PATH = RAW / "Inventario_2020.gdb"          # INAIGEM national glacier inventory (geodatabase)
NEWER_SNAPSHOT_LAYER = "Glaciares_2020"                    # layer name inside the .gdb (there's also "Lagunas_2020", not used)
CLASS_COLUMN = None    # set to a column name if the layer has multiple classes (glacier/rock-glacier/lake); else treated as binary glacier/non-glacier
CELL_SIZE_M = 100      # grid resolution in meters -- coarser is much lighter to simulate
# ------------------------------------------------------------

CLASSES = {"non_glacier": 0, "glacier": 1}

def load_utm(path, layer=None):
    gdf = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    gdf["geometry"] = gdf.geometry.force_2d()  # INAIGEM ships 3D (Z) geometries; rasterize only needs X/Y
    if gdf.crs and gdf.crs.is_geographic:
        gdf = gdf.to_crs(gdf.estimate_utm_crs())
    return gdf

def rasterize_snapshot(gdf, ref_transform, ref_shape):
    shapes = [(geom, CLASSES["glacier"]) for geom in gdf.geometry]
    grid = rasterize(shapes, out_shape=ref_shape, transform=ref_transform, fill=CLASSES["non_glacier"], dtype="uint8")
    return grid

def main():
    if not OLDER_SNAPSHOT_PATH.exists() or not NEWER_SNAPSHOT_PATH.exists():
        print(f"Expected snapshot files not found:\n  {OLDER_SNAPSHOT_PATH}\n  {NEWER_SNAPSHOT_PATH}")
        print("Edit the paths at the top of this script to match your real downloaded files.")
        raise SystemExit(1)

    older_gdf = load_utm(OLDER_SNAPSHOT_PATH)
    newer_gdf = load_utm(NEWER_SNAPSHOT_PATH, layer=NEWER_SNAPSHOT_LAYER).to_crs(older_gdf.crs)

    # shared bounding box (union of both) so both snapshots land on the SAME grid
    b1, b2 = older_gdf.total_bounds, newer_gdf.total_bounds
    bounds = (min(b1[0], b2[0]), min(b1[1], b2[1]), max(b1[2], b2[2]), max(b1[3], b2[3]))
    width = max(1, int((bounds[2] - bounds[0]) / CELL_SIZE_M))
    height = max(1, int((bounds[3] - bounds[1]) / CELL_SIZE_M))
    shape = (height, width)
    transform = from_bounds(*bounds, width, height)

    older_grid = rasterize_snapshot(older_gdf, transform, shape)
    newer_grid = rasterize_snapshot(newer_gdf, transform, shape)

    np.save(OUT / "snapshot_before.npy", older_grid)
    np.save(OUT / "snapshot_after.npy", newer_grid)
    print(f"Rasterized grid shape: {shape}  (cell size {CELL_SIZE_M}m)")
    print(f"Older snapshot glacier cells: {(older_grid == 1).sum()}")
    print(f"Newer snapshot glacier cells: {(newer_grid == 1).sum()}")
    print(f"Written to {OUT}")

if __name__ == "__main__":
    main()