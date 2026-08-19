"""
Synthetic end-to-end smoke test: builds two fake glacier-extent polygons
(a bigger "older" one and a smaller "newer" one, simulating retreat),
rasterizes them, fits the CA-Markov transition matrix, simulates forward,
and sanity-checks that glacier area keeps declining -- without needing
the real INAIGEM/RGI download.
"""
import sys
from pathlib import Path
import numpy as np
import geopandas as gpd
from shapely.geometry import box

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))

from ca_markov import transition_matrix, simulate, class_areas

def make_synthetic_shapefiles(tmp_dir):
    tmp_dir.mkdir(parents=True, exist_ok=True)
    crs = "EPSG:32718"  # UTM 18S, plausible for Peru
    older = gpd.GeoDataFrame({"geometry": [box(0, 0, 1000, 1000)]}, crs=crs)
    newer = gpd.GeoDataFrame({"geometry": [box(200, 200, 800, 800)]}, crs=crs)
    older_path = tmp_dir / "synthetic_older.shp"
    newer_path = tmp_dir / "synthetic_newer.shp"
    older.to_file(older_path)
    newer.to_file(newer_path)
    return older_path, newer_path

def test_rasterize_and_transition():
    tmp_dir = BASE / "tests" / "_tmp_smoke"
    older_path, newer_path = make_synthetic_shapefiles(tmp_dir)

    older_gdf = gpd.read_file(older_path)
    newer_gdf = gpd.read_file(newer_path)

    from rasterio.features import rasterize
    from rasterio.transform import from_bounds

    b1, b2 = older_gdf.total_bounds, newer_gdf.total_bounds
    bounds = (min(b1[0], b2[0]), min(b1[1], b2[1]), max(b1[2], b2[2]), max(b1[3], b2[3]))
    cell_size = 50
    width = int((bounds[2] - bounds[0]) / cell_size)
    height = int((bounds[3] - bounds[1]) / cell_size)
    transform = from_bounds(*bounds, width, height)

    before = rasterize([(g, 1) for g in older_gdf.geometry], out_shape=(height, width), transform=transform, fill=0, dtype="uint8")
    after = rasterize([(g, 1) for g in newer_gdf.geometry], out_shape=(height, width), transform=transform, fill=0, dtype="uint8")

    assert (before == 1).sum() > (after == 1).sum(), "Synthetic 'older' extent should be bigger than 'newer' (retreat)."

    n_classes = 2
    trans = transition_matrix(before, after, n_classes)
    assert np.allclose(trans.sum(axis=1), 1.0), "Transition matrix rows must sum to 1."
    assert trans[1, 0] > 0, "Some glacier cells should have transitioned to non-glacier (retreat signal)."

    grids = simulate(after, trans, n_classes, n_steps=5, window=3, seed=42)
    areas = [class_areas(g, n_classes)[1] for g in grids]
    print("Simulated glacier cell counts over steps:", areas)
    assert areas[-1] <= areas[0], "Glacier area should not increase under a retreat-favoring transition matrix."

    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("OK: rasterize + transition_matrix + simulate all behave sensibly on synthetic retreat data.")

if __name__ == "__main__":
    test_rasterize_and_transition()
