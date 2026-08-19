"""
Run this FIRST. INAIGEM distributes the glacier inventory as an Esri
File Geodatabase (.gdb folder) with several feature layers (glaciers,
rock glaciers, glacial lakes, bofedales). This script lists whatever
vector files/layers are actually present under data/raw and prints
their columns + CRS, since layer names and attribute columns aren't
knowable for certain until you have the real download.

Note: a single INAIGEM download is one time snapshot (~2020 imagery).
A real CA-Markov calibration needs at least TWO time points to compute
a transition matrix -- if INAIGEM's geodatabase doesn't include a
historical layer, pair it with an older Randolph Glacier Inventory (RGI)
extent for the same cordillera as the "before" snapshot (see README).
"""
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw"

if not any(RAW.iterdir()):
    print(f"{RAW} is empty. Download the geodatabase first (see README).")
    raise SystemExit(1)

print(f"Contents of {RAW}:\n")
for p in sorted(RAW.rglob("*")):
    print(" ", p.relative_to(RAW))

try:
    import geopandas as gpd
    import fiona
except ImportError:
    print("\ngeopandas/fiona not installed -- run: pip install -r requirements.txt")
    raise SystemExit(0)

gdb_candidates = list(RAW.rglob("*.gdb")) + [p for p in RAW.iterdir() if p.is_dir() and p.suffix == ".gdb"]
shp_candidates = list(RAW.rglob("*.shp"))
geojson_candidates = list(RAW.rglob("*.geojson"))

for gdb in gdb_candidates:
    print(f"\n=== Geodatabase: {gdb.name} ===")
    layers = fiona.listlayers(str(gdb))
    print("Layers:", layers)
    for layer in layers:
        gdf = gpd.read_file(gdb, layer=layer, rows=5)
        print(f"\n  Layer '{layer}': columns = {list(gdf.columns)}")
        print(f"  CRS: {gdf.crs}")
        print(gdf.head(3))

for path in shp_candidates + geojson_candidates:
    print(f"\n=== {path.name} ===")
    gdf = gpd.read_file(path, rows=5)
    print("columns:", list(gdf.columns))
    print("CRS:", gdf.crs)
    print(gdf.head(3))
