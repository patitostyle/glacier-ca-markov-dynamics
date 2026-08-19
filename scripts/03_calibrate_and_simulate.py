"""
Run this THIRD. Loads the two aligned snapshots produced by
02_rasterize_snapshots.py, estimates the Markov transition matrix
between them, then uses ca_markov.simulate() to project the glacier
extent forward N_FUTURE_STEPS beyond the "newer" snapshot.

Each simulated step represents one snapshot interval (i.e. if the
older -> newer gap is ~time T, each future step is also ~T).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from ca_markov import transition_matrix, simulate, class_areas

BASE = Path(__file__).resolve().parent.parent
PROC = BASE / "data" / "processed"
REPORTS = BASE / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

N_CLASSES = 2  # non_glacier=0, glacier=1
N_FUTURE_STEPS = 5
WINDOW = 3

def main():
    before_path = PROC / "snapshot_before.npy"
    after_path = PROC / "snapshot_after.npy"
    if not before_path.exists() or not after_path.exists():
        print(f"Missing {before_path} or {after_path} -- run 02_rasterize_snapshots.py first.")
        raise SystemExit(1)

    before = np.load(before_path)
    after = np.load(after_path)
    assert before.shape == after.shape, "Snapshots must be on the same grid (re-check 02_rasterize_snapshots.py)."

    trans = transition_matrix(before, after, N_CLASSES)
    print("Estimated transition matrix (rows=class at t0, cols=class at t1):")
    print(trans)
    pd.DataFrame(trans, index=["non_glacier", "glacier"], columns=["non_glacier", "glacier"]).to_csv(
        REPORTS / "transition_matrix.csv"
    )

    grids = simulate(after, trans, N_CLASSES, N_FUTURE_STEPS, window=WINDOW, seed=42)

    rows = []
    areas_before = class_areas(before, N_CLASSES)
    rows.append({"step": -1, "label": "before (older snapshot)", "glacier_cells": areas_before[1], "non_glacier_cells": areas_before[0]})
    for i, g in enumerate(grids):
        label = "after (current snapshot)" if i == 0 else f"simulated +{i}"
        areas = class_areas(g, N_CLASSES)
        rows.append({"step": i, "label": label, "glacier_cells": areas[1], "non_glacier_cells": areas[0]})

    df = pd.DataFrame(rows)
    df.to_csv(REPORTS / "area_over_time.csv", index=False)
    print("\nGlacier area (cell counts) over time:")
    print(df.to_string(index=False))

    np.save(PROC / "simulated_grids.npy", np.stack(grids))
    print(f"\nSaved transition matrix, area-over-time table, and {len(grids)} simulated grids to {REPORTS} / {PROC}")

if __name__ == "__main__":
    main()
