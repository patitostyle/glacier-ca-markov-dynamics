"""
Run this FOURTH. Plots (1) the older/current/simulated-future maps side
by side and (2) glacier area over time, from the outputs of
03_calibrate_and_simulate.py.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

BASE = Path(__file__).resolve().parent.parent
PROC = BASE / "data" / "processed"
REPORTS = BASE / "reports"
FIGS = REPORTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

CMAP = ListedColormap(["#f0ead6", "#2f7fbf"])  # non_glacier, glacier

def main():
    before = np.load(PROC / "snapshot_before.npy")
    grids = np.load(PROC / "simulated_grids.npy")  # [after, +1, +2, ..., +N]
    area_df = pd.read_csv(REPORTS / "area_over_time.csv")

    n_panels_maps = 3  # before, after (current), final simulated step
    fig, axes = plt.subplots(1, n_panels_maps, figsize=(4 * n_panels_maps, 4))
    axes[0].imshow(before, cmap=CMAP, vmin=0, vmax=1)
    axes[0].set_title("Before (older snapshot)")
    axes[1].imshow(grids[0], cmap=CMAP, vmin=0, vmax=1)
    axes[1].set_title("After (current snapshot)")
    axes[2].imshow(grids[-1], cmap=CMAP, vmin=0, vmax=1)
    axes[2].set_title(f"Simulated +{len(grids) - 1} steps")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("CA-Markov glacier extent: observed -> simulated")
    fig.tight_layout()
    fig.savefig(FIGS / "map_panels.png", dpi=150)
    print(f"Saved {FIGS / 'map_panels.png'}")

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(area_df["step"], area_df["glacier_cells"], marker="o")
    ax2.axvline(0, color="gray", linestyle="--", linewidth=1, label="current snapshot")
    ax2.set_xlabel("Step (snapshot intervals; negative = past, positive = simulated future)")
    ax2.set_ylabel("Glacier cells")
    ax2.set_title("Glacier extent over time (observed + CA-Markov projection)")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(FIGS / "area_over_time.png", dpi=150)
    print(f"Saved {FIGS / 'area_over_time.png'}")

if __name__ == "__main__":
    main()
