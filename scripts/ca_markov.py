"""
Core CA-Markov engine: transition-probability matrix estimated from two
real classified snapshots (Markov component) combined with a spatial
neighborhood-suitability weight (Cellular Automata component), following
the same logic used in tools like IDRISI Land Change Modeler or the
MOLUSCE QGIS plugin, implemented here from scratch in numpy so it's not
tied to any GIS desktop software.

Two categorical raster snapshots in, N future snapshots out.
"""
import numpy as np
from scipy.ndimage import uniform_filter

def transition_matrix(before, after, n_classes):
    """P[i,j] = probability a cell in class i at t0 is in class j at t1."""
    mat = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        mask = before == i
        total = mask.sum()
        if total == 0:
            mat[i, i] = 1.0
            continue
        for j in range(n_classes):
            mat[i, j] = np.sum(mask & (after == j)) / total
    return mat

def neighborhood_suitability(grid, target_class, window=3):
    """Fraction of neighbors (in a window x window kernel) already in target_class -- cells surrounded by more of a class are more 'suitable' to transition into it, the classic CA neighborhood rule."""
    is_target = (grid == target_class).astype(float)
    return uniform_filter(is_target, size=window, mode="nearest")

def step(grid, trans_matrix, n_classes, window=3, rng=None):
    """One CA-Markov step: for each cell, transition probability = Markov
    probability (from trans_matrix) scaled by local neighborhood
    suitability, renormalized, then sampled."""
    rng = rng or np.random.default_rng()
    suitability = np.stack([neighborhood_suitability(grid, c, window) for c in range(n_classes)], axis=-1)

    new_grid = grid.copy()
    for i in range(n_classes):
        mask = grid == i
        if not mask.any():
            continue
        probs = trans_matrix[i] * (suitability[mask] + 1e-6)
        probs = probs / probs.sum(axis=1, keepdims=True)
        cum = np.cumsum(probs, axis=1)
        r = rng.random(size=(mask.sum(), 1))
        choice = (r < cum).argmax(axis=1)
        new_grid[mask] = choice
    return new_grid

def simulate(initial_grid, trans_matrix, n_classes, n_steps, window=3, seed=42):
    rng = np.random.default_rng(seed)
    grids = [initial_grid.copy()]
    current = initial_grid.copy()
    for _ in range(n_steps):
        current = step(current, trans_matrix, n_classes, window, rng)
        grids.append(current.copy())
    return grids

def class_areas(grid, n_classes, cell_area=1.0):
    return {c: int((grid == c).sum()) * cell_area for c in range(n_classes)}
