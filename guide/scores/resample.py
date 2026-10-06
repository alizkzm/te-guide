"""Runtime resampling plugin for single-value transferability scores."""

from __future__ import annotations

import numpy as np

from guide.scores.base import ScoreNotApplicable

MAX_DRAWS = 32

MODE_DEFAULTS = {"variance": (20, 0.8), "richardson": (10, 0.5)}


def draw_elements(prefix: str, vals: list[float]) -> dict[str, float]:
    """Package per-draw values as elements so the hub stage can RANK them."""
    out = {prefix: float(np.mean(vals)),
           f"{prefix}_sd": float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)}
    for i, v in enumerate(vals[:MAX_DRAWS]):
        out[f"{prefix}_d{i:02d}"] = float(v)
    return out


def _stratified_subsample(y: np.ndarray, frac: float,
                          rng: np.random.RandomState) -> np.ndarray:
    """Indices of ~frac * N_c of every class (no replacement), so the score is evaluated at a strictly smaller n and no class can be lost (C is fixed)."""
    keep = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        k = min(len(idx), max(2, int(round(float(frac) * len(idx)))))
        keep.append(rng.choice(idx, size=k, replace=False))
    return np.sort(np.concatenate(keep))


def resample_score_draws(scalar_fn, probe, mode: str,
                         n_iter: int, seed: int, frac: float) -> list[float]:
    """Per-draw values of `scalar_fn` over `n_iter` stratified resamples."""
    y = np.asarray(probe.labels)
    rng = np.random.RandomState(int(seed))
    full = float(scalar_fn(probe)) if mode == "richardson" else 0.0

    vals: list[float] = []
    for _ in range(int(n_iter)):
        idx = _stratified_subsample(y, frac, rng)
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            v = float(scalar_fn(probe.subset(idx)))
        except Exception:
            continue
        vals.append(2.0 * full - v if mode == "richardson" else v)
    if not vals:
        raise ScoreNotApplicable("every resample was degenerate")
    return vals
