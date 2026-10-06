"""Stratified sub-sampling - the `samples_per_class` axis of the benchmark."""

from __future__ import annotations

import numpy as np


def stratified_indices(
    y: np.ndarray, samples_per_class: str | int | None, seed: int = 42
) -> np.ndarray | None:
    """Indices keeping at most `samples_per_class` samples of each class."""
    if samples_per_class in (None, "all", "ALL", -1):
        return None
    k = int(samples_per_class)
    if k <= 0:
        return None

    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    keep: list[np.ndarray] = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        if idx.size > k:
            idx = rng.choice(idx, size=k, replace=False)
        keep.append(idx)
    out = np.sort(np.concatenate(keep))
    return out if out.size < y.size else None


def random_indices(n_total: int, n_keep: int, seed: int = 1234) -> np.ndarray:
    """Unstratified random subset."""
    if n_keep >= n_total:
        return np.arange(n_total)
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(n_total, size=n_keep, replace=False))


def resolve_n_samples(n_samples, n_total: int) -> int:
    """`"all"` / None / <=0 / >= n_total -> n_total; otherwise the int."""
    if n_samples is None:
        return int(n_total)
    if isinstance(n_samples, str):
        if n_samples.strip().lower() in ("all", "*", ""):
            return int(n_total)
        n_samples = int(n_samples)
    n = int(n_samples)
    return int(n_total) if n <= 0 or n >= n_total else n


def drop_singleton_classes(y: np.ndarray, min_count: int = 2) -> np.ndarray:
    """Indices of samples whose class has >= `min_count` members."""
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    good = set(classes[counts >= min_count].tolist())
    return np.flatnonzero(np.isin(y, list(good)))
