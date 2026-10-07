"""SA - Spread & Attract feature-space perturbation."""

from __future__ import annotations

import numpy as np

from guide.core.linalg import pca_reduce
from guide.core.probe import ProbeData
from guide.core.registry import register
from guide.scores.base import TEScore, as_float_features
from guide.scores.classic import logme_score
from guide.scores.energy import lda_score
from guide.scores.separability import SFDA


def check_sa(X: np.ndarray) -> bool:
    """Reference gate: True when the adaptive shrinkage hits its 1e-10 floor."""
    cov = np.atleast_2d(np.cov(np.asarray(X).T, bias=True))
    evals = np.linalg.eigvals(cov)
    top2 = np.sort(np.real(evals))[::-1][:2]
    shrinkage = max(float(np.exp(-5 * np.mean(top2))), 1e-10)
    return shrinkage < 1e-9


def spread_points(cluster: np.ndarray, centroid: np.ndarray, force: float = 1.0) -> np.ndarray:
    """Push every point of a class radially outward from its centroid."""
    direction = cluster - centroid
    norm = np.linalg.norm(direction, axis=1, keepdims=True)
    return cluster + force * direction / (norm + 1e-12)


def attract(X: np.ndarray, y: np.ndarray, alpha: float = 0.005, sigma: float = 0.6):
    """Pull class clusters towards each other proportionally to their gap."""
    classes = np.unique(y)
    n_classes = len(classes)
    means = np.stack([X[y == c].mean(axis=0) for c in classes])
    varis = np.stack([X[y == c].var(axis=0) for c in classes])
    radius = np.sqrt(varis.sum(axis=1))

    gi, gj = np.meshgrid(np.arange(n_classes), np.arange(n_classes))
    mask = np.triu(gi != gj, k=1)
    ii, jj = gi[mask], gj[mask]

    dist = means[ii] - means[jj]
    d = np.sqrt((dist ** 2).sum(axis=1))
    gap = (d - sigma * (radius[ii] + radius[jj])).reshape(-1, 1)
    direction = dist / (d.reshape(-1, 1) + 1e-12)

    disp = np.zeros((n_classes, n_classes, X.shape[1]))
    disp[mask] = -direction * alpha * gap
    disp = disp - np.transpose(disp, (1, 0, 2))
    return X + disp.sum(axis=1)[y]


def sa_remap(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.005,
    sigma: float = 0.6,
    pca_dim: int = 64,
    force_apply: bool = False,
    faithful: bool = True,
) -> tuple[np.ndarray, bool]:
    """Returns (remapped features, whether the perturbation actually fired)."""
    if not (force_apply or check_sa(X)):
        return X, False

    X_reduce = pca_reduce(X, pca_dim)
    X_spread = np.zeros_like(X_reduce)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        cluster = X_reduce[idx]
        X_spread[idx] = spread_points(cluster, cluster.mean(axis=0), force=1.0)

    base = X_reduce if faithful else X_spread
    return attract(base, y, alpha=alpha, sigma=sigma), True


@register
class SA(TEScore):
    name = "sa"
    paper = "Feature-space Perturbation: A Panacea to Enhanced TE (WACV 2025)"
    elements = ("sa_sfda", "sa_logme", "sa_lda")
    hparams = {
        "alpha": 0.005,
        "sigma": 0.6,
        "pca_dim": 64,
        "force_apply": False,
        "faithful": True,
        "base_metrics": ("sfda", "logme", "lda"),
    }
    note = (
        "Spread-and-attract remapping + a base metric. The paper's headline "
        "combination is SA+SFDA; SA+LogME and SA+LDA are reported too. When the "
        "`check_sa` gate does not fire the features pass through unchanged, so "
        "sa_* equals the plain metric - `perturbed` in the diagnostics tells you "
        "which happened."
    )

    def compute(self, probe: ProbeData, **hp) -> dict[str, float]:
        X, y = as_float_features(probe), probe.labels
        Xr, fired = sa_remap(
            X,
            y,
            alpha=float(hp["alpha"]),
            sigma=float(hp["sigma"]),
            pca_dim=int(hp["pca_dim"]),
            force_apply=bool(hp["force_apply"]),
            faithful=bool(hp["faithful"]),
        )

        out: dict[str, float] = {"perturbed": float(fired)}
        bases = tuple(hp["base_metrics"])
        if "sfda" in bases:
            out["sa_sfda"] = float(
                SFDA().value(ProbeData(probe.model_name, probe.dataset, features=Xr, labels=y))
            )
        if "logme" in bases:
            out["sa_logme"] = logme_score(Xr, y)
        if "lda" in bases:
            out["sa_lda"] = lda_score(Xr, y)
        return out
