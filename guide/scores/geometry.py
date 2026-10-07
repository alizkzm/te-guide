"""Spectral-geometry TE scores: RankMe, feature stable rank, DISCO."""

from __future__ import annotations

import numpy as np

from guide.core.linalg import effective_rank, normalised_stable_rank
from guide.core.probe import ProbeData
from guide.core.registry import register
from guide.core.sampling import random_indices
from guide.scores.base import SingleElementScore, as_float_features


@register
class RankMe(SingleElementScore):
    name = "rankme"
    paper = "RankMe: Assessing SSL Representations by Their Rank (ICML 2023)"
    elements = ("rankme",)
    requires = ("features",)
    label_free = True

    def value(self, probe: ProbeData, **hp) -> float:
        return effective_rank(as_float_features(probe))


@register
class FeatureStableRank(SingleElementScore):
    """Stable rank of the feature matrix."""

    name = "srank_feat"
    paper = "stable rank of the target-feature matrix (label-free baseline)"
    elements = ("srank_feat",)
    requires = ("features",)
    label_free = True
    trivial = True

    def value(self, probe: ProbeData, **hp) -> float:
        return normalised_stable_rank(as_float_features(probe))


@register
class DISCO(SingleElementScore):
    name = "disco"
    paper = "Assessing Pre-Trained Models Through Distribution of Spectral Components (AAAI 2025)"
    elements = ("disco",)
    hparams = {"n_components": 64, "max_samples": 20000, "seed": 0}
    status = "ready"
    note = (
        "Re-implemented from the paper (no official code released): each spectral "
        "component is weighted by its own between/within-class separability and "
        "by its share of the spectral energy."
    )

    def value(self, probe: ProbeData, **hp) -> float:
        X, y = as_float_features(probe), probe.labels
        if X.shape[0] > int(hp["max_samples"]):
            idx = random_indices(X.shape[0], int(hp["max_samples"]), seed=hp["seed"])
            X, y = X[idx], y[idx]

        Xc = X - X.mean(axis=0, keepdims=True)
        U, S, _ = np.linalg.svd(Xc, full_matrices=False)
        k = int(min(hp["n_components"] or len(S), len(S)))
        S = S[:k]
        weights = S / (S.sum() + 1e-12)

        classes = np.unique(y)
        out = 0.0
        for g in range(k):
            comp = U[:, g] * S[g]
            between = float(np.var([comp[y == c].mean() for c in classes]))
            within = float(np.mean([comp[y == c].var() + 1e-8 for c in classes]))
            out += weights[g] * between / within
        return out
