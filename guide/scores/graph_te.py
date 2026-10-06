"""graph_te : a two-graph transferability metric (this paper)."""

from __future__ import annotations

import numpy as np

from guide.core.probe import ProbeData
from guide.core.registry import register
from guide.scores.base import TEScore, as_float_features
from guide.scores.spectral import _build_graph


def _spec_entropy(vals) -> float:
    """Shannon entropy of a non-negative spectrum (nats)."""
    v = np.clip(np.asarray(vals, float), 0.0, None)
    s = v.sum()
    if s <= 0:
        return 0.0
    p = v / s
    p = p[p > 1e-12]
    return float(-(p * np.log(p)).sum())


def _eff_rank(vals) -> float:
    """exp(spectral entropy) = effective rank (SCALE-DEPENDENT: <= dimension)."""
    return float(np.exp(_spec_entropy(vals)))


def _decay_alpha(vals) -> float:
    """Power-law decay exponent of a spectrum: fit log(lambda_i) ~ -alpha log(i)."""
    v = np.sort(np.clip(np.asarray(vals, float), 0.0, None))[::-1]
    v = v[v > v[0] * 1e-3] if v.size and v[0] > 0 else v
    if v.size < 3:
        return 0.0
    i = np.arange(1, v.size + 1)
    return float(-np.polyfit(np.log(i), np.log(v), 1)[0])


def graph_te_score(X, y, mult: float = 2.0, max_n: int = 5000, n_local: int = 7,
                   dim_cap: int = 2000, seed: int = 0, graph: str = "knn",
                   knn_k: int = 15) -> dict[str, float]:
    X = np.asarray(X, dtype=np.float64)
    W, deg, yy, K, U, evals = _build_graph(X, y, max_n, n_local, dim_cap, seed,
                                           graph=graph, knn_k=knn_k)
    n = len(yy)

    Y = np.zeros((n, K))
    Y[np.arange(n), yy] = 1.0
    Yc = Y - Y.mean(axis=0)
    a2 = ((U.T @ Yc) ** 2).sum(axis=1)
    denom = (Yc ** 2).sum() + 1e-12
    k = int(min(max(1, round(mult * K)), U.shape[1]))
    sep = float(a2[:k].sum() / denom)

    same = np.array([W[i, yy == yy[i]].sum() for i in range(n)])
    sep_cut = float(np.mean(same / deg))
    E = (Y.T @ W @ Y); E = E / (E.sum() + 1e-12)
    ai = E.sum(axis=1); sa = float(ai @ ai)
    sep_assort = float((np.trace(E) - sa) / (1.0 - sa + 1e-12))

    D = X.shape[1]
    Xc = X - X.mean(axis=0)
    cov = (Xc.T @ Xc) / max(1, len(Xc) - 1)
    lam = np.linalg.eigvalsh(cov)
    rank = _eff_rank(lam)
    rank_norm = float(rank / D)
    rank_alpha = _decay_alpha(lam)
    rank_entropy = float(_spec_entropy(lam) / np.log(D))

    rank_graph = float(_eff_rank(np.clip(evals, 0.0, None)) / len(evals))

    return {"sep": sep, "sep_assort": sep_assort, "sep_cut": sep_cut,
            "rank_norm": rank_norm, "rank_alpha": rank_alpha,
            "rank_entropy": rank_entropy, "rank": rank, "rank_graph": rank_graph}


@register
class GraphTE(TEScore):
    """Two-graph transferability: sample-graph separability + neuron-graph rank."""

    name = "graph_te"
    paper = "two-graph transferability (this paper)"
    elements = ("sep", "sep_assort", "sep_cut", "rank_norm", "rank_alpha",
                "rank_entropy", "rank", "rank_graph")
    requires = ("features", "labels")
    hparams = {"mult": 2.0, "max_n": 5000, "n_local": 7, "dim_cap": 2000,
               "seed": 0, "graph": "knn", "knn_k": 15}
    note = (
        "Two cosine-similarity graphs from the feature matrix, read with SCALE-FREE "
        "descriptors (fractions / distribution shape, not counts) so nothing grows "
        "with the number of nodes.  SAMPLE graph -> separability: sep = label energy "
        "fraction in the low `mult*K` modes ([0,1]); sep_assort = chance-corrected "
        "label assortativity ([-1,1]).  NEURON graph -> richness (label-free, so it "
        "varies on the SSL hub where #params is constant): rank_norm = effective "
        "rank / D (dims used, (0,1]); rank_alpha = covariance-spectrum decay "
        "exponent (LOWER = richer); rank_entropy = normalised spectral entropy "
        "([0,1]).  Combined by a self-gated W (high when the pool separates the "
        "target, low at the floor) so richness takes over when sep is uninformative "
        "-- see the graph_te eval script.  Raw `rank` (=eff-rank, scale-DEPENDENT) "
        "and sep_cut/rank_graph are kept for ablation only."
    )

    def compute(self, probe: ProbeData, **hp) -> dict[str, float]:
        X, y = as_float_features(probe), probe.labels
        return graph_te_score(
            X, y, mult=float(hp["mult"]), max_n=int(hp["max_n"]),
            n_local=int(hp["n_local"]), dim_cap=int(hp["dim_cap"]),
            seed=int(hp["seed"]), graph=str(hp["graph"]), knn_k=int(hp["knn_k"]))
