"""Spectral transferability scores on the feature graph (this paper)."""

from __future__ import annotations

import numpy as np

from guide.core.probe import ProbeData
from guide.core.registry import register
from guide.scores.base import ScoreNotApplicable, TEScore, as_float_features


def _build_graph(X, y, max_n: int, n_local: int, dim_cap: int, seed: int,
                 graph: str = "knn", knn_k: int = 15):
    """Self-tuning affinity + spectral embedding, on a stratified subsample."""
    from scipy.linalg import eigh
    from sklearn.neighbors import NearestNeighbors

    X = np.asarray(X, dtype=np.float64)
    y = np.unique(np.asarray(y), return_inverse=True)[1]

    rng = np.random.RandomState(int(seed))
    if len(y) > max_n:
        keep = []
        for c in np.unique(y):
            ci = np.flatnonzero(y == c)
            take = max(2, int(round(max_n * len(ci) / len(y))))
            keep.append(rng.choice(ci, size=min(take, len(ci)), replace=False))
        sel = np.concatenate(keep)
        X, y = X[sel], np.unique(y[sel], return_inverse=True)[1]

    n = len(y)
    K = int(y.max()) + 1
    if K < 2 or n < 8:
        raise ScoreNotApplicable("need >= 2 classes and >= 8 samples")

    if graph == "knn":
        kk = int(min(knn_k, n - 1))
        dist, idx = NearestNeighbors(n_neighbors=kk + 1).fit(X).kneighbors(X)
        nl = int(min(n_local, kk))
        sig = dist[:, nl] + 1e-9
        rows = np.repeat(np.arange(n), kk)
        cols = idx[:, 1:kk + 1].ravel()
        d2 = dist[:, 1:kk + 1].ravel() ** 2
        w = np.exp(-d2 / (sig[rows] * sig[cols]))
        W = np.zeros((n, n))
        W[rows, cols] = w
        W = np.maximum(W, W.T)
    else:
        from scipy.spatial.distance import cdist
        nl = int(min(n_local, n - 1))
        dist = NearestNeighbors(n_neighbors=nl + 1).fit(X).kneighbors(X)[0]
        sig = dist[:, -1] + 1e-9
        W = np.exp(-cdist(X, X, "sqeuclidean") / (sig[:, None] * sig[None, :]))
        np.fill_diagonal(W, 0.0)

    deg = W.sum(axis=1) + 1e-12
    dinv = 1.0 / np.sqrt(deg)
    S = (W * dinv[:, None]) * dinv[None, :]

    dim = int(min(3 * K, n - 1, dim_cap))
    vals, vecs = eigh(S, subset_by_index=[n - dim, n - 1])
    U = vecs[:, ::-1]
    evals = vals[::-1]
    return W, deg, y, K, U, evals


def _shrink(Sw: np.ndarray, a: float) -> np.ndarray:
    d = Sw.shape[0]
    mu = np.trace(Sw) / d
    out = (1.0 - a) * Sw
    out.flat[:: d + 1] += a * mu
    return out


def _lda_sep(U: np.ndarray, y: np.ndarray, shrink: float) -> float:
    n, d = U.shape
    mu = U.mean(axis=0)
    Sw = np.zeros((d, d))
    Sb = np.zeros((d, d))
    for c in np.unique(y):
        Uc = U[y == c]
        m = Uc.mean(axis=0)
        Zc = Uc - m
        Sw += (Zc.T @ Zc) / n
        Sb += (len(Uc) / n) * np.outer(m - mu, m - mu)
    return float(np.trace(np.linalg.solve(_shrink(Sw, shrink), Sb)))


def _cluster_acc(pred: np.ndarray, y: np.ndarray) -> float:
    from scipy.optimize import linear_sum_assignment

    K = int(max(pred.max(), y.max())) + 1
    M = np.zeros((K, K))
    for p, t in zip(pred, y):
        M[p, t] += 1
    r, c = linear_sum_assignment(-M)
    return float(M[r, c].sum() / len(y))


def spectral_te_score(X, y, k_mults=(0.5, 1.0, 2.0), max_n: int = 5000,
                      n_local: int = 7, shrink: float = 0.2, dim_cap: int = 2000,
                      seed: int = 0, graph: str = "knn", knn_k: int = 15
                      ) -> dict[str, float]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import (adjusted_rand_score,
                                  normalized_mutual_info_score, silhouette_score)

    W, deg, y, K, U, _ = _build_graph(X, y, max_n, n_local, dim_cap, seed,
                                      graph=graph, knn_k=knn_k)
    dim = U.shape[1]

    ks = sorted({int(np.clip(round(m * K), 2, dim)) for m in k_mults})
    sep, sil, nmi, ari, acc = [], [], [], [], []
    for k in ks:
        T = U[:, :k].copy()
        T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
        sep.append(_lda_sep(T, y, shrink))
        try:
            sil.append(float(silhouette_score(T, y)))
        except Exception:
            pass
        pred = KMeans(n_clusters=k, n_init=3, random_state=int(seed)).fit_predict(T)
        nmi.append(float(normalized_mutual_info_score(y, pred)))
        ari.append(float(adjusted_rand_score(y, pred)))
        acc.append(_cluster_acc(pred, y))

    vol = np.array([deg[y == c].sum() for c in range(K)]) + 1e-12
    ncut = float(np.mean([W[np.ix_(y == c, y != c)].sum() / vol[c] for c in range(K)]))

    return {
        "sp_sep": float(np.mean(sep)),
        "sp_silhouette": float(np.mean(sil)) if sil else 0.0,
        "sp_nmi": float(np.mean(nmi)),
        "sp_ari": float(np.mean(ari)),
        "sp_acc": float(np.mean(acc)),
        "sp_ncut": ncut,
    }


@register
class SpectralTE(TEScore):
    """Class separability in the spectral embedding of the feature graph."""

    name = "spectral_te"
    paper = "spectral-embedding transferability (this paper)"
    elements = ("sp_sep", "sp_silhouette", "sp_nmi", "sp_ari", "sp_acc", "sp_ncut")
    requires = ("features", "labels")
    hparams = {"max_n": 5000, "n_local": 7, "shrink": 0.2, "dim_cap": 2000,
               "seed": 0, "graph": "knn", "knn_k": 15}
    note = (
        "Top Laplacian eigenvectors of a self-tuning kNN graph (the spectral-"
        "clustering embedding); class quality read there as LDA separability "
        "(sp_sep), silhouette, and k-means agreement (NMI/ARI/Hungarian acc), plus "
        "the normalised K-way cut of the true partition (sp_ncut, in [0,1], lower "
        "better).  Every metric is averaged over k in {K/2, K, 2K}.  Default "
        "recipe F = sp_sep + sp_silhouette + sp_ari (best of the graph/spectral "
        "scores across CNN/SSL/ViT); a geometric mean sqrt(sp_sep*sp_silhouette) "
        "is a slightly better, parameter-free two-term alternative."
    )

    def compute(self, probe: ProbeData, **hp) -> dict[str, float]:
        X, y = as_float_features(probe), probe.labels
        return spectral_te_score(
            X, y, max_n=int(hp["max_n"]), n_local=int(hp["n_local"]),
            shrink=float(hp["shrink"]), dim_cap=int(hp["dim_cap"]),
            seed=int(hp["seed"]), graph=str(hp["graph"]), knn_k=int(hp["knn_k"]))


LE_MULTS = (0.1, 0.5, 1.0, 1.5, 2.0, 2.5)


def graph_spectral_te_score(X, y, max_n: int = 5000, n_local: int = 7,
                            dim_cap: int = 2000, seed: int = 0,
                            heat_t: float = 1.0, graph: str = "knn",
                            knn_k: int = 15) -> dict[str, float]:
    """Collapse the affinity onto a K x K class graph and read exact scalars."""
    W, deg, y, K, U, evals = _build_graph(X, y, max_n, n_local, dim_cap, seed,
                                          graph=graph, knn_k=knn_k)
    n = len(y)

    Y = np.zeros((n, K))
    Y[np.arange(n), y] = 1.0
    Wcl = Y.T @ W @ Y
    vol = (Y.T @ deg) + 1e-12
    m2 = float(W.sum()) + 1e-12
    within = np.diag(Wcl).copy()

    gs_nassoc = float(np.mean(within / vol))
    gs_modularity = float(np.sum(within / m2 - (vol / m2) ** 2))

    C = Wcl - np.diag(within)
    Lc = np.diag(C.sum(axis=1)) - C
    gs_fiedler = float(np.sort(np.linalg.eigvalsh(Lc))[1])

    gs_leak = float(C.sum() / (within.sum() + 1e-12))

    iu = np.triu_indices(K, k=1)
    if len(iu[0]):
        crossn = Wcl[iu] / np.sqrt(vol[iu[0]] * vol[iu[1]])
        top = max(1, int(np.ceil(0.1 * len(crossn))))
        gs_cross = float(np.sort(crossn)[::-1][:top].mean())
    else:
        gs_cross = 0.0

    Yc = Y - Y.mean(axis=0)
    denom_y = (Yc ** 2).sum() + 1e-12
    A = U.T @ Yc
    a2 = (A ** 2).sum(axis=1)
    csum = np.cumsum(a2)
    dim = U.shape[1]
    le = {}
    for mlt in LE_MULTS:
        k = int(min(max(1, round(mlt * K)), dim))
        le[mlt] = float(csum[k - 1] / denom_y)
    gs_le010, gs_le050, gs_le100, gs_le150, gs_le200, gs_le250 = (le[m] for m in LE_MULTS)
    gs_labelenergy = gs_le100
    gs_le2k = gs_le200
    lap = 1.0 - evals
    band = max(lap[min(K, dim) - 1], 1e-6)
    w_heat = np.exp(-heat_t * lap / band)
    gs_heat = float((w_heat * a2).sum() / denom_y)

    dinv = 1.0 / np.sqrt(deg)
    SYc = (W @ (Yc * dinv[:, None])) * dinv[:, None]
    gs_smoothness = float((Yc * (Yc - SYc)).sum() / denom_y)

    return {
        "gs_nassoc": gs_nassoc,
        "gs_modularity": gs_modularity,
        "gs_fiedler": gs_fiedler,
        "gs_leak": gs_leak,
        "gs_cross": gs_cross,
        "gs_le010": gs_le010,
        "gs_le050": gs_le050,
        "gs_le100": gs_le100,
        "gs_le150": gs_le150,
        "gs_le200": gs_le200,
        "gs_le250": gs_le250,
        "gs_labelenergy": gs_labelenergy,
        "gs_le2k": gs_le2k,
        "gs_heat": gs_heat,
        "gs_smoothness": gs_smoothness,
    }


@register
class GraphSpectralTE(TEScore):
    """Exact label-based scalars off the K x K class graph of the feature graph."""

    name = "graph_spectral_te"
    paper = "class-graph spectral transferability (this paper)"
    elements = ("gs_nassoc", "gs_modularity", "gs_fiedler", "gs_leak", "gs_cross",
                "gs_le010", "gs_le050", "gs_le100", "gs_le150", "gs_le200",
                "gs_le250", "gs_labelenergy", "gs_le2k", "gs_heat", "gs_smoothness")
    requires = ("features", "labels")
    hparams = {"max_n": 5000, "n_local": 7, "dim_cap": 2000, "seed": 0,
               "heat_t": 1.0, "graph": "knn", "knn_k": 15}
    note = (
        "Collapses the self-tuning RBF affinity onto a K x K class graph and reads "
        "exact, label-based scalars: normalised within-class association "
        "(gs_nassoc = 1 - Ncut/K), Newman modularity (gs_modularity), the class "
        "graph's algebraic connectivity (gs_fiedler, lower better), cross-over-"
        "within leakage (gs_leak, lower), the hardest-pair confusability "
        "(gs_cross, lower), the label energy in the K low-frequency graph "
        "modes (gs_labelenergy), and the continuous label Dirichlet energy on "
        "L_sym (gs_smoothness, lower better -- the full-spectrum form of the "
        "normalised cut).  Default recipe blends the association terms (positive) "
        "against connectivity/leakage/cross (negative); fit F per hub."
    )

    def compute(self, probe: ProbeData, **hp) -> dict[str, float]:
        X, y = as_float_features(probe), probe.labels
        return graph_spectral_te_score(
            X, y, max_n=int(hp["max_n"]), n_local=int(hp["n_local"]),
            dim_cap=int(hp["dim_cap"]), seed=int(hp["seed"]),
            heat_t=float(hp["heat_t"]), graph=str(hp["graph"]),
            knn_k=int(hp["knn_k"]))
