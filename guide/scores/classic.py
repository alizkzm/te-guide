"""Classic TE scores: LEEP, N-LEEP, LogME, H-Score (+shrinkage), NCE, PARC, LFC."""

from __future__ import annotations

import numpy as np
import scipy.stats

from guide.core.linalg import class_means, one_hot, pca_reduce, shrinkage_cov
from guide.core.probe import ProbeData
from guide.core.registry import register
from guide.core.sampling import random_indices
from guide.scores.base import SingleElementScore, as_float_features


def leep_from_probs(source_probs: np.ndarray, y: np.ndarray) -> float:
    """LEEP = (1/n) sum_i log p(y_i | x_i) with p(y|x) = sum_z p(y|z) p(z|x)."""
    n = len(y)
    c_t = int(y.max() + 1)
    pyz = np.zeros((c_t, source_probs.shape[1]), dtype=np.float64)
    for c in range(c_t):
        pyz[c] = source_probs[y == c].sum(axis=0)
    pyz /= n
    pz = pyz.sum(axis=0, keepdims=True)
    py_given_z = np.divide(pyz, pz, out=np.zeros_like(pyz), where=pz > 0)
    py_given_x = source_probs @ py_given_z.T
    return float(np.log(py_given_x[np.arange(n), y] + 1e-12).mean())


@register
class LEEP(SingleElementScore):
    name = "leep"
    paper = "LEEP: A New Measure to Evaluate Transferability (ICML 2020)"
    elements = ("leep",)
    requires = ("logits", "labels")

    def value(self, probe: ProbeData, **hp) -> float:
        return leep_from_probs(probe.probs, probe.labels)


@register
class NLEEP(SingleElementScore):
    name = "nleep"
    paper = "Ranking Neural Checkpoints (CVPR 2021)"
    elements = ("nleep",)
    requires = ("features", "labels")
    hparams = {"component_ratio": 5, "pca_energy": 0.8, "max_components": 200, "seed": 0}
    note = "GMM replaces the source head, so it also works for head-less backbones."

    def value(self, probe: ProbeData, **hp) -> float:
        from sklearn.mixture import GaussianMixture

        X, y = as_float_features(probe), probe.labels
        n_classes = probe.n_classes
        X_pca = pca_reduce(X, float(hp["pca_energy"]), seed=hp["seed"])

        k = int(hp["component_ratio"]) * n_classes
        k = max(1, min(k, int(hp["max_components"]), X_pca.shape[0] - 1))

        gmm = GaussianMixture(n_components=k, random_state=hp["seed"]).fit(X_pca)
        return leep_from_probs(gmm.predict_proba(X_pca), y)


def _truncated_svd(x: np.ndarray):
    u, s, vh = np.linalg.svd(x.T @ x)
    s = np.sqrt(s)
    u_times_sigma = x @ vh.T
    k = int(np.sum(s > 1e-10))
    s = s.reshape(-1, 1)[:k]
    vh = vh[:k]
    u = u_times_sigma[:, :k] / s.reshape(1, -1)
    return u, s, vh


def logme_score(f: np.ndarray, y: np.ndarray, regression: bool = False) -> float:
    f = f.astype(np.float64)
    n, d = f.shape
    u, s, vh = _truncated_svd(f) if n > d else np.linalg.svd(f, full_matrices=False)
    s = np.asarray(s).reshape(-1, 1)
    sigma = s ** 2

    num_dim = y.shape[1] if regression else int(y.max() + 1)
    evidences = []
    for i in range(num_dim):
        y_ = (y[:, i] if regression else (y == i).astype(np.float64)).reshape(-1, 1)
        x = u.T @ y_
        x2 = x ** 2
        res_x2 = float((y_ ** 2).sum() - x2.sum())

        alpha, beta, t = 1.0, 1.0, 1.0
        res2 = m2 = 0.0
        for _ in range(11):
            t = alpha / beta
            gamma = float((sigma / (sigma + t)).sum())
            m2 = float((sigma * x2 / ((t + sigma) ** 2)).sum())
            res2 = float((x2 / ((1 + sigma / t) ** 2)).sum() + res_x2)
            alpha = gamma / (m2 + 1e-5)
            beta = (n - gamma) / (res2 + 1e-5)
            if abs(alpha / beta - t) / t <= 1e-3:
                break
        evidence = (
            d / 2.0 * np.log(alpha)
            + n / 2.0 * np.log(beta)
            - 0.5 * np.sum(np.log(alpha + beta * sigma))
            - beta / 2.0 * res2
            - alpha / 2.0 * m2
            - n / 2.0 * np.log(2 * np.pi)
        ) / n
        evidences.append(evidence)
    return float(np.mean(evidences))


@register
class LogME(SingleElementScore):
    name = "logme"
    paper = "LogME: Practical Assessment of Pre-trained Models (ICML 2021)"
    elements = ("logme",)

    def value(self, probe: ProbeData, **hp) -> float:
        return logme_score(as_float_features(probe), probe.labels)


def h_score(X: np.ndarray, y: np.ndarray, shrinkage: float = 0.0) -> float:
    """H = tr( Cov(f)^+ Cov(E[f|y]) )."""
    cov_f = shrinkage_cov(X, shrinkage) if shrinkage > 0 else np.cov(X.T, bias=True)
    g = np.zeros_like(X)
    for c, m in zip(np.unique(y), class_means(X, y)):
        g[y == c] = m
    cov_g = np.cov(g.T, bias=True)
    return float(np.trace(np.linalg.pinv(np.atleast_2d(cov_f), rcond=1e-15) @ np.atleast_2d(cov_g)))


@register
class HScore(SingleElementScore):
    name = "hscore"
    paper = "An Information-Theoretic Approach to Transferability (ICIP 2019)"
    elements = ("hscore",)

    def value(self, probe: ProbeData, **hp) -> float:
        return h_score(as_float_features(probe), probe.labels)


@register
class ShrinkageHScore(SingleElementScore):
    name = "hscore_reg"
    paper = "Newer is Not Always Better - shrinkage H-Score (ECML PKDD 2022)"
    elements = ("hscore_reg",)
    hparams = {"shrinkage": None}
    note = "shrinkage=None -> Ledoit-Wolf estimate of the optimal intensity."

    def value(self, probe: ProbeData, **hp) -> float:
        X, y = as_float_features(probe), probe.labels
        a = hp.get("shrinkage")
        if a is None:
            from sklearn.covariance import ledoit_wolf

            _, a = ledoit_wolf(X - X.mean(axis=0, keepdims=True))
        return h_score(X, y, shrinkage=float(np.clip(a, 1e-6, 1.0)))


@register
class NCE(SingleElementScore):
    name = "nce"
    paper = "Transferability and Hardness of Supervised Classification (ICCV 2019)"
    elements = ("nce",)
    requires = ("logits", "labels")
    note = "z = argmax of the source head on the target images (the usual TE proxy)."

    def value(self, probe: ProbeData, **hp) -> float:
        z = probe.source_pseudo_labels
        y = probe.labels
        n = len(y)
        c_s, c_t = int(z.max() + 1), int(y.max() + 1)
        joint = np.zeros((c_t, c_s), dtype=np.float64)
        np.add.at(joint, (y, z), 1.0 / n)
        p_z = joint.sum(axis=0, keepdims=True)
        ratio = np.divide(joint, p_z, out=np.zeros_like(joint), where=p_z > 0)
        with np.errstate(divide="ignore"):
            log_ratio = np.where(ratio > 0, np.log(ratio + 1e-20), 0.0)
        return float((joint * log_ratio).sum())


@register
class PARC(SingleElementScore):
    name = "parc"
    paper = "Scalable Diverse Model Selection - PARC (NeurIPS 2021)"
    elements = ("parc",)
    hparams = {"n_dims": 32, "max_samples": 5000, "seed": 0}
    note = "Spearman between the feature RDM and the one-hot-label RDM (x100)."

    def value(self, probe: ProbeData, **hp) -> float:
        from sklearn.preprocessing import StandardScaler

        X, y = as_float_features(probe), probe.labels
        if X.shape[0] > int(hp["max_samples"]):
            idx = random_indices(X.shape[0], int(hp["max_samples"]), seed=hp["seed"])
            X, y = X[idx], y[idx]

        X = pca_reduce(X, int(hp["n_dims"]), seed=hp["seed"])
        labels = one_hot(y, int(y.max() + 1))

        X = StandardScaler().fit_transform(X)
        rdm_f = 1 - np.corrcoef(X)
        rdm_y = 1 - np.corrcoef(labels)
        iu = np.triu_indices(rdm_f.shape[0], 1)
        return float(scipy.stats.spearmanr(rdm_f[iu], rdm_y[iu])[0] * 100)


@register
class LFC(SingleElementScore):
    name = "lfc"
    paper = "Label-Feature Correlation (baseline in Model-Spider, NeurIPS 2023)"
    elements = ("lfc",)
    hparams = {"max_samples": 8000, "seed": 0}

    def value(self, probe: ProbeData, **hp) -> float:
        X, y = as_float_features(probe), probe.labels
        if X.shape[0] > int(hp["max_samples"]):
            idx = random_indices(X.shape[0], int(hp["max_samples"]), seed=hp["seed"])
            X, y = X[idx], y[idx]

        Y = one_hot(y, int(y.max() + 1))
        theta_y = Y @ Y.T
        theta_y = np.where(theta_y == 0, -1.0, 1.0)
        theta_x = X @ X.T

        theta_x = theta_x - theta_x.mean()
        theta_y = theta_y - theta_y.mean()
        denom = np.sqrt((theta_x ** 2).sum()) * np.sqrt((theta_y ** 2).sum())
        return float((theta_x * theta_y).sum() / (denom + 1e-12))
