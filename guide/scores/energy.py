"""ETran - Gholami et al., ICCV 2023 (energy + LDA + regression)."""

from __future__ import annotations

import numpy as np
from scipy import linalg as sla
from scipy.special import logsumexp

from guide.core.linalg import class_means, iterative_A, shrinkage_cov, softmax
from guide.core.probe import ProbeData
from guide.core.registry import register
from guide.scores.base import TEScore, as_float_features


class ETranLDA:
    def __init__(self, shrinkage: float | None = None):
        self.shrinkage = shrinkage

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ETranLDA":
        classes, y_idx = np.unique(y, return_inverse=True)
        self.priors_ = np.bincount(y_idx) / float(len(y))
        self.means_ = class_means(X, y)

        d = X.shape[1]
        sw = np.zeros((d, d))
        for i, c in enumerate(classes):
            sw += self.priors_[i] * np.atleast_2d(np.cov(X[y == c].T, bias=True))

        if self.shrinkage is None:
            self.shrinkage = float(max(np.exp(-5 * iterative_A(sw, 3)), 1e-10))
        a = self.shrinkage

        st = shrinkage_cov(X, a)
        mu = np.trace(sw) / d
        sw_reg = (1.0 - a) * sw
        sw_reg.flat[:: d + 1] += a * mu
        sb = st - sw_reg

        try:
            evals, evecs = np.linalg.eigh(np.linalg.inv(sw_reg) @ sb)
        except np.linalg.LinAlgError:
            evals, evecs = sla.eigh(sb, sw_reg)
        evecs = evecs[:, np.argsort(evals)[::-1]]

        self.scalings_ = evecs
        self.coef_ = self.means_ @ evecs @ evecs.T
        self.intercept_ = -0.5 * np.diag(self.means_ @ self.coef_.T) + np.log(self.priors_)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return softmax(X @ self.coef_.T + self.intercept_)


def lda_score(X: np.ndarray, y: np.ndarray) -> float:
    prob = ETranLDA().fit(X, y).predict_proba(X)
    return float(prob[np.arange(len(y)), y].sum() / len(y))


def energy_score(logits: np.ndarray, percent: float = 100.0, tail: str = "top") -> float:
    """Mean free energy logsumexp(logits) over the selected tail of the batch."""
    e = logsumexp(np.asarray(logits, dtype=np.float64), axis=-1)
    n = e.shape[0]
    k = int(percent * 10) * n // 1000
    k = int(np.clip(k, 1, n))
    order = np.argsort(e)
    chosen = order[:k] if tail == "bot" else order[-k:]
    return float(e[chosen].mean())


def regression_score(X: np.ndarray) -> float:
    """ETran's regression term: total spectral energy of the feature matrix."""
    s = np.linalg.svd(np.asarray(X, dtype=np.float64), compute_uv=False)
    return float((s ** 2).sum())


@register
class ETran(TEScore):
    name = "etran"
    paper = "ETran: Energy-Based Transferability Estimation (ICCV 2023)"
    elements = ("energy", "lda", "regression")
    requires = ("features", "labels")
    hparams = {"percent": 100.0, "tail": "top", "with_regression": False}
    note = (
        "`energy` needs the source head; head-less SSL backbones produce lda "
        "(and optionally regression) only."
    )

    def compute(self, probe: ProbeData, **hp) -> dict[str, float]:
        X, y = as_float_features(probe), probe.labels
        out = {"lda": lda_score(X, y)}
        if probe.logits is not None:
            out["energy"] = energy_score(probe.logits, float(hp["percent"]), str(hp["tail"]))
        if hp.get("with_regression"):
            out["regression"] = regression_score(X)
        return out


@register
class EnergyOnly(TEScore):
    """The energy term on its own - a useful, genuinely label-free baseline."""

    name = "energy"
    paper = "ETran energy term (ICCV 2023)"
    elements = ("energy",)
    requires = ("logits",)
    label_free = True
    hparams = {"percent": 100.0, "tail": "top"}

    def compute(self, probe: ProbeData, **hp) -> dict[str, float]:
        return {"energy": energy_score(probe.logits, float(hp["percent"]), str(hp["tail"]))}


def lda_refined_score(
    X: np.ndarray,
    y: np.ndarray,
    standardise: bool = False,
    leave_one_out: bool = True,
    pca_dim: int | None = None,
) -> float:
    """`lda` with its self-inclusion artefact removed."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    n_classes = int(y.max()) + 1

    if standardise:
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)

    if pca_dim is not None and 0 < int(pca_dim) < min(X.shape):
        Xc = X - X.mean(axis=0)
        _, _, vt = np.linalg.svd(Xc, full_matrices=False)
        X = Xc @ vt[: int(pca_dim)].T

    mu = class_means(X, y)
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    log_prior = np.log(counts / len(y))

    eta = X @ mu.T - 0.5 * np.einsum("cd,cd->c", mu, mu) + log_prior

    if leave_one_out:
        keep = counts[y] > 1
        n_i = counts[y][:, None]
        mu_i = (n_i * mu[y] - X) / np.maximum(n_i - 1.0, 1.0)
        own = (
            np.einsum("nd,nd->n", X, mu_i)
            - 0.5 * np.einsum("nd,nd->n", mu_i, mu_i)
            + log_prior[y]
        )
        rows = np.flatnonzero(keep)
        eta[rows, y[rows]] = own[rows]

    prob = softmax(eta)
    return float(prob[np.arange(len(y)), y].mean())


@register
class LDARefined(TEScore):
    """`lda` with its scale and dimension biases removed (this paper)."""

    name = "lda_refined"
    paper = "de-biased ETran LDA term (this paper)"
    elements = ("lda_refined",)
    requires = ("features", "labels")
    hparams = {"standardise": False, "leave_one_out": True, "pca_dim": None}
    note = (
        "lda with leave-one-out class means, which removes the self-inclusion "
        "artefact at no cost in signal.  standardise defaults to OFF: it does "
        "remove the scale bias, but on real hubs it also flattens the feature "
        "spectrum and discards capacity information that is genuinely "
        "predictive (0.734 -> 0.627 on the SSL hub, 0.512 -> 0.470 on "
        "supervised CNNs).  pca_dim is an ablation lever, also off."
    )

    def compute(self, probe: ProbeData, **hp) -> dict[str, float]:
        X, y = as_float_features(probe), probe.labels
        return {
            "lda_refined": lda_refined_score(
                X,
                y,
                standardise=bool(hp["standardise"]),
                leave_one_out=bool(hp["leave_one_out"]),
                pca_dim=hp["pca_dim"],
            )
        }
