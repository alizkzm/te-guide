"""LDA separability score of the target features."""

from __future__ import annotations

import numpy as np
from scipy import linalg as sla

from guide.core.linalg import class_means, iterative_A, shrinkage_cov, softmax
from guide.core.probe import ProbeData
from guide.core.registry import register
from guide.scores.base import TEScore, as_float_features


class LDAClassifier:
    """Single-stage shrinkage linear discriminant fitted on target features."""

    def __init__(self, shrinkage: float | None = None):
        self.shrinkage = shrinkage

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LDAClassifier":
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
    """Mean true-class probability of the LDA classifier on the target samples."""
    prob = LDAClassifier().fit(X, y).predict_proba(X)
    return float(prob[np.arange(len(y)), y].sum() / len(y))


@register
class LDA(TEScore):
    """LDA separability score: mean true-class probability on the target features."""

    name = "lda"
    paper = "Linear discriminant analysis - class separability of the target features"
    elements = ("lda",)
    requires = ("features", "labels")

    def compute(self, probe: ProbeData, **hp) -> dict[str, float]:
        X, y = as_float_features(probe), probe.labels
        return {"lda": lda_score(X, y)}
