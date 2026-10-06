"""Class-separability TE scores: TransRate, GBC, SFDA, NCTI, TMI."""

from __future__ import annotations

import numpy as np
from scipy import linalg as sla

from guide.core.linalg import (
    class_means,
    coding_rate,
    iterative_A,
    outer_class_means,
    pca_reduce,
    shrinkage_cov,
    softmax,
)
from guide.core.probe import ProbeData
from guide.core.registry import register
from guide.core.sampling import drop_singleton_classes
from guide.scores.base import ScoreNotApplicable, SingleElementScore, TEScore, as_float_features


@register
class TransRate(SingleElementScore):
    name = "transrate"
    paper = "Frustratingly Easy Transferability Estimation (ICML 2022)"
    elements = ("transrate",)
    hparams = {"eps": 1e-4}
    note = "TrR = R(Z) - R(Z|Y), the coding-rate estimate of I(Z; Y)."

    def value(self, probe: ProbeData, **hp) -> float:
        X, y = as_float_features(probe), probe.labels
        X = X - X.mean(axis=0, keepdims=True)
        eps = float(hp["eps"])
        rz = coding_rate(X, eps)
        classes = np.unique(y)
        rzy = float(np.mean([coding_rate(X[y == c], eps) for c in classes]))
        return rz - rzy


@register
class GBC(SingleElementScore):
    name = "gbc"
    paper = "Transferability Estimation using Bhattacharyya Class Separability (CVPR 2022)"
    elements = ("gbc",)
    hparams = {"pca_dim": 64, "gaussian_type": "spherical", "seed": 0}
    note = (
        "Negative sum of pairwise Bhattacharyya coefficients. "
        "gaussian_type='spherical' (the paper's default) uses diagonal variances "
        "and is far more stable than the full-covariance variant on small classes."
    )

    def value(self, probe: ProbeData, **hp) -> float:
        X, y = as_float_features(probe), probe.labels
        keep = drop_singleton_classes(y, min_count=2)
        X, y = X[keep], y[keep]
        if np.unique(y).size < 2:
            raise ScoreNotApplicable("GBC needs >= 2 classes with >= 2 samples")

        X = pca_reduce(X, int(hp["pca_dim"]), seed=hp["seed"])
        spherical = str(hp["gaussian_type"]).lower() == "spherical"

        mus, sigmas = [], []
        for c in np.unique(y):
            Xc = X[y == c]
            mus.append(Xc.mean(axis=0))
            if spherical:
                sigmas.append(Xc.var(axis=0) + 1e-8)
            else:
                sigmas.append(np.cov(Xc.T, bias=True) + 1e-6 * np.eye(X.shape[1]))

        score = 0.0
        n = len(mus)
        for i in range(n):
            for j in range(i + 1, n):
                dmu = mus[i] - mus[j]
                if spherical:
                    sig = 0.5 * (sigmas[i] + sigmas[j])
                    d_b = 0.125 * float((dmu ** 2 / sig).sum()) + 0.5 * float(
                        np.log(sig).sum()
                        - 0.5 * np.log(sigmas[i]).sum()
                        - 0.5 * np.log(sigmas[j]).sum()
                    )
                else:
                    sig = 0.5 * (sigmas[i] + sigmas[j])
                    sign_s, logdet_s = np.linalg.slogdet(sig)
                    _, logdet_i = np.linalg.slogdet(sigmas[i])
                    _, logdet_j = np.linalg.slogdet(sigmas[j])
                    d_b = 0.125 * float(dmu @ np.linalg.pinv(sig) @ dmu) + 0.5 * (
                        logdet_s - 0.5 * (logdet_i + logdet_j)
                    )
                score -= float(np.exp(-d_b))
        return 2.0 * score


def _qda_fold(Xtr, ytr, Xte, reg, pool):
    """Fit a regularised-QDA on (Xtr, ytr), return test-set logits."""
    cls, ytr_idx = np.unique(ytr, return_inverse=True)
    d = Xtr.shape[1]
    priors = np.bincount(ytr_idx) / float(len(ytr))
    mus, covs = [], []
    pooled = np.zeros((d, d))
    for i in range(len(cls)):
        Xc = Xtr[ytr_idx == i]
        mus.append(Xc.mean(axis=0))
        Sc = np.atleast_2d(np.cov(Xc.T, bias=True)) if len(Xc) > 1 else np.zeros((d, d))
        covs.append(Sc)
        pooled += priors[i] * Sc
    logits = np.empty((len(Xte), len(cls)))
    for i in range(len(cls)):
        sig = (1.0 - pool) * covs[i] + pool * pooled
        mu_d = np.trace(sig) / d
        sig = (1.0 - reg) * sig + reg * mu_d * np.eye(d)
        sig.flat[:: d + 1] += 1e-6
        _, logdet = np.linalg.slogdet(sig)
        prec = np.linalg.inv(sig)
        diff = Xte - mus[i]
        maha = np.einsum("ni,ij,nj->n", diff, prec, diff)
        logits[:, i] = -0.5 * maha - 0.5 * logdet + np.log(priors[i])
    return cls, logits


def qda_score(X: np.ndarray, y: np.ndarray, reg: float = 0.2, pool: float = 0.5,
              n_splits: int = 5, seed: int = 0) -> float:
    """Cross-validated mean true-class probability of a regularised QDA."""
    from sklearn.model_selection import StratifiedKFold

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    counts = np.unique(y, return_counts=True)[1]
    k = int(min(n_splits, counts.min()))
    if k < 2:
        raise ScoreNotApplicable("QDA needs >= 2 samples in the smallest class")
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    probs = []
    for tr, te in skf.split(X, y):
        cls, logits = _qda_fold(X[tr], y[tr], X[te], reg, pool)
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        col = np.searchsorted(cls, y[te])
        probs.append(p[np.arange(len(te)), col])
    return float(np.concatenate(probs).mean())


@register
class QDA(SingleElementScore):
    """Quadratic Discriminant Analysis separability of the target features."""

    name = "qda"
    paper = "Regularized Quadratic Discriminant Analysis (class separability of target features)"
    elements = ("qda",)
    hparams = {"pca_dim": 64, "reg": 0.2, "pool": 0.5, "n_splits": 5, "seed": 0}
    note = (
        "Per-class covariance shrunk toward the pooled within-class covariance "
        "(pool, Friedman's RDA) and a scaled identity (reg), after a PCA to "
        "pca_dim; scored out of sample by stratified k-fold mean true-class "
        "probability.  The pooling keeps it stable on fine-grained targets with "
        "many small classes, where a pure QDA covariance is unestimable."
    )

    def value(self, probe: ProbeData, **hp) -> float:
        X, y = as_float_features(probe), probe.labels
        keep = drop_singleton_classes(y, min_count=2)
        X, y = X[keep], y[keep]
        if np.unique(y).size < 2:
            raise ScoreNotApplicable("QDA needs >= 2 classes with >= 2 samples")
        X = pca_reduce(X, int(hp["pca_dim"]), seed=hp["seed"])
        return qda_score(X, y, reg=float(hp["reg"]), pool=float(hp["pool"]),
                         n_splits=int(hp["n_splits"]), seed=int(hp["seed"]))


class RegularizedFDA:
    """Regularised Fisher discriminant with the adaptive shrinkage of SFDA."""

    def __init__(self, shrinkage: float | None = None):
        self.shrinkage = shrinkage

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RegularizedFDA":
        classes, y_idx = np.unique(y, return_inverse=True)
        self.priors_ = np.bincount(y_idx) / float(len(y))
        self.means_ = class_means(X, y)

        d = X.shape[1]
        sw = np.zeros((d, d))
        for i, c in enumerate(classes):
            sw += self.priors_[i] * np.atleast_2d(np.cov(X[y == c].T, bias=True))

        if self.shrinkage is None:
            largest_eval = iterative_A(sw, max_iterations=3)
            self.shrinkage = float(max(np.exp(-5 * largest_eval), 1e-10))
        a = self.shrinkage

        st = shrinkage_cov(X, a)
        mu = np.trace(sw) / d
        sw_reg = (1.0 - a) * sw
        sw_reg.flat[:: d + 1] += a * mu
        sb = st - sw_reg

        try:
            evals, evecs = sla.eigh(sb, sw_reg)
        except (np.linalg.LinAlgError, ValueError):
            evals, evecs = np.linalg.eigh(np.linalg.pinv(sw_reg) @ sb)
        evecs = evecs[:, np.argsort(evals)[::-1]]

        self.scalings_ = evecs
        self.coef_ = self.means_ @ evecs @ evecs.T
        self.intercept_ = -0.5 * np.diag(self.means_ @ self.coef_.T) + np.log(self.priors_)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_.T + self.intercept_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return softmax(self.decision_function(X))


@register
class SFDA(SingleElementScore):
    name = "sfda"
    paper = "Not All Models Are Equal - Self-challenging Fisher Space (ECCV 2022)"
    elements = ("sfda",)
    note = "Two-stage FDA with ConfMix self-challenging between the stages."

    def value(self, probe: ProbeData, **hp) -> float:
        X, y = as_float_features(probe).copy(), probe.labels
        n = len(y)

        stage1 = RegularizedFDA().fit(X, y)
        prob = stage1.predict_proba(X)
        prob = np.exp(prob) / np.exp(prob).sum(axis=1, keepdims=True)

        others = outer_class_means(X, y)
        conf = prob[np.arange(n), y].reshape(-1, 1)
        for i, c in enumerate(np.unique(y)):
            mask = y == c
            X[mask] = conf[mask] * X[mask] + (1.0 - conf[mask]) * others[i]

        stage2 = RegularizedFDA(shrinkage=stage1.shrinkage).fit(X, y)
        prob2 = stage2.predict_proba(X)
        return float(prob2[np.arange(n), y].mean())


@register
class NCTI(TEScore):
    name = "ncti"
    paper = "How Far Pre-Trained Models Are from Neural Collapse (ICCV 2023)"
    elements = ("feature_nuc", "cls_conf", "log_class_pred_nuc")
    hparams = {"pca_dim": 64, "top_k_var": 32, "seed": 0}
    note = (
        "Three raw terms. Reference combination (min-max per hub): "
        "cls_conf + feature_nuc - log_class_pred_nuc."
    )

    def compute(self, probe: ProbeData, **hp) -> dict[str, float]:
        from sklearn.decomposition import PCA
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        X, y = as_float_features(probe), probe.labels
        keep = drop_singleton_classes(y, min_count=2)
        X, y = X[keep], y[keep]

        k = int(min(hp["pca_dim"], X.shape[0] - 1, X.shape[1]))
        pca = PCA(n_components=k, random_state=hp["seed"])
        X = pca.fit_transform(X)
        temp = max(float(np.exp(-pca.explained_variance_[: int(hp["top_k_var"])].sum())), 1e-10)

        if temp <= 1e-10:
            clf = LinearDiscriminantAnalysis(solver="svd")
        else:
            clf = LinearDiscriminantAnalysis(solver="eigen", shrinkage=float(min(temp, 1.0)))

        low_feat = clf.fit_transform(X, y)
        low_feat = low_feat - low_feat.mean(axis=0, keepdims=True)
        feature_nuc = float(np.linalg.norm(low_feat, ord="nuc"))

        low_pred = clf.predict_proba(X)
        cls_conf = float(low_pred[np.arange(X.shape[0]), y].sum() / X.shape[0])

        class_pred_nuc = 0.0
        for c in np.unique(y):
            class_pred_nuc += float(np.linalg.norm(low_pred[y == c], ord="nuc"))

        return {
            "feature_nuc": feature_nuc,
            "cls_conf": cls_conf,
            "log_class_pred_nuc": float(np.log(class_pred_nuc + 1e-12)),
        }


@register
class TMI(SingleElementScore):
    name = "tmi"
    paper = "Fast and Accurate Transferability Measurement via Intra-Class Variance (ICCV 2023)"
    elements = ("tmi",)
    note = "Sample-weighted mean intra-class feature variance (higher = better)."

    def value(self, probe: ProbeData, **hp) -> float:
        X, y = as_float_features(probe), probe.labels
        total = 0.0
        for c in np.unique(y):
            Xc = X[y == c]
            if Xc.shape[0] < 2:
                continue
            total += float(Xc.var(axis=0, ddof=0).sum()) * Xc.shape[0]
        return total / len(y)
