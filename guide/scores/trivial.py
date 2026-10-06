"""Trivial / naive baselines."""

from __future__ import annotations

import numpy as np

from guide.core.linalg import class_means
from guide.core.probe import ProbeData
from guide.core.registry import register
from guide.core.sampling import drop_singleton_classes, random_indices
from guide.scores.base import ScoreNotApplicable, SingleElementScore, as_float_features


class _MetaScore(SingleElementScore):
    requires = ("meta",)
    trivial = True
    label_free = True
    meta_key = ""

    def value(self, probe: ProbeData, **hp) -> float:
        v = probe.meta.get(self.meta_key)
        if v is None:
            raise ScoreNotApplicable(
                f"probe metadata has no '{self.meta_key}' "
                f"(re-run extraction, or it is unavailable for this model)"
            )
        return float(v)


@register
class NumParams(_MetaScore):
    name = "n_params"
    paper = "number of parameters (dataset-agnostic heuristic)"
    elements = ("n_params",)
    meta_key = "n_params"


@register
class Flops(_MetaScore):
    name = "flops"
    paper = "forward FLOPs at 224x224 (dataset-agnostic heuristic)"
    elements = ("flops",)
    meta_key = "flops"
    status = "partial"
    note = (
        "INCOMPLETE unless FLOPs were counted at extraction time: needs "
        "`pip install fvcore` (or thop) and `python run.py extract --flops`."
    )


@register
class SourceTop1(_MetaScore):
    name = "source_top1"
    paper = "ImageNet top-1 of the backbone (dataset-agnostic heuristic)"
    elements = ("source_top1",)
    meta_key = "source_top1"
    note = "Undefined for self-supervised backbones with no classification head."


@register
class FeatureDim(_MetaScore):
    name = "feature_dim"
    paper = "penultimate feature dimensionality (dataset-agnostic heuristic)"
    elements = ("feature_dim",)
    meta_key = "feature_dim"


@register
class RandomScore(SingleElementScore):
    name = "random"
    paper = "seeded random score (null baseline)"
    elements = ("random",)
    requires = ()
    trivial = True
    label_free = True
    hparams = {"seed": 0}

    def value(self, probe: ProbeData, **hp) -> float:
        import hashlib

        key = f"{probe.model_name}|{probe.dataset}|{hp['seed']}".encode()
        digest = hashlib.blake2b(key, digest_size=4).digest()
        return float(np.random.RandomState(int.from_bytes(digest, "big")).rand())


def _cv_accuracy(estimator, X, y, n_splits=3, seed=0, max_samples=20000):
    from sklearn.model_selection import StratifiedKFold

    if X.shape[0] > max_samples:
        idx = random_indices(X.shape[0], max_samples, seed=seed)
        X, y = X[idx], y[idx]
    keep = drop_singleton_classes(y, min_count=n_splits)
    if keep.size < n_splits or np.unique(y[keep]).size < 2:
        raise ScoreNotApplicable("not enough samples per class for cross-validation")
    X, y = X[keep], y[keep]

    accs = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        accs.append(estimator().fit(X[tr], y[tr]).score(X[te], y[te]))
    return float(np.mean(accs))


@register
class KNN(SingleElementScore):
    name = "knn"
    paper = "k-NN as a Simple and Effective Estimator of Transferability (arXiv 2025)"
    elements = ("knn",)
    trivial = False
    hparams = {"k": 20, "n_splits": 3, "seed": 0, "max_samples": 20000}

    def value(self, probe: ProbeData, **hp) -> float:
        from sklearn.neighbors import KNeighborsClassifier

        k = int(hp["k"])
        return _cv_accuracy(
            lambda: KNeighborsClassifier(n_neighbors=k),
            as_float_features(probe),
            probe.labels,
            n_splits=int(hp["n_splits"]),
            seed=int(hp["seed"]),
            max_samples=int(hp["max_samples"]),
        )


@register
class LinearProbe(SingleElementScore):
    name = "linear_probe"
    paper = "logistic-regression linear probe accuracy (strong naive baseline)"
    elements = ("linear_probe",)
    trivial = True
    hparams = {"C": 1.0, "max_iter": 200, "n_splits": 3, "seed": 0, "max_samples": 10000}

    def value(self, probe: ProbeData, **hp) -> float:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        def est():
            return make_pipeline(
                StandardScaler(),
                LogisticRegression(C=float(hp["C"]), max_iter=int(hp["max_iter"])),
            )

        return _cv_accuracy(
            est,
            as_float_features(probe),
            probe.labels,
            n_splits=int(hp["n_splits"]),
            seed=int(hp["seed"]),
            max_samples=int(hp["max_samples"]),
        )


@register
class LDAAccuracy(SingleElementScore):
    name = "lda_acc"
    paper = "cross-validated LDA accuracy on the frozen features"
    elements = ("lda_acc",)
    trivial = True
    hparams = {"n_splits": 3, "seed": 0, "pca_dim": 64, "max_samples": 20000}

    def value(self, probe: ProbeData, **hp) -> float:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        from guide.core.linalg import pca_reduce

        X = pca_reduce(as_float_features(probe), int(hp["pca_dim"]), seed=int(hp["seed"]))
        return _cv_accuracy(
            lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
            X,
            probe.labels,
            n_splits=int(hp["n_splits"]),
            seed=int(hp["seed"]),
            max_samples=int(hp["max_samples"]),
        )


@register
class GNBAccuracy(SingleElementScore):
    name = "gnb_acc"
    paper = "cross-validated Gaussian naive-Bayes accuracy"
    elements = ("gnb_acc",)
    trivial = True
    hparams = {"n_splits": 3, "seed": 0, "max_samples": 20000}

    def value(self, probe: ProbeData, **hp) -> float:
        from sklearn.naive_bayes import GaussianNB

        return _cv_accuracy(
            GaussianNB,
            as_float_features(probe),
            probe.labels,
            n_splits=int(hp["n_splits"]),
            seed=int(hp["seed"]),
            max_samples=int(hp["max_samples"]),
        )


@register
class Silhouette(SingleElementScore):
    name = "silhouette"
    paper = "silhouette coefficient of the ground-truth clustering"
    elements = ("silhouette",)
    trivial = True
    hparams = {"max_samples": 5000, "seed": 0}

    def value(self, probe: ProbeData, **hp) -> float:
        from sklearn.metrics import silhouette_score

        X, y = as_float_features(probe), probe.labels
        if X.shape[0] > int(hp["max_samples"]):
            idx = random_indices(X.shape[0], int(hp["max_samples"]), seed=int(hp["seed"]))
            X, y = X[idx], y[idx]
        if np.unique(y).size < 2:
            raise ScoreNotApplicable("silhouette needs >= 2 classes")
        return float(silhouette_score(X, y))


@register
class FisherRatio(SingleElementScore):
    name = "fisher_ratio"
    paper = "tr(S_b) / tr(S_w) - the textbook Fisher separability ratio"
    elements = ("fisher_ratio",)
    trivial = True

    def value(self, probe: ProbeData, **hp) -> float:
        X, y = as_float_features(probe), probe.labels
        mu = X.mean(axis=0)
        means = class_means(X, y)
        counts = np.array([(y == c).sum() for c in np.unique(y)], dtype=np.float64)

        sb = float((counts[:, None] * (means - mu) ** 2).sum())
        sw = 0.0
        for i, c in enumerate(np.unique(y)):
            sw += float(((X[y == c] - means[i]) ** 2).sum())
        return sb / (sw + 1e-12)


@register
class FeatureVariance(SingleElementScore):
    name = "feature_var"
    paper = "mean per-dimension feature variance (label-free null baseline)"
    elements = ("feature_var",)
    requires = ("features",)
    trivial = True
    label_free = True

    def value(self, probe: ProbeData, **hp) -> float:
        return float(as_float_features(probe).var(axis=0).mean())


@register
class KMeansNMI(SingleElementScore):
    name = "kmeans_nmi"
    paper = "NMI between k-means clusters (k = #classes) and the true labels"
    elements = ("kmeans_nmi",)
    trivial = True
    hparams = {"max_samples": 10000, "seed": 0}

    def value(self, probe: ProbeData, **hp) -> float:
        from sklearn.cluster import KMeans
        from sklearn.metrics import normalized_mutual_info_score

        X, y = as_float_features(probe), probe.labels
        if X.shape[0] > int(hp["max_samples"]):
            idx = random_indices(X.shape[0], int(hp["max_samples"]), seed=int(hp["seed"]))
            X, y = X[idx], y[idx]
        k = int(np.unique(y).size)
        if k < 2:
            raise ScoreNotApplicable("k-means NMI needs >= 2 classes")
        km = KMeans(n_clusters=k, n_init=4, random_state=int(hp["seed"])).fit(X)
        return float(normalized_mutual_info_score(y, km.labels_))
