"""De-biased variants of H-Score, LogME and SFDA (this paper)."""

from __future__ import annotations

import numpy as np

from guide.core.linalg import class_means, outer_class_means, softmax
from guide.core.probe import ProbeData
from guide.core.registry import register
from guide.scores.base import TEScore, as_float_features
from guide.scores.classic import _truncated_svd
from guide.scores.separability import RegularizedFDA


def hscore_refined_score(
    X: np.ndarray,
    y: np.ndarray,
    debias_between: bool = True,
    rcond: float = 1e-15,
    n_components: int | None = None,
) -> float:
    """H-Score with the finite-sample inflation of its numerator removed."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    classes = np.unique(y)
    n_classes = len(classes)

    cov_f = np.atleast_2d(np.cov(X.T, bias=True))
    if n_components is not None and 0 < int(n_components) < cov_f.shape[0]:
        w, V = np.linalg.eigh(cov_f)
        keep = np.argsort(w)[::-1][: int(n_components)]
        f_inv = V[:, keep] @ np.diag(1.0 / np.maximum(w[keep], 1e-12)) @ V[:, keep].T
    else:
        f_inv = np.linalg.pinv(cov_f, rcond=float(rcond))

    mu = class_means(X, y)
    g = np.zeros_like(X)
    for c, m in zip(classes, mu):
        g[y == c] = m
    cov_g = np.atleast_2d(np.cov(g.T, bias=True))
    h = float(np.trace(f_inv @ cov_g))

    if debias_between and n_classes > 1:
        sw = np.zeros_like(cov_f)
        k_eff = 0.0
        for c in classes:
            Z = X[y == c]
            if len(Z) < 2:
                continue
            Zc = Z - Z.mean(axis=0)
            sw += (Zc.T @ Zc) / (len(Z) - 1)
            k_eff += len(Z)
        sw /= max(n_classes, 1)
        k_eff = max(k_eff / max(n_classes, 1), 1.0)
        h -= (1.0 - 1.0 / n_classes) / k_eff * float(np.trace(f_inv @ sw))
    return h


@register
class HScoreRefined(TEScore):
    """H-Score with the finite-sample inflation of `Cov_g` removed."""

    name = "hscore_refined"
    paper = "de-biased H-Score (this paper)"
    elements = ("hscore_refined",)
    requires = ("features", "labels")
    hparams = {"debias_between": True, "rcond": 1e-15, "n_components": None}
    note = (
        "H-Score minus the (1-1/C) Sigma_w / k inflation of its numerator - a "
        "finite-sample artefact of estimating the class means, removable at no "
        "cost in signal.  rcond/n_components truncate the pseudo-inverse and "
        "default to OFF: they fix a real dimension bias on synthetic data but "
        "discard capacity signal on real hubs."
    )

    def compute(self, probe: ProbeData, **hp) -> dict[str, float]:
        X, y = as_float_features(probe), probe.labels
        return {
            "hscore_refined": hscore_refined_score(
                X, y,
                debias_between=bool(hp["debias_between"]),
                rcond=float(hp["rcond"]),
                n_components=hp["n_components"],
            )
        }


def logme_refined_score(
    X: np.ndarray,
    y: np.ndarray,
    effective_dim: bool = True,
) -> float:
    """LogME with its complexity term charged for the parameters it USES."""
    f = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    n, d = f.shape
    u, s, _ = _truncated_svd(f) if n > d else np.linalg.svd(f, full_matrices=False)
    s = np.asarray(s).reshape(-1, 1)
    sigma = s ** 2

    evidences = []
    for i in range(int(y.max()) + 1):
        y_ = (y == i).astype(np.float64).reshape(-1, 1)
        x = u.T @ y_
        x2 = x ** 2
        res_x2 = float((y_ ** 2).sum() - x2.sum())

        alpha, beta, t = 1.0, 1.0, 1.0
        res2 = m2 = 0.0
        gamma = float(d)
        for _ in range(11):
            t = alpha / beta
            gamma = float((sigma / (sigma + t)).sum())
            m2 = float((sigma * x2 / ((t + sigma) ** 2)).sum())
            res2 = float((x2 / ((1 + sigma / t) ** 2)).sum() + res_x2)
            alpha = gamma / (m2 + 1e-5)
            beta = (n - gamma) / (res2 + 1e-5)
            if abs(alpha / beta - t) / t <= 1e-3:
                break

        charge = gamma if effective_dim else float(d)
        evidences.append(
            (
                charge / 2.0 * np.log(alpha)
                + n / 2.0 * np.log(beta)
                - 0.5 * np.sum(np.log(alpha + beta * sigma))
                - beta / 2.0 * res2
                - alpha / 2.0 * m2
                - n / 2.0 * np.log(2 * np.pi)
            )
            / n
        )
    return float(np.mean(evidences))


@register
class LogMERefined(TEScore):
    """LogME charged for the parameters the data determines, not for `D`."""

    name = "logme_refined"
    paper = "LogME with an effective-dimension complexity term (this paper)"
    elements = ("logme_refined",)
    requires = ("features", "labels")
    hparams = {"effective_dim": True}
    note = (
        "LogME with gamma (the effective number of parameters, already computed "
        "inside its fixed point) replacing D in the (D/2) log alpha prior term. "
        "LogME is the one benchmark score that penalises feature dimension "
        "(pearson(D, logme) = -0.931 on noise-only dimensions), and width "
        "tracks capacity on real hubs, so that penalty costs signal.  This is a "
        "hypothesis about the accounting, not a derived artefact removal."
    )

    def compute(self, probe: ProbeData, **hp) -> dict[str, float]:
        X, y = as_float_features(probe), probe.labels
        return {
            "logme_refined": logme_refined_score(
                X, y, effective_dim=bool(hp["effective_dim"])
            )
        }


def _fda_lno_proba(fda: RegularizedFDA, X: np.ndarray, y: np.ndarray,
                   leave_frac: float = 0.0) -> np.ndarray:
    """Class probabilities of a fitted `RegularizedFDA`, with each sample's OWN class logit recomputed against a class mean that leaves out its fold."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    classes = np.unique(y)
    idx = {c: i for i, c in enumerate(classes)}
    yi = np.array([idx[v] for v in y])
    counts = np.bincount(yi, minlength=len(classes)).astype(np.float64)
    log_prior = np.log(counts / counts.sum())

    metric = fda.scalings_ @ fda.scalings_.T
    dec = X @ fda.coef_.T + fda.intercept_
    if leave_frac < 0:
        return softmax(dec)

    n = len(y)
    nout = np.maximum(1, np.round(float(leave_frac) * counts).astype(int))
    order = np.argsort(yi, kind="stable")
    class_start = np.searchsorted(yi[order], np.arange(len(classes)))
    within = np.empty(n, dtype=np.int64)
    within[order] = np.arange(n) - class_start[yi[order]]
    gkey = yi.astype(np.int64) * (int(within.max()) + 1) + within // nout[yi]
    _, inv, cnt = np.unique(gkey, return_inverse=True, return_counts=True)

    mx = X @ metric
    foldsum_x = np.zeros((len(cnt), X.shape[1]))
    foldsum_mx = np.zeros((len(cnt), X.shape[1]))
    np.add.at(foldsum_x, inv, X)
    np.add.at(foldsum_mx, inv, mx)

    nc = counts[yi]
    sz = cnt[inv].astype(np.float64)
    valid = nc > nout[yi]
    denom = np.where(valid, nc - sz, 1.0)[:, None]
    mu_lno = (nc[:, None] * fda.means_[yi] - foldsum_x[inv]) / denom
    m_mu = (nc[:, None] * fda.coef_[yi] - foldsum_mx[inv]) / denom
    own = (
        np.einsum("nd,nd->n", mx, mu_lno)
        - 0.5 * np.einsum("nd,nd->n", mu_lno, m_mu)
        + log_prior[yi]
    )
    rows = np.flatnonzero(valid)
    dec[rows, yi[rows]] = own[rows]
    return softmax(dec)


def _within_class_lw_shrinkage(X: np.ndarray, y: np.ndarray) -> float:
    """Ledoit-Wolf optimal shrinkage intensity for the pooled within-class covariance toward its isotropic target ``(tr Sw / D) I``."""
    from sklearn.covariance import ledoit_wolf_shrinkage
    res = np.concatenate([X[y == c] - X[y == c].mean(axis=0) for c in np.unique(y)])
    return float(np.clip(ledoit_wolf_shrinkage(res, assume_centered=True), 1e-10, 1.0))


def sfda_refined_score(
    X: np.ndarray,
    y: np.ndarray,
    leave_frac: float = 0.0,
    standardise: bool = False,
    pca_dim: int | None = None,
    shrinkage: str = "native",
    double_softmax: bool = True,
) -> float:
    """`sfda` with the self-inclusion artefact of its in-sample scoring removed."""
    X = np.asarray(X, dtype=np.float64).copy()
    y = np.asarray(y)
    n = len(y)

    if standardise:
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    if pca_dim is not None and 0 < int(pca_dim) < min(X.shape):
        Xc = X - X.mean(axis=0)
        _, _, vt = np.linalg.svd(Xc, full_matrices=False)
        X = Xc @ vt[: int(pca_dim)].T

    a = _within_class_lw_shrinkage(X, y) if shrinkage == "ledoit_wolf" else None

    stage1 = RegularizedFDA(shrinkage=a).fit(X, y)
    prob = _fda_lno_proba(stage1, X, y, leave_frac)
    if double_softmax:
        prob = np.exp(prob) / np.exp(prob).sum(axis=1, keepdims=True)

    others = outer_class_means(X, y)
    conf = prob[np.arange(n), y].reshape(-1, 1)
    for i, c in enumerate(np.unique(y)):
        mask = y == c
        X[mask] = conf[mask] * X[mask] + (1.0 - conf[mask]) * others[i]

    stage2 = RegularizedFDA(shrinkage=stage1.shrinkage).fit(X, y)
    prob2 = _fda_lno_proba(stage2, X, y, leave_frac)
    return float(prob2[np.arange(n), y].mean())


@register
class SFDARefined(TEScore):
    """`sfda` with its in-sample self-inclusion optimism removed (this paper)."""

    name = "sfda_refined"
    paper = "de-biased Self-challenging Fisher (this paper)"
    elements = ("sfda_refined",)
    requires = ("features", "labels")
    hparams = {"leave_frac": 0.0, "standardise": False, "pca_dim": None,
               "shrinkage": "native", "double_softmax": True}
    note = (
        "sfda scored with leave-fold-out class means in both Fisher stages, which "
        "removes the self-inclusion optimism of scoring on the same points the "
        "discriminant was fitted on - a finite-sample artefact, removable at no "
        "cost in signal.  leave_frac is the fold size as a FRACTION of each class "
        "(0.0 = exact leave-one-out, the default; p>0 = leave a p-fraction out; "
        "<0 = raw), so the holdout is comparable across datasets with different "
        "class counts.  standardise/pca_dim address SFDA's residual scale and "
        "(effective-)dimension sensitivity and default OFF (partly signal on real "
        "hubs).  shrinkage='ledoit_wolf' swaps SFDA's exp(-5 lambda_max) heuristic "
        "for the optimal-risk within-class shrinkage; double_softmax=False drops "
        "the reference's confidence-flattening second softmax.  Non-defaults are "
        "A/B levers -- measure per hub."
    )

    def compute(self, probe: ProbeData, **hp) -> dict[str, float]:
        X, y = as_float_features(probe), probe.labels
        return {
            "sfda_refined": sfda_refined_score(
                X, y,
                leave_frac=float(hp["leave_frac"]),
                standardise=bool(hp["standardise"]),
                pca_dim=hp["pca_dim"],
                shrinkage=str(hp["shrinkage"]),
                double_softmax=bool(hp["double_softmax"]),
            )
        }
