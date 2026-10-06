"""PAC-Bayesian TE scores: PACTran (Dirichlet / Gamma / Gaussian)."""

from __future__ import annotations

import numpy as np
import scipy.optimize
import scipy.special

from guide.core.linalg import one_hot, pca_reduce
from guide.core.probe import ProbeData
from guide.core.registry import register
from guide.scores.base import TEScore, as_float_features


def pactran_dirichlet(prob: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> float:
    """PACTran-Dirichlet cost (lower = better)."""
    Y = one_hot(y)
    soft_sum = Y.sum(axis=0)[:, None]
    a0 = alpha * soft_sum / soft_sum.sum() + 1e-10

    qz = prob
    log_s = np.log(prob + 1e-10)
    aw = a0
    log_qz = log_s
    for _ in range(10):
        aw = a0 + np.einsum("BY,BZ->YZ", Y, qz)
        logits_qz = (
            log_s
            + Y @ scipy.special.digamma(aw)
            - scipy.special.digamma(aw.sum(axis=0)).reshape(1, -1)
        )
        log_qz = logits_qz - scipy.special.logsumexp(logits_qz, axis=-1, keepdims=True)
        qz = np.exp(log_qz)

    log_c0 = scipy.special.loggamma(a0.sum()) - scipy.special.loggamma(a0).sum()
    log_c = scipy.special.loggamma(aw.sum(axis=0)) - scipy.special.loggamma(aw).sum(axis=0)
    pac_dir = np.sum(log_c0 - log_c - np.sum(qz * (log_qz - log_s), axis=0))
    return float(-pac_dir / Y.size)


def pactran_gamma(prob: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> float:
    """PACTran-Gamma cost (lower = better)."""
    Y = one_hot(y)
    soft_sum = Y.sum(axis=0)[:, None]
    a0 = alpha * soft_sum / soft_sum.sum() + 1e-10
    beta = 1.0

    qz = prob
    s = prob
    log_s = np.log(prob + 1e-10)
    aw, bw = a0, beta
    lw = s.sum(axis=-1, keepdims=True) * (aw / bw).sum()
    log_qz = log_s
    for _ in range(10):
        aw = a0 + np.einsum("BY,BZ->YZ", Y, qz)
        lw = s @ (aw / bw).sum(axis=0)[:, None]
        logits_qz = log_s + Y @ (scipy.special.digamma(aw) - np.log(bw))
        log_qz = logits_qz - scipy.special.logsumexp(logits_qz, axis=-1, keepdims=True)
        qz = np.exp(log_qz)

    pac = np.sum(
        scipy.special.loggamma(a0)
        - scipy.special.loggamma(aw)
        + aw * np.log(bw)
        - a0 * np.log(beta)
    ) + np.sum(np.sum(qz * (log_qz - log_s), axis=-1) + np.log(np.squeeze(lw, axis=-1)) - 1.0)
    return float(pac / Y.size + 1.0)


def pactran_gaussian(X: np.ndarray, y: np.ndarray, lda_factor: float = 1.0) -> float:
    """PACTran-Gaussian cost (lower = better) - the L-BFGS variant of the paper."""
    Y = one_hot(y)
    n_classes = Y.shape[1]
    X = X - X.mean(axis=0, keepdims=True)
    bs, d = X.shape
    kd = d * n_classes
    ldas2 = lda_factor * bs
    dinv = 1.0 / float(d)

    def loss(theta):
        theta = theta.reshape(d + 1, n_classes)
        w, b = theta[:d], theta[d:]
        logits = X @ w + b
        log_qz = logits - scipy.special.logsumexp(logits, axis=-1, keepdims=True)
        xent = np.sum(Y * (np.log(Y + 1e-10) - log_qz)) / bs
        return xent + 0.5 * np.sum(w ** 2) / ldas2

    def grad(theta):
        theta = theta.reshape(d + 1, n_classes)
        w, b = theta[:d], theta[d:]
        logits = X @ w + b
        gf = scipy.special.softmax(logits, axis=-1) - Y
        gf /= bs
        gw = X.T @ gf + w / ldas2
        gb = gf.sum(axis=0, keepdims=True)
        return np.ravel(np.concatenate([gw, gb], axis=0))

    def grad2(theta):
        theta = theta.reshape(d + 1, n_classes)
        w, b = theta[:d], theta[d:]
        logits = X @ w + b
        p = scipy.special.softmax(logits, axis=-1)
        g2f = p - p ** 2
        g2w = (X ** 2).T @ g2f + 1.0 / ldas2
        g2b = g2f.sum(axis=0, keepdims=True)
        return np.ravel(np.concatenate([g2w, g2b], axis=0))

    rng = np.random.RandomState(2)
    theta0 = np.ravel(
        np.concatenate([rng.normal(size=(d, n_classes)) * 0.03, np.zeros((1, n_classes))], axis=0)
    )
    theta = scipy.optimize.minimize(
        loss, theta0, method="L-BFGS-B", jac=grad, options=dict(maxiter=100), tol=1e-6
    ).x

    pac_opt = loss(theta)
    sigma2_inv = np.sum(grad2(theta)) * ldas2 / kd + 1e-10
    s2 = {10.0: 100.0, 1.0: 10.0, 0.1: 1.0}.get(lda_factor, 10.0) * dinv
    return float(pac_opt + 0.5 * kd / ldas2 * s2 * np.log(sigma2_inv))


@register
class PACTran(TEScore):
    name = "pactran"
    paper = "PACTran: PAC-Bayesian Metrics for Transferability (ECCV 2022)"
    elements = ("pactran_dirichlet", "pactran_gamma", "pactran_gaussian")
    hparams = {
        "alpha": 1.0,
        "lda_factor": 1.0,
        "pca_dim": 64,
        "seed": 0,
        "variants": ("dirichlet", "gamma", "gaussian"),
    }
    note = (
        "All three priors, sign-flipped to higher-is-better. Dirichlet/Gamma need "
        "a probability vector: the source head when available, otherwise a "
        "softmax over PCA-reduced features."
    )

    def compute(self, probe: ProbeData, **hp) -> dict[str, float]:
        from guide.core.linalg import softmax

        X, y = as_float_features(probe), probe.labels
        Xr = pca_reduce(X, int(hp["pca_dim"]), seed=hp["seed"])

        if probe.logits is not None:
            prob = probe.probs
        else:
            prob = softmax(Xr - Xr.mean(axis=0, keepdims=True))

        out: dict[str, float] = {}
        variants = tuple(hp["variants"])
        if "dirichlet" in variants:
            out["pactran_dirichlet"] = -pactran_dirichlet(prob, y, float(hp["alpha"]))
        if "gamma" in variants:
            out["pactran_gamma"] = -pactran_gamma(prob, y, float(hp["alpha"]))
        if "gaussian" in variants:
            out["pactran_gaussian"] = -pactran_gaussian(Xr, y, float(hp["lda_factor"]))
        return out
