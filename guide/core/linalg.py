"""Small numerical helpers shared by several TE scores."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

EPS = 1e-8


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def log_softmax(logits: np.ndarray) -> np.ndarray:
    return logits - logsumexp(logits, axis=1, keepdims=True)


def one_hot(y: np.ndarray, n_classes: int | None = None) -> np.ndarray:
    y = np.asarray(y).astype(int)
    n_classes = int(y.max() + 1) if n_classes is None else n_classes
    out = np.zeros((y.size, n_classes), dtype=np.float64)
    out[np.arange(y.size), y] = 1.0
    return out


def remap_labels(y: np.ndarray) -> np.ndarray:
    """Force labels into a dense 0..C-1 range (many scores index by label)."""
    _, inv = np.unique(np.asarray(y), return_inverse=True)
    return inv.astype(np.int64)


def class_means(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """[C, D] matrix of per-class feature means (classes in sorted order)."""
    classes = np.unique(y)
    return np.stack([X[y == c].mean(axis=0) for c in classes])


def outer_class_means(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """For each class c, the mean of *all other* class means (SFDA's ConfMix)."""
    means = class_means(X, y)
    c = means.shape[0]
    if c < 2:
        return means.copy()
    total = means.sum(axis=0, keepdims=True)
    return (total - means) / (c - 1)


def shrinkage_cov(X: np.ndarray, shrinkage: float = 0.0) -> np.ndarray:
    """(1-a)*Cov + a*mu*I with mu = tr(Cov)/D (Ledoit-Wolf style target)."""
    cov = np.cov(np.asarray(X, dtype=np.float64).T, bias=True)
    cov = np.atleast_2d(cov)
    d = cov.shape[0]
    if shrinkage <= 0.0:
        return cov
    mu = np.trace(cov) / d
    out = (1.0 - shrinkage) * cov
    out.flat[:: d + 1] += shrinkage * mu
    return out


def iterative_A(A: np.ndarray, max_iterations: int = 3) -> float:
    """Power-iteration estimate of the largest eigenvalue (SFDA / ETran)."""
    x = A.sum(axis=1)
    y = x
    for _ in range(max_iterations):
        tmp = A @ x
        y = tmp / (np.linalg.norm(tmp, 2) + EPS)
        tmp = A @ y
        x = tmp / (np.linalg.norm(tmp, 2) + EPS)
    return float(x.T @ A @ y)


def stable_rank(M: np.ndarray, eps: float = EPS) -> float:
    """srank(M) = ||M||_F^2 / ||M||_2^2 = sum(s_i^2) / max(s_i)^2."""
    M = np.asarray(M, dtype=np.float64)
    if M.size == 0:
        return 0.0
    s = np.linalg.svd(M, compute_uv=False)
    if s.size == 0:
        return 0.0
    return float((s ** 2).sum() / (s.max() ** 2 + eps))


def normalised_stable_rank(M: np.ndarray, eps: float = EPS) -> float:
    """srank / min(m, n), comparable across architectures."""
    M = np.asarray(M)
    if M.size == 0:
        return 0.0
    return stable_rank(M, eps) / max(1, min(M.shape))


def effective_rank(M: np.ndarray, eps: float = 1e-7) -> float:
    """exp(entropy of the normalised singular-value distribution) (RankMe)."""
    s = np.linalg.svd(np.asarray(M, dtype=np.float64), compute_uv=False)
    p = s / (np.abs(s).sum() + eps) + eps
    return float(np.exp(-(p * np.log(p)).sum()))


def coding_rate(Z: np.ndarray, eps: float = 1e-4) -> float:
    """R(Z) = 0.5 * logdet(I + Z^T Z / (n*eps)) (TransRate)."""
    Z = np.asarray(Z, dtype=np.float64)
    n, d = Z.shape
    if n == 0:
        return 0.0
    _, logdet = np.linalg.slogdet(np.eye(d) + Z.T @ Z / (n * eps))
    return 0.5 * float(logdet)


def pca_reduce(X: np.ndarray, n_components, seed: int = 0) -> np.ndarray:
    """PCA that never asks for more components than the data can give."""
    from sklearn.decomposition import PCA

    X = np.asarray(X, dtype=np.float64)
    if n_components is None:
        return X
    if isinstance(n_components, float):
        return PCA(n_components=n_components, random_state=seed).fit_transform(X)
    k = int(min(n_components, X.shape[0] - 1, X.shape[1]))
    if k < 1 or k >= min(X.shape):
        return X
    return PCA(n_components=k, random_state=seed).fit_transform(X)


def maybe_downproject(X: np.ndarray, max_dim: int | None, seed: int = 12345) -> np.ndarray:
    """Gaussian random projection down to `max_dim`."""
    if max_dim is None or X.shape[1] <= max_dim:
        return X
    rng = np.random.RandomState(seed)
    P = rng.randn(X.shape[1], max_dim) / np.sqrt(max_dim)
    return X @ P
