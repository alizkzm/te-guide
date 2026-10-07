"""OTCE - Tan, Li & Huang, CVPR 2021."""

from __future__ import annotations

import numpy as np

from guide.core.probe import ProbeData
from guide.core.registry import register
from guide.core.sampling import random_indices
from guide.scores.base import ScoreNotApplicable, TEScore, as_float_features


def _transport_plan(Xs: np.ndarray, Xt: np.ndarray, reg: float = 0.0):
    """Exact EMD via POT when installed, otherwise entropic Sinkhorn."""
    cost = np.sqrt(
        np.maximum(
            ((Xs ** 2).sum(1)[:, None] + (Xt ** 2).sum(1)[None, :] - 2 * Xs @ Xt.T), 0.0
        )
    )
    a = np.full(Xs.shape[0], 1.0 / Xs.shape[0])
    b = np.full(Xt.shape[0], 1.0 / Xt.shape[0])

    if reg <= 0:
        try:
            import ot

            return ot.emd(a, b, cost), cost
        except ImportError:
            reg = 0.05 * float(cost.mean())

    K = np.exp(-cost / max(reg, 1e-8))
    u = np.ones_like(a)
    v = np.ones_like(b)
    for _ in range(500):
        u = a / (K @ v + 1e-300)
        v = b / (K.T @ u + 1e-300)
    return u[:, None] * K * v[None, :], cost


@register
class OTCE(TEScore):
    name = "otce"
    paper = "OTCE: A Transferability Metric for Cross-Domain Cross-Task Representations (CVPR 2021)"
    elements = ("domain_difference", "task_difference")
    requires = ("features", "labels", "logits", "source_probe")
    hparams = {"max_samples": 1500, "reg": 0.0, "seed": 0}
    note = (
        "The one score that cannot be defined without a source domain - it "
        "transports source features onto target features. Uses the source probe "
        "(ImageNet val by default), which `run.py extract` builds unless you "
        "pass --no-source. Source labels are the argmax of the source head on "
        "that probe, not the dataset's own labels. O(n_s * n_t) memory, so both "
        "sides are subsampled to `max_samples`."
    )

    def compute(self, probe: ProbeData, **hp) -> dict[str, float]:
        src = probe.attach_source_probe(required=True)
        if src.logits is None:
            raise ScoreNotApplicable("OTCE needs source-head logits on the source probe")

        k = int(hp["max_samples"])
        ti = random_indices(probe.features.shape[0], k, seed=hp["seed"])
        si = random_indices(src.features.shape[0], k, seed=hp["seed"] + 1)

        Xt = as_float_features(probe)[ti]
        yt = probe.labels[ti]
        Xs = np.asarray(src.features, dtype=np.float64)[si]
        ys = src.source_pseudo_labels[si]

        coupling, cost = _transport_plan(Xs, Xt, reg=float(hp["reg"]))
        w = float((coupling * cost).sum())

        from guide.core.linalg import one_hot

        Ys = one_hot(ys, int(ys.max() + 1))
        Yt = one_hot(yt, int(yt.max() + 1))
        joint = Ys.T @ coupling @ Yt
        joint /= joint.sum() + 1e-300

        p_s = joint.sum(axis=1, keepdims=True)
        ratio = np.divide(joint, p_s, out=np.zeros_like(joint), where=p_s > 0)
        with np.errstate(divide="ignore"):
            log_ratio = np.where(ratio > 0, np.log(ratio), 0.0)
        h = float(-(joint * log_ratio).sum())

        return {"domain_difference": w, "task_difference": h}
