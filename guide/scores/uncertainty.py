"""Label-free *uncertainty* scores."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from guide.core.probe import ProbeData
from guide.core.registry import register
from guide.core.sampling import random_indices, resolve_n_samples
from guide.scores.base import ScoreNotApplicable, SingleElementScore, TEScore, as_float_features


def _probs(probe: ProbeData, k=None, seed: int = 0) -> np.ndarray:
    """`k` accepts "all" / None / an int; anything >= N uses every row."""
    p = probe.probs
    n = resolve_n_samples(k, p.shape[0])
    if n < p.shape[0]:
        p = p[random_indices(p.shape[0], n, seed=seed)]
    return p


def _entropy(p: np.ndarray, normalise: bool = True) -> np.ndarray:
    h = -(p * np.log(p + 1e-12)).sum(axis=1)
    return h / np.log(p.shape[1]) if normalise else h


@register
class NormalisedEntropy(SingleElementScore):
    """TE / complete."""

    name = "norm_entropy"
    paper = "Ranked from Within (ICML 2025) - normalised predictive entropy"
    elements = ("norm_entropy",)
    requires = ("logits",)
    label_free = True
    hparams = {"num_samples": None, "seed": 0}

    def value(self, probe: ProbeData, **hp) -> float:
        p = _probs(probe, hp.get("num_samples"), int(hp["seed"]))
        return float(-_entropy(p, normalise=True).mean())


@register
class NegativeLogLikelihood(SingleElementScore):
    """TE / complete."""

    name = "nll"
    paper = "Ranked from Within (ICML 2025) - mean negative log-likelihood"
    elements = ("nll",)
    requires = ("logits",)
    label_free = True
    hparams = {"num_samples": None, "seed": 0}

    def value(self, probe: ProbeData, **hp) -> float:
        p = _probs(probe, hp.get("num_samples"), int(hp["seed"]))
        return float(np.log(p.max(axis=1) + 1e-12).mean())


@register
class MaxSoftmaxProbability(SingleElementScore):
    """TRIVIAL."""

    name = "msp"
    paper = "mean max-softmax confidence (naive uncertainty baseline)"
    elements = ("msp",)
    requires = ("logits",)
    label_free = True
    trivial = True
    hparams = {"num_samples": None, "seed": 0}

    def value(self, probe: ProbeData, **hp) -> float:
        return float(_probs(probe, hp.get("num_samples"), int(hp["seed"])).max(axis=1).mean())


@register
class SelfConsistency(SingleElementScore):
    """TE / INCOMPLETE - needs stochastic generations, which a deterministic image classifier does not produce."""

    name = "self_consistency"
    paper = "Ranked from Within (ICML 2025) - Sample_BLEU / Sample_BERT"
    elements = ("self_consistency",)
    requires = ()
    label_free = True
    status = "partial"
    note = (
        "Defined only for generative models: it needs T>1 stochastic decodes per "
        "input (temperature 0.7) and a text-similarity function. A deterministic "
        "vision classifier has a single deterministic output, so this variant of "
        "the paper is not transferable. Use `norm_entropy` / `nll` / `atc` instead."
    )

    def value(self, probe: ProbeData, **hp) -> float:
        raise ScoreNotApplicable(self.note)


@register
class ATC(TEScore):
    """TE / complete when the source probe exists, otherwise falls back."""

    name = "atc"
    paper = "Average Thresholded Confidence (Garg et al., ICLR 2022)"
    elements = ("atc_mc", "atc_ne")
    requires = ("logits",)
    label_free = True
    hparams = {"num_samples": None, "seed": 0, "fallback": "target_quantile"}
    note = (
        "Calibrates the threshold on the source probe (ImageNet val) using the "
        "model's published ImageNet top-1 as the source accuracy. Without a "
        "source probe it falls back to the raw mean confidence, which is `msp` "
        "-- so check `diagnostics.calibrated` before comparing across models."
    )

    @staticmethod
    def _scores(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return p.max(axis=1), -_entropy(p, normalise=False)

    def compute(self, probe: ProbeData, **hp) -> dict[str, float]:
        p_t = _probs(probe, hp.get("num_samples"), int(hp["seed"]))
        mc_t, ne_t = self._scores(p_t)

        src = probe.attach_source_probe(required=False)
        acc_src = probe.meta.get("source_top1")
        if src is None or src.logits is None or acc_src is None:
            return {"atc_mc": float(mc_t.mean()), "atc_ne": float(ne_t.mean())}

        p_s = _probs(src, hp.get("num_samples"), int(hp["seed"]))
        mc_s, ne_s = self._scores(p_s)
        err = 1.0 - float(acc_src) / 100.0

        def atc(s_src: np.ndarray, s_tgt: np.ndarray) -> float:
            t = float(np.quantile(s_src, np.clip(err, 0.0, 1.0)))
            return float((s_tgt >= t).mean())

        return {"atc_mc": atc(mc_s, mc_t), "atc_ne": atc(ne_s, ne_t)}


@register
class LogitEntropy(SingleElementScore):
    """TE / complete."""

    name = "entropy"
    paper = "Logit-Based Entropy (zero-shot VLM-ranking baseline)"
    elements = ("entropy",)
    requires = ("logits",)
    label_free = True
    hparams = {"num_samples": "all", "seed": 0}
    note = (
        "For a CLIP-style model the logits are image-vs-class-text similarities; "
        "for the supervised hubs here they are the source head's logits."
    )

    def value(self, probe: ProbeData, **hp) -> float:
        p = _probs(probe, hp["num_samples"], int(hp["seed"]))
        return float(-_entropy(p, normalise=False).mean())


@register
class RepresentationVariance(TEScore):
    """TE / complete."""

    name = "rep_var"
    paper = "Representation Variance (zero-shot VLM-ranking baseline)"
    elements = ("rep_var_feature", "rep_var_logit")
    requires = ("features",)
    label_free = True
    hparams = {"num_samples": "all", "seed": 0, "standardise": True}
    note = (
        "The paper sums variance over the vision *and* language projection "
        "layers of a VLM. For a vision classifier the two available projections "
        "are the penultimate features and the logits; the default recipe weights "
        "them equally and negates the sum."
    )

    @staticmethod
    def _var(M: np.ndarray, standardise: bool) -> float:
        if standardise:
            M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
        return float(M.var(axis=0).sum())

    def compute(self, probe: ProbeData, **hp) -> dict[str, float]:
        seed = int(hp["seed"])
        k = resolve_n_samples(hp["num_samples"], probe.features.shape[0])
        idx = random_indices(probe.features.shape[0], k, seed=seed)
        out = {"rep_var_feature": self._var(as_float_features(probe)[idx], hp["standardise"])}
        if probe.logits is not None:
            out["rep_var_logit"] = self._var(
                np.asarray(probe.logits, dtype=np.float64)[idx], hp["standardise"]
            )
        return out


@register
class INB(SingleElementScore):
    """TRIVIAL."""

    name = "inb"
    paper = "ImageNet Baseline (INB) - the common-practice model-selection heuristic"
    elements = ("inb",)
    requires = ("meta",)
    trivial = True
    label_free = True

    def value(self, probe: ProbeData, **hp) -> float:
        v = probe.meta.get("source_top1")
        if v is None:
            raise ScoreNotApplicable(
                "no published ImageNet top-1 for this backbone "
                "(self-supervised models have no source classifier)"
            )
        return float(v)


@register
class FreeEnergy(SingleElementScore):
    """TRIVIAL."""

    name = "free_energy"
    paper = "free energy of the source head (OOD baseline)"
    elements = ("free_energy",)
    requires = ("logits",)
    trivial = True
    label_free = True

    def value(self, probe: ProbeData, **hp) -> float:
        return float(logsumexp(np.asarray(probe.logits, dtype=np.float64), axis=1).mean())
