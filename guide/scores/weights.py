"""Weight-only transferability priors - data-free."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from guide import config
from guide.core.probe import ProbeData
from guide.core.registry import register
from guide.scores.base import ScoreNotApplicable, SingleElementScore


def _weightstats_dir() -> Path:
    return config.WEIGHTSTATS_DIR


@lru_cache(maxsize=None)
def _load_stats(model_name: str) -> dict:
    path = _weightstats_dir() / f"{model_name}.json"
    if not path.exists():
        raise ScoreNotApplicable(
            f"no weight stats for '{model_name}' at {path} - "
            f"run `python tools/weight_probe.py` first"
        )
    return json.loads(path.read_text(encoding="utf-8"))


class _WeightStatScore(SingleElementScore):
    """Reads one precomputed weight statistic; `sign` flips it to higher-is-better."""
    requires = ("meta",)
    label_free = True
    stat_key: str = ""
    sign: float = 1.0

    def value(self, probe: ProbeData, **hp) -> float:
        stats = _load_stats(probe.model_name)
        v = stats.get(self.stat_key)
        if v is None or not isinstance(v, (int, float)):
            raise ScoreNotApplicable(f"weight stat '{self.stat_key}' missing for {probe.model_name}")
        import math
        if not math.isfinite(float(v)):
            raise ScoreNotApplicable(f"weight stat '{self.stat_key}' is non-finite for {probe.model_name}")
        return self.sign * float(v)


@register
class WLogSpectralNorm(_WeightStatScore):
    name = "w_log_spectral_norm"
    paper = "mean log10 spectral norm of the weight matrices (data-free weight prior)"
    elements = ("w_log_spectral_norm",)
    trivial = True
    stat_key = "log_spectral_norm"


@register
class WLogFrobenius(_WeightStatScore):
    name = "w_log_frobenius"
    paper = "mean log10 Frobenius norm of the weight matrices (data-free weight prior)"
    elements = ("w_log_frobenius",)
    trivial = True
    stat_key = "log_frobenius"


@register
class WStableRank(_WeightStatScore):
    name = "w_stable_rank"
    paper = "mean stable rank ||W||_F^2/||W||_2^2 of the weight matrices (data-free)"
    elements = ("w_stable_rank",)
    trivial = True
    stat_key = "stable_rank"


@register
class WEffectiveRank(_WeightStatScore):
    name = "w_effective_rank"
    paper = "mean effective rank exp(H(sigma)) of the weight matrices (data-free)"
    elements = ("w_effective_rank",)
    trivial = True
    stat_key = "effective_rank"


@register
class WAlpha(_WeightStatScore):
    name = "w_alpha"
    paper = "HT-SR power-law exponent alpha (Martin & Mahoney, Nature Comms 2021)"
    elements = ("w_alpha",)
    stat_key = "alpha"
    sign = -1.0


@register
class WAlphaWeighted(_WeightStatScore):
    name = "w_alpha_weighted"
    paper = "HT-SR weighted alpha = alpha * log10(lambda_max) (Martin & Mahoney 2021)"
    elements = ("w_alpha_weighted",)
    stat_key = "alpha_weighted"
    sign = -1.0
    note = "The single best data-free predictor of test accuracy in the reference paper."


@register
class WMpSoftrank(_WeightStatScore):
    name = "w_mp_softrank"
    paper = "HT-SR Marchenko-Pastur soft rank lambda+/lambda_max (Martin & Mahoney 2021)"
    elements = ("w_mp_softrank",)
    stat_key = "mp_softrank"
    sign = -1.0
