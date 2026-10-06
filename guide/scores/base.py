"""`TEScore` - the interface every transferability score implements."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from guide.core.probe import ProbeData

REQUIREMENTS = {
    "features",
    "labels",
    "logits",
    "weights",
    "source_probe",
    "meta",
}


class ScoreNotApplicable(RuntimeError):
    """The score cannot run on this probe (e.g."""


@dataclass
class ScoreOutput:
    elements: dict[str, float]
    run_time: float
    hyperparameters: dict


class TEScore:
    name: str = ""
    paper: str = ""
    elements: tuple[str, ...] = ()
    requires: tuple[str, ...] = ("features", "labels")
    trivial: bool = False
    label_free: bool = False
    status: str = "ready"
    note: str = ""
    hparams: dict = {}
    _resamplable: bool | None = None
    resample_seed: int = 0

    @property
    def resamplable(self) -> bool:
        """Whether the `variance` / `richardson` run flags apply to this score."""
        if self._resamplable is not None:
            return bool(self._resamplable)
        return len(self.elements) <= 1 and set(self.requires) <= {"features", "labels"}

    @property
    def _primary_element(self) -> str:
        return self.elements[0] if self.elements else self.name

    @property
    def kind(self) -> str:
        """"TE" for a published method, "TRIVIAL" for a heuristic baseline."""
        return "TRIVIAL" if self.trivial else "TE"

    @property
    def completeness(self) -> str:
        """"complete" or "incomplete" (see `status` / `note`)."""
        return "complete" if self.status == "ready" else "incomplete"

    def compute(self, probe: ProbeData, **hp) -> dict[str, float]:
        raise NotImplementedError

    def resolved_hparams(self, **overrides) -> dict:
        hp = dict(self.hparams)
        hp.update({k: v for k, v in overrides.items() if v is not None})
        return hp

    def check(self, probe: ProbeData) -> None:
        """Raise `ScoreNotApplicable` when the probe lacks something needed."""
        if "features" in self.requires and probe.features is None:
            raise ScoreNotApplicable("probe has no features")
        if "labels" in self.requires and probe.labels is None:
            raise ScoreNotApplicable("probe has no labels")
        if "logits" in self.requires and probe.logits is None:
            raise ScoreNotApplicable(
                f"{probe.model_name} is head-less: no source logits, "
                f"so {self.name} is undefined for it"
            )
        if "weights" in self.requires and not probe.weights:
            raise ScoreNotApplicable("probe has no cached weight matrices")
        if "source_probe" in self.requires:
            probe.attach_source_probe(required=True)

    def run(self, probe: ProbeData, **overrides) -> ScoreOutput:
        """`check` -> `compute` (or the resampling plugin) -> timed elements."""
        self.check(probe)
        hp = self.resolved_hparams(**overrides)
        variance = bool(hp.pop("variance", False))
        richardson = bool(hp.pop("richardson", False))
        n_iter = hp.pop("resample_iter", None)
        seed = int(hp.pop("resample_seed", self.resample_seed))

        t0 = time.perf_counter()
        mode = None
        if (variance or richardson) and self.resamplable:
            mode = "richardson" if richardson else "variance"
            raw = self._resample(probe, mode, hp, n_iter, seed)
        else:
            raw = self.compute(probe, **hp)
        dt = time.perf_counter() - t0

        elements = {}
        for k, v in raw.items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                fv = float("nan")
            elements[k] = fv
        if mode is not None:
            hp = {**hp, "resample": mode, "resample_seed": seed}
        return ScoreOutput(elements=elements, run_time=dt, hyperparameters=hp)

    def _resample(self, probe: ProbeData, mode: str, hp: dict,
                  n_iter: int | None, seed: int) -> dict[str, float]:
        """Re-run this score on stratified resamples via the plugin engine."""
        from guide.scores.resample import (MODE_DEFAULTS, draw_elements,
                                            resample_score_draws)

        default_iter, frac = MODE_DEFAULTS[mode]
        prim = self._primary_element

        def scalar_fn(pr: ProbeData) -> float:
            return float(self.compute(pr, **hp)[prim])

        vals = resample_score_draws(
            scalar_fn, probe, mode,
            n_iter=int(n_iter or default_iter), seed=seed, frac=frac,
        )
        return draw_elements(prim, vals)


class SingleElementScore(TEScore):
    """Subclass and implement `value()`; the element is named after the score."""

    def value(self, probe: ProbeData, **hp) -> float:
        raise NotImplementedError

    @property
    def _element_name(self) -> str:
        return self.elements[0] if self.elements else self.name

    def compute(self, probe: ProbeData, **hp) -> dict[str, float]:
        return {self._element_name: float(self.value(probe, **hp))}


def as_float_features(probe: ProbeData) -> np.ndarray:
    return np.ascontiguousarray(probe.features, dtype=np.float64)
