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
    """The score cannot run on this probe."""


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
        """`check` -> `compute` -> timed elements."""
        self.check(probe)
        hp = self.resolved_hparams(**overrides)

        t0 = time.perf_counter()
        raw = self.compute(probe, **hp)
        dt = time.perf_counter() - t0

        elements = {}
        for k, v in raw.items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                fv = float("nan")
            elements[k] = fv
        return ScoreOutput(elements=elements, run_time=dt, hyperparameters=hp)


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
