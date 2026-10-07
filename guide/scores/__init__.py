"""Importing this package registers every score in `guide.core.registry`."""

from guide.scores import (
    bayesian,
    classic,
    dynamics,
    energy,
    external,
    geometry,
    lda,
    optimal_transport,
    perturbation,
    separability,
    spectral,
    trivial,
    uncertainty,
    weights,
)
from guide.scores.base import ScoreNotApplicable, ScoreOutput, TEScore

__all__ = ["TEScore", "ScoreOutput", "ScoreNotApplicable"]
