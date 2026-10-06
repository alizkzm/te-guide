"""Importing this package registers every score in `guide.core.registry`."""

from guide.scores import (
    bayesian,
    classic,
    dynamics,
    energy,
    external,
    geometry,
    graph_te,
    lda,
    optimal_transport,
    perturbation,
    refined,
    separability,
    spectral,
    span,
    trivial,
    uncertainty,
    weights,
)
from guide.scores.base import ScoreNotApplicable, ScoreOutput, TEScore

__all__ = ["TEScore", "ScoreOutput", "ScoreNotApplicable"]
