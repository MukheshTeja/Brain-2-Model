from .base import Surrogate, get_surrogate, register
from .gaussian import GaussianSurrogate
from .phase_shuffle import PhaseShuffleSurrogate
from .covariance_matched import CovarianceMatchedSurrogate
from .max_entropy import MaxEntropySurrogate
from .stimulus_shuffle import StimulusShuffleSurrogate
from .knockout import StructureKnockout

RUNG_ORDER = ["R0", "R1", "R2", "R3", "R4", "R5"]

__all__ = [
    "Surrogate",
    "get_surrogate",
    "register",
    "GaussianSurrogate",
    "PhaseShuffleSurrogate",
    "CovarianceMatchedSurrogate",
    "MaxEntropySurrogate",
    "StimulusShuffleSurrogate",
    "StructureKnockout",
    "RUNG_ORDER",
]
