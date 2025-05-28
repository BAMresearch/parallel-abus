"""Datastructures and Algorithms for model classes, inference and results."""

# Imports

import logging
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Sequence

import numpy as np
import pandas as pd
from parallel_abus.ERADistNataf import ERADist

# Datatypes

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Parameter:
    """A parameter has a name, a symbol and an unit."""

    name: str
    symbol: str
    unit: str


@dataclass(frozen=True)
class PriorParameter(Parameter):
    """A prior parameter is a parameter with a prior distribution."""

    distribution: ERADist


@dataclass(frozen=True)
class PosteriorParameter(Parameter):
    """A posterior parameter is a parameter with posterior samples."""

    samples: np.ndarray  # 1-dimensional np.float64 array


# An observeable quantity has a name, a symbol and an unit, just like a parameter
Observeable = Parameter
Observeables = Sequence[Observeable]
ObservationValues = Union[float, np.ndarray]


@dataclass(frozen=True)
class Observation:
    """An observation consists of a Sequence of observeable quantities and the observed values."""

    observeables: Observeables
    values: ObservationValues


@dataclass(frozen=True)
class Prior:
    """A prior consists of prior parameters and their correlation."""

    parameters: Sequence[PriorParameter]
    correlation: np.ndarray

    def __post_init__(self):
        """Verify parameters."""
        assert (
            self.correlation.ndim == 2
        ), f"correlation must be an 2-D array, but is: {self.correlation.shape}"
        assert (
            self.correlation.shape[0] == self.correlation.shape[1]
        ), f"correlation must be a matrix of equal shapes, but is: {self.correlation.shape}"
        assert np.allclose(
            self.correlation, self.correlation.T
        ), "correlation must be symmetric."
        assert (
            len(self.parameters) == self.correlation.shape[0]
        ), f"Lenght of parameters and shape of correlation must match, but are: {len(self.parameters)} and {self.correlation.shape}"


@dataclass(frozen=True)
class Posterior:
    """A posterior consists of posterior parameters and a model evidence and inference parameters."""

    parameters: Sequence[PosteriorParameter]
    p: np.ndarray
    logcE: float
    c: float
    lambda_par: float


@dataclass(frozen=True)
class FullResults:
    """Full inference results for debugging."""

    parameters: Sequence[Parameter]  # only used for names
    samples: np.ndarray  # dimensions: n_samples x (n_parameters + 1) x n_levels
    logcE: float
    c: float
    lambda_par: float


# Types for Likelihood and ModelClass:
ParameterSamples = np.ndarray
IndexedParameterSamples = Tuple[int, np.ndarray]


class ModelClass(metaclass=ABCMeta):
    """A model class consists of a prior and a log_likelihood."""

    prior: Prior = NotImplemented
    observeables: Observeables = NotImplemented
    data: Union[pd.DataFrame, np.ndarray] = NotImplemented
    physical_parameters_indices: Sequence[int] = NotImplemented
    uncertainty_parameters_indices: Sequence[int] = NotImplemented
    bias_parameters_indices: Sequence[int] = NotImplemented

    @abstractmethod
    def log_likelihood(self, theta: ParameterSamples) -> float:
        ...

    def indexed_log_likelihood(self, i_theta:IndexedParameterSamples) -> Tuple[int, float]:
        """Return the log-likelihood of a parameter vector theta, along with the index of the evaluated sample.
        
        This is needed, since in the algorighms the sample index has to be kept track of."""
        return i_theta[0], self.log_likelihood(i_theta[1])

    @staticmethod
    @abstractmethod
    def predict_model(
        theta: ParameterSamples,
        input: Optional[Union[Tuple[float, ...], np.ndarray]],
    ) -> ObservationValues:
        ...

    @classmethod
    @abstractmethod
    def predict_observation(
        cls,
        theta: ParameterSamples,
        input: Optional[Union[Tuple[float, ...], np.ndarray]],
    ) -> ObservationValues:
        ...

    @abstractmethod
    def __init__(self, data):
        ...
