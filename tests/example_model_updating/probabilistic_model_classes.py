"""Probabilistic model classes for Bayesian inference of rc beam.


m1_timoshenko_f     Undamaged; Timoshenko, determin. nu; OpenSEES FE model, 1cm fine; Prediction Errors Multivariate Normal; Bias terms;


"""

import logging
import pathlib
from typing import Any, Tuple, Union

import numpy as np
import pandas as pd

from inference import (
    ModelClass,
    Observeable,
    ParameterSamples,
    Prior,
    PriorParameter,
)
from mechanical_models import (
    global_model_m1_undamaged_stat_timoshenko_fine,
    mm,
)
from parallel_abus.ERADistNataf import ERADist
from scipy import stats

REALMIN = np.finfo(np.double).tiny  # to prevent log(0)

logger = logging.getLogger(__name__)

RANDOM_GENERATOR = np.random.default_rng(1234567)



def static_data_to_numpy(data: pd.DataFrame) -> np.ndarray:
    """Sanitize DataFrame input to common numpy format.

    The data contains n_o observations of the n_i inputs and n_d observeable quantities.

    The observed temperature is labled as 'NennTemperatur' or 'Temperatur' or 'temperature'.

    Observed positions are labled as 'Position' or 'position'.
    If they are not numerical, they are interpreted as:
    Pos_1 = 0.68 m
    Pos_2 = 1.36 m
    Pos_3 = 1.69 m

    The observed measurements are 'W1', 'W2', 'W3' or 'disp_W1', 'disp_W2', 'disp_W3'
    for the displacements and 'Neigungssensor_3_k01', 'Neigungssensor_4_k02' or 'rot_N1', 'rot_N2'
    for the rotations. The units are assumed to be [m] and [rad].

    The data are stored as a numpy array (n_o x (n_i + n_d)) with columns ordering:
    Temperature, position, W1, W2, W3, N1, N2.
    """
    columns = set(data.columns)

    temperature_column = columns & {"NennTemperatur", "Temperatur", "temperature", "Temperature"}
    assert len(temperature_column) == 1
    temperature_column = temperature_column.pop()

    position_column = columns & {"Position", "position"}
    assert len(position_column) == 1
    position_column = position_column.pop()

    W1_column = columns & {"W1", "disp_W1"}
    assert len(W1_column) == 1
    W1_column = W1_column.pop()

    W2_column = columns & {"W2", "disp_W2"}
    assert len(W2_column) == 1
    W2_column = W2_column.pop()

    W3_column = columns & {"W3", "disp_W3"}
    assert len(W3_column) == 1
    W3_column = W3_column.pop()

    N1_column = columns & {"Neigungssensor_3_k01", "rot_N1", "N1"}
    assert len(N1_column) == 1
    N1_column = N1_column.pop()

    N2_column = columns & {"Neigungssensor_4_k02", "rot_N2", "N2"}
    assert len(N2_column) == 1
    N2_column = N2_column.pop()

    return data.replace(["Pos_1", "Pos_2", "Pos_3"], [0.68, 1.36, 1.69])[
        [
            temperature_column,
            position_column,
            W1_column,
            W2_column,
            W3_column,
            N1_column,
            N2_column,
        ]
    ].to_numpy()



class M1_Timoshenko_Beam(ModelClass):
    """Model 1: Timoshenko beam.

    The beam is assumed to be in undamaged state.

    Static load tests with a moving load of 3 kN.
    Observeable quantities are three deflections and two rotations.

    The mechanical partial model class predicts these observeable quantities
    based on the Timoshenko hypothesis with openSEES finite element models.

    This model is discretized finely with 1cm elements.

    Temperature dependent stiffness modelled as:
    $$EI(T) = EI_0 + t * \alpha$$

    The poisson's ratio is deterministically set to 0.2

    Prediction errors of the rotations and the deflection follow
    a multivariate normal distribution with marginal means of zero
    and standard deviations of sigma_e,W1; sigma_e,W2; sigma_e,W3;
    sigma_e,N1 and sigma_e,N2

    All observations are assumed to have additive biases.

    Correlations between prediction errors are modelled with a
    correllation length l_rho of an exponential distribution.

    # Correlations between prediction errors at the same temperature: k

    Priors (mean, stdev):
    EI_0         ~ Lognormal(8.8 MNm², 8.8 MNm² * 0.15)
    alpha        ~ Normal(-0.01, -0.01 * 0.25)
    sigma_e_W1   ~ Lognormal(0.01mm, 0.01mm * 0.5)
    sigma_e_W2   ~ Lognormal(0.01mm, 0.01mm * 0.5)
    sigma_e_W3   ~ Lognormal(0.01mm, 0.01mm * 0.5)
    sigma_e_N1   ~ Lognormal(0.015mrad, 0.015mrad * 0.5)
    sigma_e_N2   ~ Lognormal(0.015mrad, 0.015mrad * 0.5)
    l_rho        ~ Logormal(1.4 meter, 1.4 meter * 1.00)
    b_W1   ~ Normal(0.0mm, 0.01mm * 0.15)
    b_W2   ~ Normal(0.1mm, 0.1mm * 0.5)
    b_W3   ~ Normal(0.0mm, 0.01mm * 0.15)
    b_N1   ~ Normal(0.0mrad, 0.015mrad * 0.15)
    b_N2   ~ Normal(0.0mrad, 0.015mrad * 0.15)

    """

    # class property, not instance property
    prior = Prior(
        (
            PriorParameter(
                "E-Modul",
                "EI_0",
                "MNm²",
                ERADist("lognormal", "MOM", (8.8, 8.8 * 0.15)),
            ),
            PriorParameter(
                "alpha",
                r"\alpha",
                "MNm²/K",
                ERADist("normal", "MOM", (-0.01, 0.01 * 0.25)),
            ),
            PriorParameter(
                "sigma_e_W1",
                r"\sigma_{e,W1}",
                "mm",
                ERADist("lognormal", "MOM", (0.01, 0.01 * 0.5)),
            ),
            PriorParameter(
                "sigma_e_W2",
                r"\sigma_{e,W2}",
                "mm",
                ERADist("lognormal", "MOM", (0.01, 0.01 * 0.5)),
            ),
            PriorParameter(
                "sigma_e_W3",
                r"\sigma_{e,W3}",
                "mm",
                ERADist("lognormal", "MOM", (0.01, 0.01 * 0.5)),
            ),
            PriorParameter(
                "sigma_e_N1",
                r"\sigma_{e,N1}",
                "mrad",
                ERADist("lognormal", "MOM", (0.015, 0.015 * 0.5)),
            ),
            PriorParameter(
                "sigma_e_N2",
                r"\sigma_{e,N2}",
                "mrad",
                ERADist("lognormal", "MOM", (0.015, 0.015 * 0.5)),
            ),
            PriorParameter(
                "l_rho",
                r"l_{\rho}",
                "m",
                ERADist("lognormal", "MOM", (1.4, 1.4 * 1.0)),
            ),
            PriorParameter(
                "b_W1",
                r"b_{W1}",
                "mm",
                ERADist("normal", "MOM", (0.0, 0.01 * 0.15)),
            ),
            PriorParameter(
                "b_W2",
                r"b_{W2}",
                "mm",
                ERADist("normal", "MOM", (0.1, 0.1 * 0.05)),
            ),
            PriorParameter(
                "b_W3",
                r"b_{W3}",
                "mm",
                ERADist("normal", "MOM", (0.0, 0.01 * 0.15)),
            ),
            PriorParameter(
                "b_N1",
                r"b_{N1}",
                "mrad",
                ERADist("normal", "MOM", (0.0, 0.015 * 0.15)),
            ),
            PriorParameter(
                "b_N2",
                r"b_{N2}",
                "mrad",
                ERADist("normal", "MOM", (0.0, 0.015 * 0.15)),
            ),
        ),
        np.eye(13),
    )

    physical_parameters_indices: Tuple[int, ...] = (0, 1)
    uncertainty_parameters_indices: Tuple[int, ...] = (2, 3, 4, 5, 6, 7)
    bias_parameters_indices: Tuple[int, ...] = (8, 9, 10, 11, 12)

    observeables = (
        Observeable(name="disp_W1", symbol="W1", unit="m"),
        Observeable(name="disp_W2", symbol="W2", unit="m"),
        Observeable(name="disp_W3", symbol="W3", unit="m"),
        Observeable(name="rot_N1", symbol="N1", unit="rad"),
        Observeable(name="rot_N2", symbol="N2", unit="rad"),
    )

    def __init__(self, data: pd.DataFrame):
        """Instatiate a model class given data.

        The data contains n_o observations of the n_i inputs and n_d observeable quantities.

        The observed temperature is labled as 'NennTemperatur' or 'Temperatur' or 'temperature'.

        Observed positions are labled as 'Position' or 'position'.
        If they are not numerical, they are interpreted as:
        Pos_1 = 0.68 m
        Pos_2 = 1.36 m
        Pos_3 = 1.69 m

        The observed measurements are 'W1', 'W2', 'W3' or 'disp_W1', 'disp_W2', 'disp_W3'
        for the displacements and 'Neigungssensor_3_k01', 'Neigungssensor_4_k02' or 'rot_N1', 'rot_N2'
        for the rotations. The units are assumed to be [m] and [rad].

        The data are stored as a numpy array (n_o x (n_i + n_d)) with columns ordering:
        Temperature, position, W1, W2, W3, N1, N2.

        Args:
            data (pd.DataFrame): A pandas DataFrame containing observed data and inputs.
        """

        self.data = static_data_to_numpy(data)

    @staticmethod
    def _random_field_correlation(correlation_length: float) -> np.ndarray:
        """Calculate the correlation matrix of a single position based on a random field."""
        # distance matrix hardcoded:
        return np.exp(
            -np.array(
                [
                    [0.00, 0.68, 1.36, 0.68, 2.04],
                    [0.68, 0.00, 0.68, 1.36, 1.36],
                    [1.36, 0.68, 0.00, 2.04, 0.68],
                    [0.68, 1.36, 2.04, 0.00, 2.72],
                    [2.04, 1.36, 0.68, 2.72, 0.00],
                ]
            )
            / correlation_length
        )

    @classmethod
    def sigma(
        cls,
        sigma_e_W1: float,
        sigma_e_W2: float,
        sigma_e_W3: float,
        sigma_e_N1: float,
        sigma_e_N2: float,
        l_rho: float,
    ) -> np.ndarray:
        stdev_vector = np.array(
            [sigma_e_W1, sigma_e_W2, sigma_e_W3, sigma_e_N1, sigma_e_N2]
        )
        Sigma = (
            np.diag(stdev_vector)
            @ cls._random_field_correlation(l_rho)
            @ np.diag(stdev_vector)
        )

        assert np.allclose(Sigma, Sigma.T), "Sigma should be symmetric"

        return Sigma

    @classmethod
    def predict_observation(
        cls, theta: ParameterSamples, input: Union[Tuple[float, ...], np.ndarray]
    ) -> np.ndarray:
        model_predictions = cls.predict_model(theta, input)  # in meter

        error_samples = (
            stats.multivariate_normal(cov=cls.sigma(*theta[2:8])).rvs() * mm
        )  # cast from mm/mrad to meter/rad
        biases = np.array(theta[8:13]) * mm  # cast from mm/mrad to meter/rad
        return model_predictions + biases + error_samples

    def log_likelihood(self, theta: ParameterSamples) -> float:
        n_observations = self.data.shape[0]
        n_d = len(self.observeables)  # dimension of 1 observation: 5 values

        sigma = self.sigma(*theta[2:8])  # in millimeter & mrad
        biases = np.array(theta[8:13])

        det_sigma = np.linalg.det(sigma)
        assert det_sigma > 0, "det(Sigma) should be > 0"
        sigma_inv = np.linalg.inv(sigma)

        term_1 = n_observations * (
            -n_d / 2 * np.log(2 * np.pi) - 0.5 * np.log(det_sigma + REALMIN)
        )
        # iterate over unique temperatures and load positions:
        unique_input_combinations = np.unique(self.data[:, 0:2], axis=0)
        J_i = []
        for input_comb in unique_input_combinations:
            model_predictions = self.predict_model(theta, input_comb)
            measurements = self.data[
                np.all(self.data[:, 0:2] == input_comb, axis=1), 2:
            ]
            prediction_errors = (
                (measurements - model_predictions) / mm - biases
            )  # cast to millimeter & mrad for likelihood, biases are already in mm & mrad # cast to millimeter for likelihood
            result = (
                np.einsum(
                    "ij,jk,ik->i", prediction_errors, sigma_inv, prediction_errors
                )
                / 2
            )
            J_i.append(np.sum(result))
        return term_1 - np.sum(J_i)

    @staticmethod
    def predict_model(
        theta: ParameterSamples,
        input: Union[Tuple[float, ...], np.ndarray[Any, Any], None],
    ) -> np.ndarray:
        # input: temperature, position
        temperature = input[0]
        position = input[1]
        logger.debug(f"{input=}")
        return global_model_m1_undamaged_stat_timoshenko_fine(
            x_F=position, EI_0=theta[0], alpha=theta[1], T=temperature
        )
