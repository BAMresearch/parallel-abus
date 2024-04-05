"""Utils for storing and converting inference datatypes."""

import csv
import numpy as np
from os import PathLike
from typing import List

from parallel_abus.ERADistNataf import ERANataf

from .datatypes import Prior, Posterior, ModelClass, PosteriorParameter, FullResults


def prior_as_nataf(prior: Prior) -> ERANataf:
    """Transform a prior to an ERANataf object."""
    return ERANataf([p.distribution for p in prior.parameters], prior.correlation)


def draw_prior_samples(prior: Prior, n_samples: int) -> np.ndarray:
    """Draw n_samples joint samples from prior."""
    return prior_as_nataf(prior).random(n_samples)


def abus_to_posterior(
    model_class: ModelClass,
    samplesX: List[np.ndarray],
    logcE: float,
    c: float,
    lambda_par: float,
) -> Posterior:
    """Transform abus results to Posterior datatype.

    Assumes samplesX to be in the same order as the model class prior parameters.

    Args:
        model_class (ModelClass): _description_
        samplesX (List[np.ndarray]): _description_
        logcE (float): _description_
        c (float): _description_
        lambda_par (float): _description_

    Returns:
        Posterior: _description_
    """
    posterior_parameters = tuple(
        PosteriorParameter(
            name=par.name,
            symbol=par.symbol,
            unit=par.unit,
            samples=par_samples,  # TODO: richtig machen
        )
        for par, par_samples in zip(
            model_class.prior.parameters, samplesX[-1][:, :-1].T
        )
    )

    return Posterior(
        parameters=posterior_parameters,
        p=samplesX[-1][:, -1],
        logcE=logcE,
        c=c,
        lambda_par=lambda_par,
    )


def abus_to_full_results(
    model_class: ModelClass,
    samplesX: List[np.ndarray],
    logcE: float,
    c: float,
    lambda_par: float,
) -> FullResults:
    """Extract data from all subset levels."""

    return FullResults(
        parameters=model_class.prior.parameters,
        samples=np.dstack(samplesX),
        logcE=logcE,
        c=c,
        lambda_par=lambda_par,
    )


def posterior_to_csv(posterior: Posterior, file: PathLike) -> None:
    """Write posterior to a csv file."""
    if isinstance(file, PathLike) or isinstance(file, str):
        with open(file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["c", posterior.c])
            writer.writerow(["logcE", posterior.logcE])
            writer.writerow(["lambda_par", posterior.lambda_par])
            for par in posterior.parameters:
                writer.writerow([par.name, par.symbol, par.unit] + par.samples.tolist())
            writer.writerow(["p", "p", "-"] + posterior.p.tolist())
    else:  # assume file open or buffer
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["c", posterior.c])
        writer.writerow(["logcE", posterior.logcE])
        writer.writerow(["lambda_par", posterior.lambda_par])
        for par in posterior.parameters:
            writer.writerow([par.name, par.symbol, par.unit] + par.samples.tolist())
        writer.writerow(["p", "p", "-"] + posterior.p.tolist())

    return None


def posterior_from_csv(file: PathLike) -> Posterior:
    """Read posterior from a csv file."""
    if isinstance(file, PathLike) or isinstance(file, str):
        with open(file, "r") as f:
            reader = csv.reader(f)
            kwargs = {}
            parameters = []
            for row in reader:
                k = row[0]
                v = row[1:]
                if len(v) == 1:
                    kwargs[k] = float(v[0])
                else:
                    if k == "p":
                        kwargs[k] = np.array(v[2:], dtype=float)
                    else:
                        parameters.append(
                            PosteriorParameter(
                                name=k,
                                symbol=v[0],
                                unit=v[1],
                                samples=np.array(v[2:], dtype=float),
                            )
                        )
    else:  # assume file open or buffer
        reader = csv.reader(file)
        kwargs = {}
        parameters = []
        for row in reader:
            k = row[0]
            v = row[1:]
            if len(v) == 1:
                kwargs[k] = float(v[0])
            else:
                if k == "p":
                    kwargs[k] = np.array(v[2:], dtype=float)
                else:
                    parameters.append(
                        PosteriorParameter(
                            name=k,
                            symbol=v[0],
                            unit=v[1],
                            samples=np.array(v[2:], dtype=float),
                        )
                    )

    return Posterior(parameters=tuple(parameters), **kwargs)  # noqa


def full_results_to_posterior(results: FullResults) -> Posterior:
    """Convert full results to Posterior (only including last subset level).

    Args:
        results (FullResults): _description_

    Returns:
        Posterior: _description_
    """
    posterior_parameters = tuple(
        PosteriorParameter(
            name=p.name,
            symbol=p.symbol,
            unit=p.unit,
            samples=samples,
        )
        for p, samples in zip(results.parameters, results.samples[:, :-1, -1].T)
    )

    return Posterior(
        parameters=posterior_parameters,
        p=results.samples[:, -1, -1],
        logcE=results.logcE,
        c=results.c,
        lambda_par=results.lambda_par,
    )
