"""Core functionality for bayesian inference."""

from parallel_abus import aBUS_SuS

from .datatypes import FullResults, ModelClass, Posterior
from .utils import abus_to_posterior, prior_as_nataf, abus_to_full_results


def run_model_and_return_posterior(
    model_class: ModelClass,
    abus_n: int = 3000,
    abus_p0: float = 0.1,
    seeds=None,
    lambda_par=None,
) -> Posterior:
    """Run a given model class and return posterior.

    Args:
        model_class (ModelClass): [description]

    Returns:
        Posterior: [description]
    """
    # 0) assertions, set up logging
    assert 0.0 < abus_p0 < 1.0, "Subset Probability has to be between 0 and 1."

    # 1) start inference
    # h, samplesU, samplesX, cE, c, sigma
    _, _, samplesX, logcE, c, _, lambda_par = aBUS_SuS(
        abus_n,
        abus_p0,
        model_class.log_likelihood,
        prior_as_nataf(model_class.prior),
        seeds,
        lambda_par,
    )

    return abus_to_posterior(model_class, samplesX, logcE, c, lambda_par)


def run_model_and_return_full_results(
    model_class: ModelClass,
    abus_n: int = 3000,
    abus_p0: float = 0.1,
    seeds=None,
    lambda_par=None,
) -> FullResults:
    """Run a given model class and return full results.

    Args:
        model_class (ModelClass): [description]

    Returns:
        FullResults: [description]
    """
    # 0) assertions, set up logging
    assert 0.0 < abus_p0 < 1.0, "Subset Probability has to be between 0 and 1."

    # 1) start inference
    # h, samplesU, samplesX, cE, c, sigma
    _, _, samplesX, logcE, c, _, lambda_par = aBUS_SuS(
        abus_n,
        abus_p0,
        model_class.log_likelihood,
        prior_as_nataf(model_class.prior),
        seeds,
        lambda_par,
    )

    return abus_to_full_results(model_class, samplesX, logcE, c, lambda_par)
