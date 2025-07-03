"""Run aBUS in parallel.

---------------------------------------------------------------------------
adaptive BUS with subset simulation
---------------------------------------------------------------------------
Created by:
Fong-Lin Wu (fonglin.wu@tum.de)
Matthias Willer (matthias.willer@tum.de)
Felipe Uribe (felipe.uribe@tum.de)
Engineering Risk Analysis Group
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Last Version 2019-07
* Update LSF with log_likelihood, while loop and clean up code
---------------------------------------------------------------------------
Input:
* N              : number of samples per level
* p0             : conditional probability of each subset
* log_likelihood : log-Likelihood function of the problem at hand
* T_nataf        : Nataf distribution object (probabilistic transformation)
---------------------------------------------------------------------------
Output:
* h        : intermediate levels of the subset simulation method
* samplesU : object with the samples in the standard normal space
* samplesX : object with the samples in the original space
* cE       : model evidence/marginal likelihood
* c        : scaling constant that holds 1/c >= Lmax
* sigma    : last spread of the proposal
---------------------------------------------------------------------------
Based on:
1."Bayesian inference with subset simulation: strategies and improvements"
   Betz et al.
   Computer Methods in Applied Mechanics and Engineering 331 (2018) 72-93.
2."Bayesian updating with structural reliability methods"
   Daniel Straub & Iason Papaioannou.
   Journal of Engineering Mechanics 141.3 (2015) 1-13.
---------------------------------------------------------------------------
"""

import logging
import multiprocessing
from functools import partial

import numpy as np
import scipy as sp

from parallel_abus.ERADistNataf import ERADist, ERANataf, ERARosen
from .aCS_aBUS_parallel import aCS_aBUS_batches as aCS_aBUS
from .utils import TimerContext

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


REALMAX = np.finfo(np.double).max
MAX_LOG = np.log(REALMAX)

MAX_LINE_LENGTH = 200
PRECISION = 8


def array_to_string(arr):
    return np.array2string(
        arr,
        max_line_width=MAX_LINE_LENGTH,
        separator=", ",
        precision=PRECISION,
        suppress_small=False,
    )


np.set_printoptions(
    edgeitems=3,
    infstr="inf",
    linewidth=MAX_LINE_LENGTH,
    nanstr="nan",
    precision=8,
    suppress=False,
    threshold=1000,
    formatter=None,
)


class ErrorWithData(Exception):
    """Exception that stores data of intermediate results."""

    def __init__(self, data, *args: object) -> None:
        super().__init__(*args)
        self.data = data


def h_LSF(pi_u, logl_hat, log_L):
    return np.log(sp.stats.norm.cdf(pi_u)) + logl_hat - log_L
    # limit state funtion for the observation event (Ref.1 Eq.12)
    # note that h_LSF = log(pi) + l(i) - leval
    # where pi = normcdf(u_j(end,:)) is the standard uniform variable of BUS


def u2x_distr(distr: ERADist, u):
    return distr.icdf(sp.stats.norm.cdf(u))  # from u to x


def u2x_transformation(transformation: ERANataf | ERARosen, u):
    return transformation.U2X(u)


def aBUS_SuS_parallel(
    N,
    p0,
    indexed_log_likelihood_fun,
    distr,
    pool: multiprocessing.pool.Pool,
    opc: str = "b",
    decrease_dependency: bool = True,
    max_it: int = 50,
):
    """Run parallel aBUS-SuS algorithm for Bayesian updating.

    This function implements the parallel version of the aBUS-SuS (Bayesian Updating with Subset Simulation)
    algorithm for Bayesian updating. It uses multiprocessing to parallelize the evaluation of the
    log-likelihood function.

    Args:
        N (int): Number of samples per subset level. Must be such that N*p0 and 1/p0 are integers.
        p0 (float): Probability of each subset level. Must be between 0 and 1.
        indexed_log_likelihood_fun (Callable): Function that takes a tuple of (index, theta) and returns a tuple of (index, log_likelihood).
            The index is used to track the order of samples.
        distr (Union[ERANataf, ERARosen, List[ERADist]]): Distribution object. Can be either:
            - ERANataf: For dependent random variables
            - ERARosen: For conditional random variables
            - List[ERADist]: For independent random variables
        pool (multiprocessing.pool.Pool): Multiprocessing pool for parallel evaluation
        opc (str, optional): Option for the algorithm. Defaults to "b".
        decrease_dependency (bool, optional): Whether to decrease dependency between samples. Defaults to True.
        max_it (int, optional): Maximum number of iterations. Defaults to 50.

    Returns:
        Tuple[np.ndarray, Dict[str, List[np.ndarray]], List[np.ndarray], float, float, np.ndarray, float]:
            - h: Array of intermediate levels
            - samplesU: Dictionary containing ordered samples in standard normal space
            - samplesX: List of samples in physical space
            - logcE: Log of the evidence
            - c: scaling constant that holds 1/c >= Lmax
            - sigma: last spread of the proposal
            - lambda_par: Final scaling parameter

    Raises:
        ValueError: If N*p0 or 1/p0 are not integers
        RuntimeError: If distr is not a valid ERANataf or ERADist object
    """
    if (N * p0 != np.fix(N * p0)) or (1 / p0 != np.fix(1 / p0)):
        raise ValueError(
            "N*p0 and 1/p0 must be positive integers. Adjust N and p0 accordingly"
        )

    # initial check if there exists a Nataf object
    if isinstance(distr, ERANataf):  # use Nataf transform (dependence)
        n = (
            len(distr.Marginals) + 1
        )  # number of random variables + p Uniform variable of BUS

        std_to_physical = partial(u2x_transformation, distr)

    elif isinstance(distr, ERARosen):
        # raise NotImplementedError("ERARosen is not implemented yet.")
        n = len(distr.Dist) + 1
        std_to_physical = partial(u2x_transformation, distr)
        # std_to_physical = lambda u: distr.U2X(u)

    elif isinstance(
        distr[0], ERADist
    ):  # use distribution information for the transformation (independence)
        # Here we are assuming that all the parameters have the same distribution !!!
        # Adjust accordingly otherwise or use an ERANataf object
        n = len(distr) + 1  # number of random variables + p Uniform variable of BUS

        std_to_physical = partial(u2x_distr, distr[0])
    else:
        raise ValueError(
            "Incorrect distribution. `distr` must be an ERADist, ERARosen, or ERANataf object!"
        )

    # def indexed_log_likelihood_fun(index: int, theta: Any) -> Tuple[int, float]:
    #     """Wrapper for the log-likelihood function."""
    #     return (index, log_likelihood(theta))

    def log_L_fun(u):
        n_samples = len(u)
        if isinstance(distr, ERANataf) or isinstance(distr, ERARosen):
            x = [u2x_transformation(distr, u_i[0 : n - 1]).flatten() for u_i in u]
        elif isinstance(distr[0], ERADist):
            x = [u2x_distr(distr[0], u_i[0 : n - 1]).flatten() for u_i in u]
        else:
            raise ValueError

        unordered_results = pool.map(
            indexed_log_likelihood_fun, list(zip(range(n_samples), x))
        )

        valid_unordered_results = [
            r for r in unordered_results if (r is not None) and (r[0] is not None)
        ]
        max_retries = 10
        i = 0
        while len(valid_unordered_results) != len(x):
            calculated_indices = set([val[0] for val in valid_unordered_results])
            all_indices = set(range(len(x)))
            missing_results_indices = all_indices - calculated_indices
            logger.debug(f"missing_results: {missing_results_indices}")

            missing_samples = [(index, x[index]) for index in missing_results_indices]
            recalculated_results = pool.map(indexed_log_likelihood_fun, missing_samples)
            valid_recalculated_results = [
                r
                for r in recalculated_results
                if (r is not None) and (r[0] is not None)
            ]

            valid_unordered_results = (
                valid_unordered_results + valid_recalculated_results
            )

            i = i + 1
            if i > max_retries:
                raise Exception("maximum retries reached")
        else:
            return [val[1] for val in sorted(valid_unordered_results)]

    # initialization of variables
    i = 0  # number of conditional level
    lam = 0.6  # initial scaling parameter \in (0,1)
    samplesU = {"seeds": list(), "total": list()}
    samplesX = list()
    #
    geval = np.empty(N)  # space for the LSF evaluations
    leval = np.empty(N)  # space for the LSF evaluations
    h = np.empty(max_it)  # space for the intermediate leveles
    prob = np.empty(max_it)  # space for the failure probability at each level

    # aBUS-SuS procedure
    # initial MCS step
    with TimerContext("Done inital evaluation of log-likelihood function"):
        logger.info("Evaluating log-likelihood function ...")
        u_j = np.random.normal(size=(n, N))  # N samples from the prior distribution
        leval = log_L_fun([(u_j[:, i].reshape(-1, 1)) for i in range(N)])
        leval = np.array(leval)
        logl_hat = max(leval)  # =-log(c) (Ref.1 Alg.5 Part.3)
        logger.info(f"Initial maximum log-likelihood: {logl_hat}")
        if logl_hat > REALMAX:
            logger.warning(
                f"Numerically unstable: maximum Likelihood ({logl_hat:.4g}) > {MAX_LOG:.4g}"
            )

    # SuS stage
    h[i] = np.inf
    try:
        while h[i] > 0:
            with TimerContext(f"SuS stage {i}"):
                # increase counter
                i += 1

                # compute the limit state function (Ref.1 Eq.12)
                geval = h_LSF(u_j[-1, :], logl_hat, leval)  # evaluate LSF (Ref.1 Eq.12)

                # sort values in ascending order
                idx = np.argsort(geval)
                # gsort[j,:] = geval[idx]

                # order the samples according to idx
                u_j_sort = u_j[:, idx]
                samplesU["total"].append(u_j_sort.T)  # store the ordered samples

                # intermediate level
                h[i] = np.percentile(
                    geval, p0 * 100, method="midpoint"
                )  ## BUG: if nan in geval, this doesn't work
                logger.debug(f"h[{i}]: {h[i]}")

                # number of failure points in the next level
                nF = int(sum(geval <= max(h[i], 0)))
                logger.debug(f"nF: {nF}")

                # assign conditional probability to the level
                if h[i] < 0:
                    h[i] = 0
                    prob[i - 1] = nF / N
                else:
                    prob[i - 1] = p0
                logger.debug(f"prob[{i - 1}]: {prob[i - 1]}")

                logger.info(f"Threshold level {i} = {h[i]:.4g}")

                # select seeds and randomize the ordering (to avoid bias)
                seeds = u_j_sort[:, :nF]
                idx_rnd = np.random.permutation(nF)
                rnd_seeds = seeds[:, idx_rnd]  # non-ordered seeds
                samplesU["seeds"].append(seeds.T)  # store ordered seeds

                # sampling process using adaptive conditional sampling
                if isinstance(distr, ERANataf) or isinstance(distr, ERARosen):
                    u_j, leval, lam, sigma, accrate = aCS_aBUS(
                        N,
                        lam,
                        h[i],
                        rnd_seeds,
                        indexed_log_likelihood_fun,
                        logl_hat,
                        h_LSF,
                        partial(u2x_transformation, distr),
                        pool,
                        opc=opc,
                    )
                elif isinstance(distr[0], ERADist):
                    u_j, leval, lam, sigma, accrate = aCS_aBUS(
                        N,
                        lam,
                        h[i],
                        rnd_seeds,
                        indexed_log_likelihood_fun,
                        logl_hat,
                        h_LSF,
                        partial(u2x_distr, distr[0]),
                        pool,
                        opc=opc,
                    )
                else:
                    raise ValueError(f"Unknown type of distribution: {type(distr)}.")
                logger.info(f"*aCS lambda = {lam:.4g}")
                logger.info(f"*aCS sigma = {sigma[0]}")
                logger.info(f"*aCS accrate = {accrate:.2g}")

                # update the value of the scaling constant (Ref.1 Alg.5 Part.4d)
                max_leval = max(leval)
                l_new = max(logl_hat, max_leval)
                logger.debug(f"logl_hat: {logl_hat}")
                logger.debug(f"max(leval): {max_leval}")
                logger.debug(f"logl_new: {l_new}")
                h[i] = h[i] - logl_hat + l_new
                logl_hat = l_new
                logger.info(f" Modified threshold level {i} = {h[i]:.4g}")
                if logl_hat > REALMAX:
                    logger.warning(
                        f"Numerically unstable: maximum Likelihood ({logl_hat:.4g}) > {MAX_LOG:.4g}"
                    )

                if decrease_dependency:
                    # decrease the dependence of the samples (Ref.1 Alg.5 Part.4e)
                    higher_bound = np.min(
                        [
                            np.ones(N),
                            np.nan_to_num(np.exp(leval - logl_hat + h[i]), nan=np.inf),
                        ],
                        axis=0,
                    )
                    logger.debug(f"higher_bound: {array_to_string(higher_bound)}")
                    p = np.random.uniform(
                        low=np.zeros(N),
                        high=higher_bound,
                    )
                    u_j[-1, :] = sp.stats.norm.ppf(p)  # to the standard space

    except Exception as err:
        # number of intermediate levels
        m = i

        # store final posterior samples
        samplesU["total"].append(u_j.T)  # store final failure samples (non-ordered)

        # delete unnecesary data
        if m < max_it:
            prob = prob[:m]
            h = h[:m]

        # acceptance probability and evidence (Ref.1 Alg.5 Part.6and7)
        log_p_acc = np.sum(np.log(prob))
        c = 1 / np.exp(logl_hat)  # l = -log(c) = 1/max(likelihood)
        logcE = log_p_acc + logl_hat  # exp(l) = max(likelihood)

        # transform the samples to the physical (original) space

        for i in range(m + 1):
            pp = sp.stats.norm.cdf(samplesU["total"][i][:, -1])
            try:
                samplesX.append(
                    np.concatenate(
                        (std_to_physical(samplesU["total"][i][:, :-1]), pp.reshape(-1, 1)),
                        axis=1,
                    )
                )
            except Exception as err2:
                # import IPython
                # IPython.embed()
                logger.error(
                    f"Error transforming samples to physical space: {err2}\nThis error is ignored, since another one occured prior"
                )
                logger.error(f"samplesU['total'][i].shape: {samplesU['total'][i].shape}")
                logger.error(f"samplesU['total'][i]: {samplesU['total'][i]}")
                logger.error(f"pp: {pp}")
                continue

        data = dict(samplesX=samplesX, logcE=logcE, c=c)
        data["lambda"] = lam
        raise ErrorWithData(data, *err.args)
    # number of intermediate levels
    m = i

    # store final posterior samples
    samplesU["total"].append(u_j.T)  # store final failure samples (non-ordered)

    # delete unnecesary data
    if m < max_it:
        prob = prob[:m]
        h = h[: m + 1]

    # acceptance probability and evidence (Ref.1 Alg.5 Part.6and7)
    log_p_acc = np.sum(np.log(prob))

    c = 1 / np.exp(logl_hat)  # l = -log(c) = 1/max(likelihood)
    logcE = log_p_acc + logl_hat  # exp(l) = max(likelihood)

    logger.debug(f"prob: {(array_to_string(prob))}")
    logger.debug(f"p_acc: {np.exp(log_p_acc)}")
    logger.debug(f"logl_hat: {logl_hat}")
    logger.debug(f"c: {c}")
    logger.debug(f"logcE: {logcE}")

    # import IPython
    # IPython.embed()
    # transform the samples to the physical (original) space
    for i in range(m + 1):
        pp = sp.stats.norm.cdf(samplesU["total"][i][:, -1])
        samplesX.append(
            np.concatenate(
                (std_to_physical(samplesU["total"][i][:, :-1]), pp.reshape(-1, 1)),
                axis=1,
            )
        )

    return h, samplesU, samplesX, logcE, c, sigma, lam
