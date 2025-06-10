"""adaptive conditional sampling algorithm for aBUS.

---------------------------------------------------------------------------
Created by:
Fong-Lin Wu (fonglin.wu@tum.de)
Matthias Willer (matthias.willer@tum.de)
Felipe Uribe (felipe.uribe@tum.de)
Iason Papaioannou (iason.papaioannou@tum.de)
Engineering Risk Analysis Group
Technische Universitat Munchen
www.era.bgu.tum.de
---------------------------------------------------------------------------
Version 2018-05
---------------------------------------------------------------------------
Input:
* N         : number of samples to be generated
* lam       : scaling parameter lambda
* h         : current intermediate level
* u_j       : seeds used to generate the new samples
* log_L_fun : log-likelihood function
* l         : =-log(c) ~ scaling constant of BUS for the current level
* gl        : limit state function in the standard space
---------------------------------------------------------------------------
Output:
* u_jp1      : next level samples
* leval      : log-likelihood function of the new samples
* new_lambda : next scaling parameter lambda
* sigma      : spread of the proposal
* accrate    : acceptance rate of the samples
---------------------------------------------------------------------------
NOTES
* The way the initial standard deviation is computed can be changed in line 69.
By default we use option 'a' (it is equal to one).
In option 'b', it is computed from the seeds.
* The final accrate might differ from the target 0.44 when no adaptation is
performed. Since at the last level almost all seeds are already in the failure
domain, only a few are selected to complete the required N samples. The final
accrate is computed for those few samples
---------------------------------------------------------------------------
Based on:
1."MCMC algorithms for subset simulation"
   Papaioannou et al.
   Probabilistic Engineering Mechanics 41 (2015) 83-103.
---------------------------------------------------------------------------
"""

import numpy as np
import warnings

# import platform
# import multiprocessing
from multiprocessing.pool import Pool  # for typing

from typing import Any, Callable, Iterable, Tuple, Optional

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

MAX_LINE_LENGTH = 200
PRECISION = 8

MIN_SAMPLES = 100


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


def aCS_aBUS(
    N, lambd_old, tau, theta_seeds, log_L_fun, logl_hat, h_LSF, opc: str = "b"
):
    n, Ns = theta_seeds.shape  # dimension and number of seeds

    # number of samples per chain
    Nchain = np.ones(Ns, dtype=int) * int(np.floor(N / Ns))
    Nchain[: np.mod(N, Ns)] = Nchain[: np.mod(N, Ns)] + 1

    # initialization
    theta_chain = np.zeros((n, N))  # generated samples
    leval = np.zeros(N)  # store lsf evaluations
    acc = np.zeros(N, dtype=int)  # store acceptance

    # initialization
    Na = int(
        np.ceil(100 * Ns / N)
    )  # number of chains after which the proposal is adapted
    mu_acc = np.zeros(int(np.floor(Ns / Na) + 1))  # store acceptance
    hat_acc = np.zeros(int(np.floor(Ns / Na)))  # average acceptance rate of the chains
    lambd = np.zeros(int(np.floor(Ns / Na) + 1))  # scaling parameter \in (0,1)

    # 1. compute the standard deviation
    # opc = "a"
    if opc == "a":  # 1a. sigma = ones(n,1)
        sigma_0 = np.ones(n)
    elif opc == "b":  # 1b. sigma = sigma_hat (sample standard deviations)
        sigma_0 = np.std(theta_seeds, axis=1)
    else:
        raise RuntimeError("Choose a or b")

    # 2. iteration
    star_a = 0.44  # optimal acceptance rate
    lambd[0] = lambd_old  # initial scaling parameter \in (0,1)

    # a. compute correlation parameter
    i = 0  # index for adaptation of lambda
    sigma = np.minimum(lambd[i] * sigma_0, np.ones(n))  # Ref. 1 Eq. 23
    rho = np.sqrt(1 - sigma**2)  # Ref. 1 Eq. 24
    mu_acc[i] = 0

    # b. apply conditional sampling
    for k in range(1, Ns + 1):
        idx = sum(Nchain[: k - 1])  # beginning of each chain total index

        # initial chain values
        theta_chain[:, idx] = theta_seeds[:, k - 1]

        # initial log-like evaluation
        # leval[idx] = log_L_fun(theta_chain[:,idx])
        leval[idx] = log_L_fun((theta_chain[:, idx],))[0]

        for t in range(1, Nchain[k - 1]):
            # current state
            theta_t = theta_chain[:, idx + t - 1]

            # generate candidate sample
            v_star = np.random.normal(loc=rho * theta_t, scale=sigma)

            # evaluate loglikelihood function
            # log_l_star = log_L_fun(v_star)
            log_l_star = log_L_fun((v_star,))[0]

            # evaluate limit state function
            heval = h_LSF(
                v_star[-1].reshape(-1), logl_hat, log_l_star
            )  # evaluate limit state function

            # accept or reject sample
            if heval <= tau:
                theta_chain[:, idx + t] = (
                    v_star  # accept the candidate in observation region
                )
                leval[idx + t] = log_l_star  # store the loglikelihood evaluation
                acc[idx + t] = 1  # note the acceptance
            else:
                theta_chain[:, idx + t] = (
                    theta_t  # reject the candidate and use the same state
                )
                leval[idx + t] = leval[
                    idx + t - 1
                ]  # store the loglikelihood evaluation
                acc[idx + t] = 0  # note the rejection

        # average of the accepted samples for each seed 'mu_acc'
        # here the warning "Mean of empty slice" is not an issue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mu_acc[i] = mu_acc[i] + np.minimum(
                1, np.mean(acc[idx + 1 : idx + Nchain[k - 1]])
            )

        if np.mod(k, Na) == 0:
            if Nchain[k - 1] > 1:
                # c. evaluate average acceptance rate
                hat_acc[i] = mu_acc[i] / Na  # Ref. 1 Eq. 25

                # d. compute new scaling parameter
                zeta = 1 / np.sqrt(
                    i + 1
                )  # ensures that the variation of lambda(i) vanishes
                lambd[i + 1] = np.exp(
                    np.log(lambd[i]) + zeta * (hat_acc[i] - star_a)
                )  # Ref. 1 Eq. 26

                # update parameters
                sigma = np.minimum(lambd[i + 1] * sigma_0, np.ones(n))  # Ref. 1 Eq. 23
                rho = np.sqrt(1 - sigma**2)  # Ref. 1 Eq. 24

                # update counter
                i += 1

    # next level lambda
    new_lambda = lambd[i]

    # compute mean acceptance rate of all chains
    accrate = sum(acc) / (N - Ns)

    return theta_chain, leval, new_lambda, sigma, accrate


def metropolis_hastings(
    mapdl_instance,
    seed: np.ndarray,
    n_samples: int,
    initial_leval: float,  # not needed
    initial_acc: int,  # not needed
    log_L_fun: Callable,
    h_LSF: Callable,
    rho: float,
    sigma: float,
    logl_hat: float,
    tau: float,
    u2x: Callable,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """For each seed generate a markov-chain with likelihood and LSF.

    Args:
        seed (np.ndarray): [description]
        n_samples (int): [description]
        initial_leval (float): [description]
        initial_acc (int): [description]
        log_L_fun (Callable): [description]
        h_LSF (Callable): [description]
        rho (float): [description]
        sigma (float): [description]
        logl_hat (float): [description]
        tau (float): [description]
        u2x (Callable): [description]

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: markov-chain samples, likelihood-evaluations, acceptance
    """
    # 1) determine size of problem
    n_dimensions = seed.shape[0]
    markov_chain = np.empty((n_dimensions, n_samples))
    leval = np.empty((n_samples))
    acc = np.zeros((n_samples), dtype=np.int32)

    # 2) initial step

    if mapdl_instance:
        log_l_init = log_L_fun(mapdl_instance, None, u2x(seed[:-1]).flatten())[1]
    else:
        log_l_init = log_L_fun((None, u2x(seed[:-1]).flatten()))[1]

    if log_l_init is None:
        return None
    markov_chain[:, 0] = seed
    leval[0] = log_l_init
    acc[0] = 0

    # 3) following steps:
    for i_sample in range(1, n_samples):
        # current state
        theta_t = markov_chain[:, i_sample - 1]

        # generate candidate sample
        v_star = np.random.normal(loc=rho * theta_t, scale=sigma)

        # evaluate loglikelihood function
        # TODO: discern whether log_L_fun has to be called with mapdl arguments or not
        if mapdl_instance:
            log_l_star = log_L_fun(mapdl_instance, None, u2x(v_star[:-1]).flatten())[1]
        else:
            log_l_star = log_L_fun((None, u2x(v_star[:-1]).flatten()))[1]

        if log_l_star is None:
            return None
        # evaluate limit state function
        try:
            heval = h_LSF(
                v_star[-1].reshape(-1), logl_hat, log_l_star
            )  # evaluate limit state function
        except TypeError as err:
            logger.error(f"Error in h_LSF:\n{err}")
            return None

        # accept or reject sample
        if heval <= tau:
            # theta_chain[
            #     :, idx + t
            # ] = v_star  # accept the candidate in observation region
            # leval[idx + t] = log_l_star  # store the loglikelihood evaluation
            # acc[idx + t] = 1  # note the acceptance
            markov_chain[:, i_sample] = (
                v_star  # accept the candidate in observation region
            )
            leval[i_sample] = log_l_star  # store the loglikelihood evaluation
            acc[i_sample] = 1  # note the acceptance
        else:
            # theta_chain[
            #     :, idx + t
            # ] = theta_t  # reject the candidate and use the same state
            # leval[idx + t] = leval[idx + t - 1]  # store the loglikelihood evaluation
            # acc[idx + t] = 0  # note the rejection
            markov_chain[:, i_sample] = (
                theta_t  # reject the candidate and use the same state
            )

            leval[i_sample] = leval[i_sample - 1]  # store the loglikelihood evaluation
            acc[i_sample] = 0  # note the rejection
    return markov_chain, leval, acc


def numbered_mh(
    numberedargumenttuple: Tuple[
        int,
        Tuple[
            np.ndarray,
            int,
            float,
            int,
            Callable,
            Callable,
            float,
            float,
            float,
            float,
            Callable,
        ],
    ],
) -> Tuple[int, Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Wrap metropolis_hastings to be compliant with pool.map method."""
    number = numberedargumenttuple[0]
    argument = numberedargumenttuple[1]
    seed = argument[0]
    n_samples = argument[1]
    initial_leval = argument[2]
    initial_acc = argument[3]
    log_L_fun = argument[4]
    h_LSF = argument[5]
    rho = argument[6]
    sigma = argument[7]
    logl_hat = argument[8]
    tau = argument[9]
    u2x = argument[10]

    return (
        number,
        metropolis_hastings(
            None,
            seed=seed,
            n_samples=n_samples,
            initial_leval=initial_leval,
            initial_acc=initial_acc,
            log_L_fun=log_L_fun,
            h_LSF=h_LSF,
            rho=rho,
            sigma=sigma,
            logl_hat=logl_hat,
            tau=tau,
            u2x=u2x,
        ),
    )


def numbered_mh_mapdl(
    mapdl,
    number: int,
    argument: Tuple[
        np.ndarray,
        int,
        float,
        int,
        Callable,
        Callable,
        float,
        float,
        float,
        float,
        Callable,
    ],
) -> Tuple[int, Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Wrap metropolis_hastings to be compliant with pool.map method."""
    # number = numberedargumenttuple[0]
    # argument = numberedargumenttuple[1]
    seed = argument[0]
    n_samples = argument[1]
    initial_leval = argument[2]
    initial_acc = argument[3]
    log_L_fun = argument[4]
    h_LSF = argument[5]
    rho = argument[6]
    sigma = argument[7]
    logl_hat = argument[8]
    tau = argument[9]
    u2x = argument[10]

    return (
        number,
        metropolis_hastings(
            mapdl_instance=mapdl,
            seed=seed,
            n_samples=n_samples,
            initial_leval=initial_leval,
            initial_acc=initial_acc,
            log_L_fun=log_L_fun,
            h_LSF=h_LSF,
            rho=rho,
            sigma=sigma,
            logl_hat=logl_hat,
            tau=tau,
            u2x=u2x,
        ),
    )


def pool_map_dummy(
    function: Callable[[Tuple[int, Tuple]], Any],
    numberedargumenttuples: Iterable[Tuple[int, Tuple]],
):
    """Emulate the behavior of pool.map."""
    return [function(numbered_arg) for numbered_arg in numberedargumenttuples]


def failsafe_mh_pool(mh_arguments, pool):
    """Pool evaluations of metropolis-hastings chains."""
    n_chains = len(mh_arguments)
    # check type of pool!
    if isinstance(pool, Pool):
        unordered_results = pool.map(
            numbered_mh, list(zip(range(n_chains), mh_arguments))
        )
    else:
        raise ValueError(f"Unknown type of pool: {type(pool)}")
    valid_unordered_results = [
        r
        for r in unordered_results
        if (r is not None) and (r[0] is not None) and (r[1] is not None)
    ]
    max_retries = 20
    i = 0
    while len(valid_unordered_results) != n_chains:
        calculated_indices = set([val[0] for val in valid_unordered_results])
        all_indices = set(range(n_chains))
        missing_results_indices = all_indices - calculated_indices
        logger.debug(f"missing_results: {missing_results_indices}")
        missing_chains = [
            (index, mh_arguments[index]) for index in missing_results_indices
        ]
        if isinstance(pool, Pool):
            recalculated_results = pool.map(numbered_mh, missing_chains)
        else:
            raise ValueError(f"Unknown type of pool: {type(pool)}")

        valid_recalculated_results = [
            r
            for r in recalculated_results
            if (r is not None) and (r[0] is not None) and (r[1] is not None)
        ]

        valid_unordered_results = valid_unordered_results + valid_recalculated_results

        i = i + 1
        if i > max_retries:
            raise Exception("maximum retries reached")
    else:
        valid_ordered_results = [val[1] for val in sorted(valid_unordered_results)]
        for res in valid_ordered_results:
            assert not np.any(np.isnan(res[1])), "nan in likelihood evaluations"

        return valid_ordered_results


def aCS_aBUS_batches(
    n_samples: int,
    lambd_old: float,
    tau: float,
    theta_seeds: np.ndarray,
    log_L_fun: Callable,
    logl_hat: float,
    h_LSF: Callable,
    u2x: Callable,
    pool: Pool,
    opc: str = "b",
    n_chains: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, float]:
    """
    Perform adaptive conditional sampling for Bayesian Updating Simulation (BUS)
    in batches. This function is designed to run multiple sampling processes in parallel,
    enhancing computational efficiency and robustness.

    Parameters:
    n_samples (int): Total number of samples to be generated.
    lambd_old (float): Previous scaling parameter lambda.
    tau (float): Threshold parameter for the limit state function.
    theta_seeds (np.ndarray): Array of seed values for generating new samples.
    log_L_fun (Callable): Log-likelihood function.
    logl_hat (float): Log-likelihood estimate.
    h_LSF (Callable): Limit state function in the standard space.
    u2x (Callable): Transformation function from standard space to original space.
    pool (Pool): Pool object for parallel execution.
    opc (str, optional): Option for computing initial standard deviation ('a' or 'b'). Defaults to "b".
    n_chains (int, optional): Number of parallel chains for sampling. Defaults to None.

    Returns:
    Tuple[np.ndarray, np.ndarray, float, np.ndarray, float]: A tuple containing:
        - theta_chain_result (np.ndarray): Generated samples array.
        - leval_result (np.ndarray): Log-likelihood evaluations of the samples.
        - new_lambda (float): Updated scaling parameter lambda.
        - sigma (np.ndarray): Spread of the proposal distribution.
        - accrate (float): Acceptance rate of the samples.

    Raises:
    ValueError: If the pool type is unknown or if an invalid 'opc' option is provided.

    Example:
    >>> # Assuming necessary functions and a pool object are defined
    >>> result = aCS_aBUS_batches(100, 0.5, 0.1, seed_array, log_likelihood_function, 0.2, limit_state_function, u2x_transform, my_pool)
    >>> print(result)
    """
    n_dimensions, n_seeds = theta_seeds.shape  # dimension and number of seeds

    if n_chains is None:
        if isinstance(pool, Pool):
            n_processes: int = pool._processes  # type: ignore
            n_chains = n_processes
        else:
            try:
                n_processes: int = len(pool)
                n_chains = n_processes
            except Exception as err:
                raise ValueError(
                    f"Could not get size of pool {pool} \n Exception:\n {err}"
                )
                logger.debug("setting n_chains to standard value of 10")
                n_chains = 10

    # number of samples per chain
    # TODO: introduce minimum criterion to reach 100 samples per batch
    n_max_samples_per_chain = int(np.ceil(n_samples / n_seeds))

    # 0. initialization
    theta_chain = []
    leval = []
    acc = []

    seed_list = [theta_seeds[:, i] for i in range(n_seeds)]

    # mu_acc = []
    # hat_acc = []
    lambd = []
    already_evaluated_samples_per_chain = []

    # 1. compute the standard deviation
    # opc = "a"
    if opc == "a":  # 1a. sigma = ones(n,1)
        sigma_0 = np.ones(n_dimensions)
    elif opc == "b":  # 1b. sigma = sigma_hat (sample standard deviations)
        sigma_0 = np.std(theta_seeds, axis=1)  # type: ignore
    else:
        raise ValueError("Choose a or b")

    logger.debug(f"sigma_0: {array_to_string(sigma_0)}")
    sigma_0 = np.nan_to_num(sigma_0, nan=1.0)
    logger.debug(f"sigma_0_corrected: {array_to_string(sigma_0)}")
    logger.debug(f"lambd_old: {lambd_old}")

    # 2. iteration
    star_a = 0.44  # optimal acceptance rate
    lambd.append(lambd_old)  # initial scaling parameter \in (0,1)

    # 2a. compute correlation parameter
    i_adaption = 0  # index for adaptation of lambda
    sigma = np.minimum(lambd[0] * sigma_0, np.ones(n_dimensions))  # Ref. 1 Eq. 23
    rho = np.sqrt(1 - sigma**2)  # Ref. 1 Eq. 24
    logger.debug(f"initial_sigma: {array_to_string(sigma)}")
    logger.debug(f"initial_rho: {array_to_string(rho)}")

    # b. apply batchwise conditional sampling

    while len(seed_list) > 0:
        # 1. load samples from seeds and build parameter list
        # if there are less samples needed than seeds, push the seeds to the results and exit
        n_already_evaluated = int(np.sum(already_evaluated_samples_per_chain))
        n_left_to_evaluate = n_samples - n_already_evaluated
        if n_left_to_evaluate <= len(seed_list):
            for seed in seed_list:
                theta_chain.append(seed[:, np.newaxis])
                acc.append(np.ones(1, dtype=int))

            # TODO: parallelize this special case
            while len(seed_list) > 0:
                unordered_results = pool.map(
                    log_L_fun,
                    list(enumerate([u2x(seed[:-1]).flatten() for seed in seed_list])),
                )

                valid_unordered_results = [
                    r for r in unordered_results if (r[1] is not None)
                ]

                leval += [np.array(r) for i, r in valid_unordered_results]

                valid_idx = [i for i, r in valid_unordered_results]
                invalid_idx = set(range(len(seed_list))) - set(valid_idx)
                recalculate_seeds = [seed_list[i] for i in invalid_idx]
                logger.debug(f"missing_results for seeds: {recalculate_seeds}")
                seed_list = recalculate_seeds

            break

        # if there are less seeds than n_chains, limit n_chains_in_batch
        n_chains_in_batch = min(n_chains, len(seed_list))
        # if there are more to evaluate than n_chains * n_max_samples_per_chain,
        # evaluate full amount of chains and samples per chains
        if n_left_to_evaluate >= n_chains_in_batch * n_max_samples_per_chain:
            samples_per_chain = [n_max_samples_per_chain] * n_chains_in_batch
        else:
            samples_per_chain = np.ones(n_chains_in_batch, dtype=int) * int(
                np.floor(n_left_to_evaluate / n_chains_in_batch)
            )
            samples_per_chain[: np.mod(n_left_to_evaluate, n_chains_in_batch)] = (
                samples_per_chain[: np.mod(n_left_to_evaluate, n_chains_in_batch)] + 1
            )

        # n_seeds_for_chains = min(n_chains, len(seed_list))
        # TODO: correct samples per chain to prevent excess evaluation
        # samples_per_chain.append(Nchain[i_batch * Na + j_chain])
        seed_per_chain = []
        for j_chain in range(n_chains_in_batch):
            seed_per_chain.append(seed_list.pop())

        init_acc_per_chain = [None] * n_chains_in_batch  # not needed
        initial_leval_per_chain = [None] * n_chains_in_batch  # not needed
        log_L_fun_per_chain = [log_L_fun] * n_chains_in_batch
        h_LSF_per_chain = [h_LSF] * n_chains_in_batch
        rho_per_chain = [rho] * n_chains_in_batch
        sigma_per_chain = [sigma] * n_chains_in_batch
        logl_hat_per_chain = [logl_hat] * n_chains_in_batch
        tau_per_chain = [tau] * n_chains_in_batch
        u2x_per_chain = [u2x] * n_chains_in_batch

        mh_arguments = list(
            zip(
                seed_per_chain,
                samples_per_chain,
                initial_leval_per_chain,
                init_acc_per_chain,
                log_L_fun_per_chain,
                h_LSF_per_chain,
                rho_per_chain,
                sigma_per_chain,
                logl_hat_per_chain,
                tau_per_chain,
                u2x_per_chain,
            )
        )

        # 2. evaluate batch of mcmc chains with pool.map
        if isinstance(pool, Pool):
            unordered_results = pool.map(
                numbered_mh, list(zip(range(n_chains_in_batch), mh_arguments))
            )
        else:
            raise ValueError(f"Unknown type of pool: {type(pool)}")

        # 3. filter valid and invald results
        valid_unordered_results = [r for r in unordered_results if r[1] is not None]
        indices_of_valid_results = [r[0] for r in valid_unordered_results]
        indices_of_invalid_results = set(range(n_chains_in_batch)) - set(
            indices_of_valid_results
        )
        logger.debug(f"indices of valid results: {indices_of_valid_results}")
        logger.debug(f"n_chains_in_batch: {n_chains_in_batch}")
        logger.debug(f"indices of invalid results: {indices_of_invalid_results}")
        # invalid_unordered_results = set(unordered_results) - set(valid_unordered_results)

        # 4. push invalid seeds back to seed_list

        for invalid_index in indices_of_invalid_results:
            seed_list.append(seed_per_chain[invalid_index])

        # 5. store valid results
        theta_chain += [res[0] for _, res in valid_unordered_results]
        leval_batch = [res[1] for _, res in valid_unordered_results]
        if np.any(np.isnan(np.array(leval))):
            logger.warning("nan in likelihood evaluations")
        leval += leval_batch
        acc += [res[2] for _, res in valid_unordered_results]

        # 6. evaluate batch parameters (based on minimal criteria)
        # n_valid_results = len(valid_unordered_results)
        already_evaluated_samples_per_chain += [
            samples_per_chain[k_idx] for k_idx in indices_of_valid_results
        ]
        cumulative_reversed_samples = np.cumsum(
            already_evaluated_samples_per_chain[::-1]
        )
        n_chains_for_update = np.sum(cumulative_reversed_samples < MIN_SAMPLES) + 1
        # do not update if there are too little samples
        if n_chains_for_update > len(acc):
            continue
        else:
            total_samples_of_update = np.sum(
                [len(acc[-l_chain]) for l_chain in range(n_chains_for_update)]
            )
            accepted_samples_of_update = np.sum(
                [np.sum(acc[-l_chain]) for l_chain in range(n_chains_for_update)]
            )
            hat_acc = accepted_samples_of_update / total_samples_of_update
            zeta = 1 / np.sqrt(
                i_adaption + 1
            )  # ensures that the variation of lambda(i) vanishes
            logger.debug(f"hat_acc: {hat_acc:.2f}")
            lambd.append(
                np.exp(np.log(lambd[i_adaption]) + zeta * (hat_acc - star_a))
            )  # Ref. 1 Eq. 26
            sigma = np.minimum(
                lambd[i_adaption + 1] * sigma_0, np.ones(n_dimensions)
            )  # Ref. 1 Eq. 23
            rho = np.sqrt(1 - sigma**2)  # Ref. 1 Eq. 24
            logger.debug(f"sigma: {array_to_string(sigma)}")
            logger.debug(f"rho: {array_to_string(rho)}")

            # update counter
            i_adaption += 1

    # c. concatenate results

    n_chains_evaluated = len(leval)
    assert len(acc) == n_chains_evaluated
    # make sure that not more than n_samples from n_chains_evaluated are returned
    theta_chain_result = np.hstack(theta_chain)
    leval_result = np.hstack(leval)

    # next level lambda
    new_lambda = lambd[i_adaption]
    # new_lambda = lambd[-1]

    # compute mean acceptance rate of all chains
    accrate = np.sum([np.sum(a) for a in acc]) / (n_samples - n_seeds)
    logger.debug(f"total acceptance rate: {accrate:.2f}")

    return (
        theta_chain_result[:, :n_samples],
        leval_result[:n_samples],
        new_lambda,
        sigma,
        accrate,
    )
