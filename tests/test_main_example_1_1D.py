"""
---------------------------------------------------------------------------
Example 1: 1D posterior
---------------------------------------------------------------------------
Created by:
Fong-Lin Wu
Felipe Uribe
Luca Sardi

Engineering Risk Analysis Group
Technische Universitat Munchen
www.bgu.tum.de/era
Contact: Antonios Kamariotis (antonis.kamariotis@tum.de)
---------------------------------------------------------------------------
Last Version 2021-03
* Adaptation to new ERANataf class
---------------------------------------------------------------------------
References:
1."Bayesian inference with subset simulation: strategies and improvements"
   Wolfgang Betz et al.
   Computer Methods in Applied Mechanics and Engineering 331 (2018) 72-93.
---------------------------------------------------------------------------
"""

# =================================================================
# import libraries
import os
import sys
from multiprocessing import Pool
from typing import Optional

import matplotlib.pylab as plt
import numpy as np
from parallel_abus.aBUS_SuS import aBUS_SuS, aBUS_SuS_parallel
from parallel_abus.aBUS_SuS.utils import TimerContext

# Make sure ERADist, ERANataf classes are in the path
# https://www.bgu.tum.de/era/software/eradist/
from parallel_abus.ERADistNataf import ERADist
from scipy.stats import multivariate_normal as mvn

plt.close("all")

# =================================================================
# define the prior
d = 1  # number of dimensions (uncertain parameters)
prior_pdf_1 = ERADist("normal", "PAR", [0, 1])
prior_pdf = [prior_pdf_1]

# correlation matrix
# R = np.eye(d)   # independent case

# object with distribution information
# prior_pdf = ERANataf(prior_pdf,R)

# =================================================================
# define likelihood
mu_obs = 5
sigma_obs = 0.2
#
realmin = np.finfo(np.double).tiny  # to prevent log(0)


def likelihood(theta):
    return mvn.pdf(theta, mu_obs, sigma_obs**2)


def log_likelihood(theta):
    return np.log(likelihood(theta) + realmin)


def indexed_log_likelihood(indexed_theta):
    return indexed_theta[0], log_likelihood(indexed_theta[1])


# =================================================================
# aBUS-SuS step
N = int(3e3)  # number of samples per level
p0 = 0.1  # probability of each subset


def main(n_processes: Optional[int]):
    print("\naBUS with SUBSET SIMULATION: \n")

    if n_processes == 1:
        with TimerContext("aBUS-SuS sequential"):
            h, samplesU, samplesX, logcE, c, sigma, _ = aBUS_SuS(
                N, p0, log_likelihood, prior_pdf
            )
    else:
        if n_processes is None:
            n_processes = os.cpu_count()
        if not isinstance(n_processes, int):
            raise TypeError(
                "n_processes must be an integer or None, but is {}".format(
                    type(n_processes)
                )
            )
        with TimerContext(f"aBUS-SuS parallel with {n_processes} processes"):
            with Pool(n_processes) as p:
                h, samplesU, samplesX, logcE, c, sigma, _ = aBUS_SuS_parallel(
                    N, p0, indexed_log_likelihood, prior_pdf, p
                )

    # =================================================================
    # organize samples and show results
    nsub = len(h.flatten())  # number of levels
    u1p, u0p = list(), list()
    x1p, pp = list(), list()
    #
    for i in range(nsub):
        # samples in standard
        u1p.append(samplesU["total"][i][:, 0])
        u0p.append(samplesU["total"][i][:, 1])
        # samples in physical
        x1p.append(samplesX[i][:, 0])
        pp.append(samplesX[i][:, 1])

    # show results
    # =================================================================
    # reference and BUS solutions
    mu_exact = 4.81
    sigma_exact = 0.196
    cE_exact = 2.36e-6
    print("\nExact model evidence = ", cE_exact)
    print("Model evidence aBUS-SuS = ", np.exp(logcE))
    print("\nExact posterior mean = ", mu_exact)
    print("Mean value of samples = ", np.mean(x1p[-1]))
    print("\nExact posterior std = ", sigma_exact)
    print("Std of samples = ", np.std(x1p[-1]), "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        num_processes = int(sys.argv[1])
    else:
        num_processes = None

    main(num_processes)
