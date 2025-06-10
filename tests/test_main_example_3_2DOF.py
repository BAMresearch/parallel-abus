"""
---------------------------------------------------------------------------
Example 3: parameter identification two-DOF shear building
---------------------------------------------------------------------------
Created by:
Matthias Willer
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
2."Bayesian updating with structural reliability methods"
   Daniel Straub & Iason Papaioannou.
   Journal of Engineering Mechanics 141.3 (2015) 1-13.
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
import scipy as sp
from parallel_abus.aBUS_SuS import aBUS_SuS, aBUS_SuS_parallel
from parallel_abus.aBUS_SuS.utils import TimerContext
from parallel_abus.ERADistNataf import ERADist, ERANataf

from . import shear_building_2DOF

plt.close("all")

# =================================================================
# shear building data
m1 = 16.5e3  # mass 1st story [kg]
m2 = 16.1e3  # mass 2nd story [kg]
kn = 29.7e6  # nominal values for the interstory stiffnesses [N/m]

# prior PDF for X1 and X2 (product of lognormals)
mod_log_X1 = 1.3  # mode of the lognormal 1
std_log_X1 = 1.0  # std of the lognormal 1
mod_log_X2 = 0.8  # mode of the lognormal 2
std_log_X2 = 1.0  # std of the lognormal 2


def var_fun(mu):
    return std_log_X1**2 - (np.exp(mu - np.log(mod_log_X1)) - 1) * np.exp(
        2 * mu + (mu - np.log(mod_log_X1))
    )


def var_X2(mu):
    return std_log_X2**2 - (np.exp(mu - np.log(mod_log_X2)) - 1) * np.exp(
        2 * mu + (mu - np.log(mod_log_X2))
    )


mu_X1 = sp.optimize.fsolve(var_fun, 1)  # mean of the associated Gaussian
std_X1 = np.sqrt(mu_X1 - np.log(mod_log_X1))  # std of the associated Gaussian

mu_X2 = sp.optimize.fsolve(var_X2, 0)  # mean of the associated Gaussian
std_X2 = np.sqrt(mu_X2 - np.log(mod_log_X2))  # std of the associated Gaussian


# =================================================================
# definition of the random variables
n = 2  # number of random variables (dimensions)
dist_x1 = ERADist("lognormal", "PAR", [mu_X1, std_X1])
dist_x2 = ERADist("lognormal", "PAR", [mu_X2, std_X2])

# distributions
dist_X = [dist_x1, dist_x2]

# correlation matrix
R = np.eye(n)  # independent case

# object with distribution information
prior_pdf = ERANataf(dist_X, R)

# =================================================================
# likelihood function
lam = np.array([1, 1])  # means of the prediction error
i = 9  # simulation level
var_eps = 0.5 ** (i - 1)  # variance of the prediction error
f_tilde = np.array([3.13, 9.83])  # measured eigenfrequencies [Hz]


def shear_building_model(x):
    return shear_building_2DOF.shear_building_2DOF(m1, m2, kn * x[0], kn * x[1])


def modal_measure_of_fit(x):
    return np.sum((lam**2) * (((shear_building_model(x) ** 2) / f_tilde**2) - 1) ** 2)


def likelihood_function(x):
    return np.exp(-modal_measure_of_fit(x) / (2 * var_eps))


def log_likelihood_function(x):
    return -modal_measure_of_fit(x) / (2 * var_eps)


def indexed_log_likelihood(indexed_theta):
    return indexed_theta[0], log_likelihood_function(indexed_theta[1])


# =================================================================
# aBUS-SuS
N = int(3e3)  # number of samples per level
p0 = 0.1  # probability of each subset


def main(n_processes: Optional[int]):
    print("\naBUS with SUBSET SIMULATION: \n")

    if n_processes == 1:
        with TimerContext("aBUS-SuS sequential"):
            h, samplesU, samplesX, logcE, c, sigma, _ = aBUS_SuS(
                N, p0, log_likelihood_function, prior_pdf
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
    u1p, u2p, u0p = list(), list(), list()
    x1p, x2p, pp = list(), list(), list()
    #
    for i in range(nsub):
        # samples in standard
        u1p.append(samplesU["total"][i][:, 0])
        u2p.append(samplesU["total"][i][:, 1])
        u0p.append(samplesU["total"][i][:, 2])
        # samples in physical
        x1p.append(samplesX[i][:, 0])
        x2p.append(samplesX[i][:, 1])
        pp.append(samplesX[i][:, 2])

    # =================================================================
    # reference solutions
    mu_exact = 1.12  # for x_1
    sigma_exact = 0.66  # for x_1
    cE_exact = 1.52e-3

    # show results
    print("\nExact model evidence =", cE_exact)
    print("Model evidence aBUS-SuS =", np.exp(logcE), "\n")
    print("Exact posterior mean x_1 =", mu_exact)
    print("Mean value of x_1 =", np.mean(x1p[-1]), "\n")
    print("Exact posterior std x_1 =", sigma_exact)
    print("Std of x_1 =", np.std(x1p[-1]), "\n")

    # =================================================================
    # Plots
    # Options for font-family and font-size
    plt.rc("font", size=12)
    plt.rc("axes", titlesize=20)  # fontsize of the axes title
    plt.rc("axes", labelsize=18)  # fontsize of the x and y labels
    plt.rc("figure", titlesize=20)  # fontsize of the figure title

    # plot samples in standard normal space
    plt.figure()
    plt.suptitle("Standard space")
    for i in range(nsub):
        plt.subplot(2, 3, i + 1)
        plt.plot(u1p[i], u2p[i], "r.")
        plt.xlabel("$u_1$")
        plt.ylabel("$u_2$")
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show()

    # plot samples in original space
    plt.figure()
    plt.suptitle("Original space")
    for i in range(nsub):
        plt.subplot(2, 3, i + 1)
        plt.plot(x1p[i], x2p[i], "b.")
        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        num_processes = int(sys.argv[1])
    else:
        num_processes = None

    main(num_processes)
