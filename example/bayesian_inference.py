"""Run Bayesian inference of model classes on data."""

import logging
import multiprocessing
import os
import pathlib
import time

import pandas as pd
from parallel_abus import aBUS_SuS, aBUS_SuS_parallel

from inference import ModelClass
from inference import prior_as_nataf, abus_to_posterior, posterior_to_csv
from probabilistic_model_classes import M1_Timoshenko_Beam


def main(
    save_path: pathlib.Path = pathlib.Path("./"),
    abus_n: int = 3000,
    abus_p0: float = 0.1,
    run_parallel: bool = True,
):
    """Run Bayesian inference of model classe M1 on static data."""

    # 1. choose data
    damage_types = ["ungerissen"]
    
    THIS_MODULE_FILE = pathlib.Path(__file__)
    datafile = THIS_MODULE_FILE.parent / "measurements_undamaged_beam.csv"
    df = pd.read_csv(datafile)

    model_class: ModelClass = M1_Timoshenko_Beam

    model_instance = model_class(df)  # type: ignore

    tic = time.perf_counter()
    try:
        if run_parallel:
            n_processes = int(os.cpu_count() / 2)
            logging.info(
                f"Start parallely running {model_class.__name__} on static data with {n_processes} processes."
            )
            with multiprocessing.Pool(processes=n_processes) as p:
                _, _, samplesX, logcE, c, _, lambda_par = aBUS_SuS_parallel(
                    abus_n,
                    abus_p0,
                    model_instance.indexed_log_likelihood,
                    prior_as_nataf(model_instance.prior),
                    p,
                )
        else:
            logging.info(f"Start running {model_class.__name__} on static data.")
            _, _, samplesX, logcE, c, _, lambda_par = aBUS_SuS(
                abus_n,
                abus_p0,
                model_instance.log_likelihood,
                prior_as_nataf(model_instance.prior),
            )
        posterior = abus_to_posterior(model_class, samplesX, logcE, c, lambda_par)

        save_filename = f"{model_class.__name__}_static_{'_'.join(damage_types)}.csv"
        posterior_to_csv(posterior, save_path / save_filename)
        logging.info(f"Saved full results in {save_filename}")

    except Exception as err:
        logging.info(f"unknown exception: {err}")
        raise RuntimeError("Problem") from err

    elapsed_time = time.perf_counter() - tic
    logging.info(f"Done running {model_class.__name__} on static data.")
    hours = int(elapsed_time // 3600)
    mins = int((elapsed_time - hours * 3600) // 60)
    logging.info(f"Elapsed time: {hours}h: {mins}min \n ----")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(run_parallel=True)
