import copy
import json
import itertools
from pathlib import Path

import numpy as np

from mvn.data import (
    generate_random_intercept,
    generate_doubly_correlated_homoscedastic,
    generate_doubly_correlated_heteroscedastic,
    generate_doubly_correlated_heteroscedastic_mixed,
)
from mvn.model import iterative_estimation


results_template = {
    "estimates_mvn": [],
    "estimates_lmm": [],
    "estimates_gee": [],
    "seed": [],
}


def fit_mvn(Y, covariates, config):
    pass


def fit_base_lmm(df, config):
    pass


def fit_vc_lmm(df, config):
    pass


def fit_gee(df, config):
    pass


def convert_to_df(Y, covariates):
    pass


def generate_data(exp_name, config):
    pass


def generate_output_path(exp_name, config):
    if exp_name == "random-intercept":
        output_path = Path(
            f"eval/benchmarks/{exp_name}_{config["m"]}_{config["n"]}_{config["p"]}_{config["col_rho"]}_{config["row_rho"]}_{config["theta"]}.json"
        )
    elif exp_name == "correlated-homoscedastic":
        output_path = Path(
            f"eval/benchmarks/{exp_name}_{config["m"]}_{config["n"]}_{config["p"]}_{config["col_rho"]}_{config["row_rho"]}_{config["col_corr"]}_{config["col_var"]}_{config["row_var"]}.json"
        )


def main():
    n_replicates = 100
    with open("experiments/benchmarks/config.json", "r") as file:
        full_experiment_config = json.load(file)

    for exp_name in full_experiment_config:
        list_values = [
            full_experiment_config[exp_name][key]
            for key in full_experiment_config[exp_name].keys()
        ]
        experiments = [
            dict(zip(full_experiment_config[exp_name].keys(), values))
            for values in itertools.product(*list_values)
        ]

        for exp in enumerate(experiments):
            results = copy.deepcopy(results_template)
            output_path = generate_output_path(exp_name, exp)
            if output_path.exists():
                continue
            for r in range(n_replicates):
                exp["seed"] = r

                Y, covariates = generate_data(exp_name, exp)
                df = convert_to_df(Y, covariates)
                beta_mvn = fit_mvn(Y, covariates, exp).detach().numpy().tolist()
                beta_base_lmm = fit_base_lmm(df, exp).tolist()
                beta_vc_lmm = fit_vc_lmm(df, exp).tolist()
                beta_gee = fit_gee(df, exp).tolist()
                results["estimates_mvn"].append(beta_mvn)
                results["estimates_lmm"].append(beta_base_lmm)
                results["estimates_gee"].append(beta_gee)
                results["seed"].append(r)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as fp:
                json.dump(results, fp, indent=4)


if __name__ == "__main__":
    main()
