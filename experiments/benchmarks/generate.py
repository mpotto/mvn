import copy
import json
import itertools
import warnings
from joblib import Parallel, delayed
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.tools.sm_exceptions import ConvergenceWarning, IterationLimitWarning
from tqdm import tqdm

from mvn.data import (
    generate_random_intercept,
    generate_doubly_correlated_homoscedastic,
    generate_doubly_correlated_heteroscedastic,
    generate_doubly_correlated_heteroscedastic_mixed,
)
from mvn.model import iterative_estimation
from mvn.utils import check_arrays


warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", IterationLimitWarning)


results_template = {
    "beta_true": [],
    "estimates_mvn": [],
    "estimates_base_lmm": [],
    "estimates_vc_lmm": [],
    "estimates_gee": [],
    "seed": [],
}


def generate_data(exp_name, config):
    m, n = config["m"], config["n"]
    p = config["p"]
    beta_true = np.linspace(-5, 5, p)
    generator_seed = config["seed"]
    if exp_name == "random-intercept":
        # Seed is fixed to get a fixed design matrix across simulations.
        Y, covariates = generate_random_intercept(
            m, n, beta_true, theta=config["theta"], seed=0, generator=generator_seed
        )
    elif exp_name == "correlated-homoscedastic":
        Y, covariates, _, _ = generate_doubly_correlated_homoscedastic(
            m,
            n,
            beta_true,
            col_corr=config["col_corr"],
            col_var=config["col_var"],
            row_var=config["row_var"],
            seed=0,
            generator=generator_seed
        )
    elif exp_name == "correlated-heteroscedastic":
        Y, covariates, _, _ = generate_doubly_correlated_heteroscedastic(
            m,
            n,
            beta_true,
            col_corr=config["col_corr"],
            row_corr=config["row_corr"],
            max_col_var=config["max_col_var"],
            max_row_var=config["max_row_var"],
            seed=0,
            generator=generator_seed
        )
    elif exp_name == "correlated-heteroscedastic-mixed-covariates":
        Y, covariates, _, _ = generate_doubly_correlated_heteroscedastic_mixed(
            m,
            n,
            beta_true,
            col_corr=config["col_corr"],
            row_corr=config["row_corr"],
            max_col_var=config["max_col_var"],
            max_row_var=config["max_row_var"],
            seed=0,
            generator=generator_seed
        )

    return Y, covariates, beta_true


def convert_to_df(Y, covariates):
    m, n = Y.shape
    df = pd.DataFrame(
        {
            "subject": np.repeat(np.arange(m), n), 
            "time": np.tile(
                np.arange(n), m
            ),  
            "y": Y.flatten(), 
        }
    )
    for i, X in enumerate(covariates[1:]):
        df[f"x{i+1}"] = X.flatten()
    return df


def generate_output_path(exp_name, config):
    if exp_name == "random-intercept":
        output_path = Path(
            f"eval/benchmarks/{exp_name}_{config["m"]}_{config["n"]}_{config["p"]}_{config["col_rho"]}_{config["row_rho"]}_{config["theta"]}.json"
        )
    elif exp_name == "correlated-homoscedastic":
        output_path = Path(
            f"eval/benchmarks/{exp_name}_{config["m"]}_{config["n"]}_{config["p"]}_{config["col_rho"]}_{config["row_rho"]}_{config["col_corr"]}_{config["col_var"]}_{config["row_var"]}.json"
        )
    elif exp_name in [
        "correlated-heteroscedastic",
        "correlated-heteroscedastic-mixed-covariates",
    ]:
        output_path = Path(
            f"eval/benchmarks/{exp_name}_{config['m']}_{config['n']}_{config['p']}_{config['col_rho']}_{config['row_rho']}_{config['col_corr']}_{config['row_corr']}_{config['max_col_var']}_{config['max_row_var']}.json"
        )
    return output_path


def fit_mvn(Y, covariates, config):
    Y, covariates = check_arrays(Y, covariates)
    beta_mvn, _, _ = iterative_estimation(
        Y,
        covariates,
        max_iter=10**3,
        row_rho=config["row_rho"],
        col_rho=config["col_rho"],
    )
    return beta_mvn.detach().numpy().tolist()


def fit_base_lmm(df, config):
    X_matrix = sm.add_constant(df.iloc[:, 3:])
    model = MixedLM(df["y"], X_matrix, groups=df["subject"])
    result = model.fit()
    return result.params[:-1].to_numpy().tolist()


def fit_vc_lmm(df, config):
    p = config["p"]
    df["time_cat"] = df["time"].astype("category")
    vc = {"time": "0 + C(time_cat)"}
    covariate_terms = " + ".join([f"x{i+1}" for i in range(0, p - 1)])
    formula = f"y ~ {covariate_terms}"
    model = MixedLM.from_formula(
        formula, groups=df["subject"], data=df, re_formula="1", vc_formula=vc
    )
    result = model.fit()
    return result.params[:-2].to_numpy().tolist()


def fit_gee(df, config):
    p = config["p"]
    df["time"] = df["time"].astype("int")
    fam = sm.families.Gaussian()
    ind = sm.cov_struct.Exchangeable()
    covariate_terms = " + ".join([f"x{i+1}" for i in range(0, p - 1)])
    formula = f"y ~ {covariate_terms}"
    model = smf.gee(formula, "subject", df, cov_struct=ind, family=fam, time=df["time"])
    result = model.fit()
    return result.params.to_numpy().tolist()

def run_experiment(exp_name, exp, n_replicates=50):

    results = copy.deepcopy(results_template)
    results.update({k: exp[k] for k in exp})

    output_path = generate_output_path(exp_name, exp)
    if output_path.exists():
        return
    for r in tqdm(range(n_replicates)):
        exp["seed"] = r

        Y, covariates, beta_true = generate_data(exp_name, exp)
        df = convert_to_df(Y, covariates)
        beta_mvn = fit_mvn(Y, covariates, exp)
        beta_base_lmm = fit_base_lmm(df, exp)
        beta_vc_lmm = fit_vc_lmm(df, exp)
        beta_gee = fit_gee(df, exp)
        results["estimates_mvn"].append(beta_mvn)
        results["estimates_base_lmm"].append(beta_base_lmm)
        results["estimates_vc_lmm"].append(beta_vc_lmm)
        results["estimates_gee"].append(beta_gee)
        results["seed"].append(r)

    results["beta_true"].append(beta_true.tolist())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fp:
        json.dump(results, fp, indent=4)

def main():
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

        Parallel(n_jobs=5)(delayed(run_experiment)(exp_name, exp) for exp in experiments)
            
        
if __name__ == "__main__":
    main()
