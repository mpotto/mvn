import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TABLEAU_COLORS


plt.style.use("mvn.mplstyle")
COLORS = list(TABLEAU_COLORS.values())

# # This is the updated one.
# def generate_output_path(exp_name, config):
#     m, n = config["mn"]
#     p = config["p"]
#     col_rho, row_rho = config["rho"]

#     if exp_name == "random-intercept":
#         theta = config["theta"]
#         output_path = Path(
#             f"eval/benchmarks/{exp_name}_{m}_{n}_{p}_{row_rho}_{col_rho}_{theta}.json"
#         )
#     elif exp_name == "correlated-homoscedastic":
#         col_corr, row_corr = config["corr"]
#         col_var, row_var = config["var"]
#         output_path = Path(
#             f"eval/benchmarks/{exp_name}_{m}_{n}_{p}_{row_rho}_{col_rho}_{col_corr}_{row_corr}_{col_var}_{row_var}.json"
#         )
#     elif exp_name in [
#         "correlated-heteroscedastic",
#         "correlated-heteroscedastic-mixed-covariates",
#     ]:
#         col_corr, row_corr = config["corr"]
#         max_col_var, max_row_var = config["max_var"]
#         output_path = Path(
#             f"eval/benchmarks/{exp_name}_{m}_{n}_{p}_{row_rho}_{col_rho}_{col_corr}_{row_corr}_{max_col_var}_{max_row_var}.json"
#         )
#     return output_path



def generate_output_path(exp_name, config):
    if exp_name == "random-intercept":
        output_path = Path(
            f"eval/benchmarks_v0/{exp_name}_{config["m"]}_{config["n"]}_{config["p"]}_{config["col_rho"]}_{config["row_rho"]}_{config["theta"]}.json"
        )
    elif exp_name == "correlated-homoscedastic":
        output_path = Path(
            f"eval/benchmarks_v0/{exp_name}_{config['m']}_{config['n']}_{config['p']}_{config['col_rho']}_{config['row_rho']}_{config['col_corr']}_{config['row_corr']}_{config['col_var']}_{config['row_var']}.json"
        )
    elif exp_name in [
        "correlated-heteroscedastic",
        "correlated-heteroscedastic-mixed-covariates",
    ]:
        output_path = Path(
            f"eval/benchmarks_v0/{exp_name}_{config['m']}_{config['n']}_{config['p']}_{config['col_rho']}_{config['row_rho']}_{config['col_corr']}_{config['row_corr']}_{config['max_col_var']}_{config['max_row_var']}.json"
        )
    return output_path


config_example1 = {"m": 100, "n": 100, "p": 10, "col_rho": 1, "row_rho": 1, "theta": 5}
config_example2 = {
    "m": 100,
    "n": 100,
    "p": 10,
    "col_corr": 0.5,
    "row_corr": 0.5,
    "col_rho": 1,
    "row_rho": 1,
    "col_var": 1,
    "row_var": 1,
}
config_example3 = {
    "m": 100,
    "n": 100,
    "p": 10,
    "col_corr": 0.5,
    "row_corr": 0.5,
    "col_rho": 10,
    "row_rho": 10,
    "max_col_var": 5,
    "max_row_var": 5,
}
config_example4 = {
    "m": 100,
    "n": 100,
    "p": 10,
    "col_corr": 0.5,
    "row_corr": 0.5,
    "col_rho": 10,
    "row_rho": 10,
    "max_col_var": 5,
    "max_row_var": 5,
}


def plot_coef_histogram(exp_name, config, n_coef=5):
    path = generate_output_path(exp_name, config)
    with open(path, "r") as file:
        results = json.load(file)

    beta_true = np.array(*results["beta_true"])
    mvn = np.array(results["estimates_mvn"])
    base_lmm = np.array(results["estimates_base_lmm"])
    vc_lmm = np.array(results["estimates_vc_lmm"])
    gee_lmm = np.array(results["estimates_gee"])

    mvn_first_n_coef = mvn[:, :n_coef]
    base_lmm_first_n_coef = base_lmm[:, :n_coef]
    vc_lmm_first_n_coef = vc_lmm[:, :n_coef]
    gee_lmm_first_n_coef = gee_lmm[:, :n_coef]

    fig, axes = plt.subplots(4, n_coef, sharex="col", sharey="row", figsize=(8, 8))

    methods = ["MVN", "LMM", "LMM+VC", "GEE"]
    data = [
        mvn_first_n_coef,
        base_lmm_first_n_coef,
        vc_lmm_first_n_coef,
        gee_lmm_first_n_coef,
    ]

    colors = {"MVN": COLORS[0], "LMM": COLORS[1], "LMM+VC": COLORS[2], "GEE": COLORS[3]}
    handles = []
    labels = []
    
    for i in range(n_coef):  # Loop over coefficients
        for j, (method_data, method_name) in enumerate(
            zip(data, methods)
        ):  # Loop over methods
            ax = axes[j, i]  # Select the subplot (j: method, i: coefficient)

            # Plot histogram with KDE for the current method and coefficient
            sns.histplot(
                method_data[:, i],
                kde=True,
                color=colors[method_name],
                bins=15,
                ax=ax,
                alpha=0.7,
                label=method_name,
            )
            ax.axvline(beta_true[i], linestyle="--", color="black", label="True Beta")
            if j == 0:
                ax.set_title(fr"$\beta_{i}$")
            ax.set_xlabel("Estimate")
            ax.set_ylabel("Histogram")


    handles, labels = [], []
    for ax in axes[:, 0]:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(
        handles=handles,
        labels=labels,
        loc="center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=5,
        frameon=False,
    )

    plt.tight_layout()
    plt.savefig(f"figures/{exp_name}.pdf", bbox_inches="tight")
    plt.show()

def plot_recovery_regularization(exp_name, config, coef_index):
    # Get all the paths matching the pattern
    paths = list(Path("eval/benchmarks_v0/").glob(
            f"{exp_name}_{config['m']}_{config['n']}_{config['p']}_*_*_{config['col_corr']}_{config['row_corr']}_{config['max_col_var']}_{config['max_row_var']}.json"
        ))

    def extract_numbers(path):
        parts = path.stem.split('_')
        first_star = int(parts[4])  # First * (column variation)
        second_star = int(parts[5])  # Second * (row variation)
        return first_star, second_star
    
    # Sort paths based on extracted numbers (first star for columns, second star for rows)
    paths.sort(key=extract_numbers)

    # Extract unique values for first_star and second_star for the grid layout
    first_stars = sorted(set(extract_numbers(path)[0] for path in paths))
    second_stars = sorted(set(extract_numbers(path)[1] for path in paths))
    
    # Create the grid of plots
    num_rows = len(second_stars)
    num_cols = len(first_stars)
    
    fig, axes = plt.subplots(num_rows, num_cols, sharex="col", sharey="row", figsize=(6, 6))
    
    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if num_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    # Loop over the sorted paths and plot the distribution for each coefficient
    for path in paths:
        # Extract the first and second * values from the path
        first_star, second_star = extract_numbers(path)
        
        # Open the JSON file and load the data
        with open(path, 'r') as f:
            data = json.load(f)
        beta_true = np.array(*data["beta_true"])[coef_index]
    
        # Extract the coefficients for the specific coef_index
        coef_values = np.array(data['estimates_mvn'])[:, coef_index]
        
        # Determine the grid position (row, col)
        row_idx = second_stars.index(second_star)
        col_idx = first_stars.index(first_star)
        
        # Plot the distribution of the coefficient
        ax = axes[row_idx, col_idx]
        ax.set_title(rf"$\rho_c = {first_star}, \rho_r = {second_star}$")
        sns.histplot(
            coef_values,  
            bins=15,
            kde=True,
            color=COLORS[0],
            ax=ax,
            alpha=0.7,
            label="MVN",
        )
        ax.set_xlabel("Estimate")
        ax.set_ylabel("Histogram")
        ax.axvline(beta_true, linestyle="--", color="black", label="True Beta")
        handles, labels = [], []
   
    for ax in axes[:, 0]:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(
        handles=handles,
        labels=labels,
        loc="center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=5,
        frameon=False,
    )
    
    # Adjust layout for better spacing
    plt.tight_layout()
    plt.savefig(f"figures/{exp_name}-regularization.pdf", bbox_inches="tight")
    plt.show()


def main():
    #plot_coef_histogram("random-intercept", config_example1)
    #plot_coef_histogram("correlated-homoscedastic", config_example2)
    #plot_coef_histogram("correlated-heteroscedastic", config_example3)
    #plot_coef_histogram("correlated-heteroscedastic-mixed-covariates", config_example4)

    plot_recovery_regularization("correlated-heteroscedastic-mixed-covariates", config_example4, coef_index=1)
    
    pass


if __name__ == "__main__":
    main()
