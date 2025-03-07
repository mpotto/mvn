import numpy as np

from mvn.data import generate_random_intercept, generate_doubly_correlated_heteroscedastic_mixed
from mvn.model import estimate_beta, iterative_estimation
from mvn.utils import check_arrays_and_matrices


def main():
    m, n, p = 20, 10, 5
    beta = np.arange(p)
    Y, covariates = generate_random_intercept(m, n, beta, theta=5, generator=10)
    Sigma_inv = np.eye(m)
    Delta_inv = np.eye(n)
    Y, covariates, Sigma_inv, Delta_inv = check_arrays_and_matrices(
        Y, covariates, Sigma_inv, Delta_inv
    )
    print("Random Intercept:", estimate_beta(Y, covariates, Sigma_inv, Delta_inv))

    Sigma_inv = np.eye(m)
    Delta_inv = np.eye(n)
    Y, covariates, _, _ = generate_doubly_correlated_heteroscedastic_mixed(
        m, n, beta, col_corr=0.999, row_corr=0.99
    )
    Y, covariates, Sigma_inv, Delta_inv = check_arrays_and_matrices(
        Y, covariates, Sigma_inv, Delta_inv
    )
    beta_hat, _, _ = iterative_estimation(
        Y, covariates, 5 * 10**2, row_rho=1e-4, col_rho=1e-4
    )
    print("Correlated Heteroscedastic:", beta_hat)


if __name__ == "__main__":
    main()
