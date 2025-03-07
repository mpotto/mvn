import numpy as np
from scipy.stats import matrix_normal


def generate_random_intercept(m, n, beta, theta=1, seed=0, generator=None):
    rng = np.random.default_rng(generator)
    p = beta.shape[0]
    # Control if design is fixed or not
    if seed is not None:
        covariates = np.random.default_rng(seed).standard_normal(size=(p - 1, m, n))
    else:
        covariates = rng.standard_normal(size=(p - 1, m, n))

    covariates = np.insert(covariates, 0, np.ones((m, n)), axis=0)
    subject_effects = rng.normal(loc=0, scale=theta, size=(m, 1))

    Y = (
        sum(beta[i] * covariates[i] for i in range(p))
        + rng.standard_normal(size=(m, n))
        + subject_effects
    )
    return Y, covariates


def generate_doubly_correlated_homoscedastic(
    m, n, beta, col_corr=0.5, row_corr=0.5, col_var=1, row_var=1, seed=0, generator=None
):
    rng = np.random.default_rng(generator)
    p = beta.shape[0]
    # Control if design is fixed or not
    if seed is not None:
        covariates = np.random.default_rng(seed).standard_normal(size=(p - 1, m, n))
    else:
        covariates = rng.standard_normal(size=(p - 1, m, n))

    Delta = col_var * (col_corr * np.ones((n, n)) + (1 - col_corr) * np.eye(n))
    Sigma = row_var * (row_corr * np.ones((m, m)) + (1 - row_corr) * np.eye(m))

    e = matrix_normal(colcov=Delta, rowcov=Sigma, seed=rng).rvs()

    covariates = np.insert(
        covariates, 0, np.ones((m, n)), axis=0
    )  # Shape becomes (p+1, m, n)
    Y = sum(beta[i] * covariates[i] for i in range(p)) + e
    return Y, covariates, Sigma, Delta


def generate_doubly_correlated_heteroscedastic(
    m,
    n,
    beta,
    col_corr=0.5,
    row_corr=0.5,
    max_col_var=2,
    max_row_var=2,
    seed=0,
    generator=None,
):
    rng = np.random.default_rng(generator)
    p = beta.shape[0]
    # Control both design and stability across seeds of the error term
    if seed is not None:
        covariates = np.random.default_rng(seed).standard_normal(size=(p - 1, m, n))
        col_var = np.random.default_rng(seed).uniform(0, max_col_var, size=n)
        row_var = np.random.default_rng(seed).uniform(0, max_row_var, size=m)
    else:
        covariates = rng.standard_normal(size=(p - 1, m, n))
        col_var = rng.uniform(0, max_col_var, size=n)
        row_var = rng.uniform(0, max_row_var, size=m)

    Delta = np.array(
        [
            [
                col_corr * np.sqrt(col_var[i] * col_var[j]) if i != j else col_var[i]
                for j in range(n)
            ]
            for i in range(n)
        ]
    )
    Sigma = np.array(
        [
            [
                row_corr * np.sqrt(row_var[i] * row_var[j]) if i != j else row_var[i]
                for j in range(m)
            ]
            for i in range(m)
        ]
    )

    e = matrix_normal(colcov=Delta, rowcov=Sigma, seed=rng).rvs()

    covariates = np.insert(
        covariates, 0, np.ones((m, n)), axis=0
    )  # Shape becomes (p+1, m, n)
    Y = sum(beta[i] * covariates[i] for i in range(p)) + e
    return Y, covariates, Sigma, Delta

def generate_doubly_correlated_heteroscedastic_mixed(
    m,
    n,
    beta,
    col_corr=0.5,
    row_corr=0.5,
    max_col_var=2,
    max_row_var=2,
    seed=0,
    generator=None,
):
    rng = np.random.default_rng(generator)
    p = beta.shape[0]
    half_p = (p - 1) // 2

    # Control both design and stability across seeds of the error term
    if seed is not None:
        binary_covariates = np.random.default_rng(seed).integers(0, 2, size=(half_p, m, n))
        continuous_covariates = np.random.default_rng(seed).standard_normal(size=(p - 1 - half_p, m, n))
        col_var = np.random.default_rng(seed).uniform(0, max_col_var, size=n)
        row_var = np.random.default_rng(seed).uniform(0, max_row_var, size=m)
    else:
        binary_covariates = rng.integers(0, 2, size=(half_p, m, n))
        continuous_covariates = rng.standard_normal(size=(p - 1 - half_p, m, n))
        col_var = rng.uniform(0, max_col_var, size=n)
        row_var = rng.uniform(0, max_row_var, size=m)

    covariates = np.concatenate((binary_covariates, continuous_covariates), axis=0)

    Delta = np.array(
        [
            [
                col_corr * np.sqrt(col_var[i] * col_var[j]) if i != j else col_var[i]
                for j in range(n)
            ]
            for i in range(n)
        ]
    )
    Sigma = np.array(
        [
            [
                row_corr * np.sqrt(row_var[i] * row_var[j]) if i != j else row_var[i]
                for j in range(m)
            ]
            for i in range(m)
        ]
    )

    e = matrix_normal(colcov=Delta, rowcov=Sigma, seed=rng).rvs()

    covariates = np.insert(
        covariates, 0, np.ones((m, n)), axis=0
    )  # Shape becomes (p, m, n)
    Y = sum(beta[i] * covariates[i] for i in range(p)) + e
    return Y, covariates, Sigma, Delta
