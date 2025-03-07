
import torch

from mvn.covariance import l2_regularized_covariance_estimation

def log_likelihood(Y, covariates, beta, Sigma_inv, Delta_inv):
    p = beta.shape[0]
    Y_c = Y - sum(beta[i] * covariates[i] for i in range(p))
    return -torch.trace(Sigma_inv @ Y_c @ Delta_inv @ Y_c.T)

def estimate_beta(Y, covariates, Sigma_inv, Delta_inv):
    p = len(covariates)
    ell = lambda beta: log_likelihood(Y, covariates, beta, Sigma_inv, Delta_inv)
    A = torch.autograd.functional.hessian(ell, torch.zeros(p))
    b = torch.autograd.functional.jacobian(ell, torch.zeros(p))
    estimates = torch.linalg.solve(A, -b)
    return estimates

def iterative_estimation(Y, covariates, max_iter=10**3, row_rho=1, col_rho=1, abs_tol=1e-8):
    m, n = Y.shape 
    p = len(covariates)
    Sigma_inv = torch.eye(m, dtype=torch.float64)
    Delta_inv = torch.eye(n, dtype=torch.float64)
    beta_prev = torch.inf
    for iter in range(max_iter):
        beta_iter = estimate_beta(Y, covariates, Sigma_inv, Delta_inv)
        Y_c = Y - sum(beta_iter[i] * covariates[i] for i in range(p))
        Sigma_star, Delta_star = l2_regularized_covariance_estimation(Y_c, col_rho=col_rho, row_rho=row_rho)
        Sigma_inv = torch.linalg.inv(Sigma_star)
        Delta_inv = torch.linalg.inv(Delta_star)
        if torch.norm(beta_iter - beta_prev) <= abs_tol:
            break
        beta_prev = beta_iter
    return beta_iter, Sigma_star, Delta_star

