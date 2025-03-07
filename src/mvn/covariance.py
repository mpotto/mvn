import torch

def l2_regularized_covariance_estimation(X, row_rho, col_rho):
    # Perform Singular Value Decomposition (SVD)
    U, D, Vt = torch.linalg.svd(X, full_matrices=True)
    d = D  # Singular values
    n, p = X.shape
    r = torch.linalg.matrix_rank(X)

    # Initialize beta* and theta*
    beta_star = torch.zeros(n, dtype=torch.float64)
    theta_star = torch.zeros(p, dtype=torch.float64)

    for i in range(n):
        if i >= r:
            beta_star[i] = 2 * torch.sqrt(torch.tensor(row_rho / p))
        else:
            c1_i = -4 * col_rho * p**2
            c2_i = 32 * row_rho * col_rho * p + d[i]**4 * (n - p)
            c3_i = 4 * row_rho * (d[i]**4 - 16 * row_rho * col_rho)

            # Compute beta* using the given formula
            discriminant = c2_i**2 - 4 * c1_i * c3_i
            if discriminant > 0:
                beta_star[i] = torch.sqrt((-c2_i - torch.sqrt(discriminant)) / (2 * c1_i))
            else:
                beta_star[i] = 0.0 
    for i in range(p):
        if i >= r:
            theta_star[i] = 2 * torch.sqrt(torch.tensor(col_rho / n))
        else:
            denominator = p * beta_star[i]**2 - 4 * row_rho
            theta_star[i] = (d[i]**2 * beta_star[i]) / denominator

    Sigma_star = U @ torch.diag(beta_star) @ U.T
    Delta_star = Vt.T @ torch.diag(theta_star) @ Vt

    return Sigma_star, Delta_star
