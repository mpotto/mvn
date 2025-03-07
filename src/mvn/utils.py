import torch

def check_arrays_and_matrices(Y, covariates, Sigma_inv, Delta_inv):
    Y = torch.from_numpy(Y)
    covariates = torch.from_numpy(covariates)
    Sigma_inv = torch.from_numpy(Sigma_inv)
    Delta_inv = torch.from_numpy(Delta_inv)
    return Y, covariates, Sigma_inv, Delta_inv

def check_arrays(Y, covariates):
    Y = torch.from_numpy(Y)
    covariates = torch.from_numpy(covariates)
    return Y, covariates
