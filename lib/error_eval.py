import os
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import linalg
from scipy.interpolate import interp1d
from tqdm import tqdm

# #####################################################

def estimate_MISE(myGenerator, myRegularizer, h_interval, N, nbr_iter, x_grid, kernel_or_coeff):
    # init
    f_exact = myGenerator.pdf(x_grid)
    dx = x_grid[1] - x_grid[0]
    f_hat = np.zeros((nbr_iter, len(h_interval), len(x_grid)))

    # loop
    for k in range(nbr_iter):
        X_sample = myGenerator.sample(N)
        myReg = myRegularizer(X_sample, x_grid, kernel_or_coeff)
        for (idx_h, h) in enumerate(h_interval):
            f_hat[k, idx_h, :] = myReg(h)  # estimate f_regularize

        if k % 10 == 0:
            print(k)

    integrated_squared_error = np.trapz((f_exact - f_hat) ** 2, x_grid, axis=2)
    mean_squared_error = np.mean(integrated_squared_error, axis=0)

    integrated_variance = np.trapz(np.var(f_hat, axis=0), x_grid, axis=1)

    squared_bias = (np.mean(f_hat, axis=0) - f_exact) ** 2
    integrated_squared_bias = np.trapz(squared_bias, x_grid, axis=1)

    df = pd.DataFrame(
        {
            "h": h_interval,
            "mise": mean_squared_error,
            "integ_var": integrated_variance,
            "integ_sq_bias": integrated_squared_bias,
        }
    )
    # print("mise df created")
    return df

def compute_mise(myGenerator, myRegularizer, h_interval, N, nbr_iter, x_grid, kernel_or_coeff):
    """
    Compute the MISE for a given generator and regularizer.
    It returns for various bandwith 'h' the corresponding mise as well as a decomposition bias- variance.
    """
    # initialize
    f_exact = myGenerator.pdf(x_grid)
    mise = np.zeros(len(h_interval))
    error_bias_f_hat = np.zeros(len(h_interval))
    error_var_f_hat = np.zeros(len(h_interval))

    # loop
    for idx_h, h in enumerate(tqdm(h_interval)):
    #for idx_h, h in enumerate(h_interval):
        ise = 0
        f_hat_all = np.zeros((nbr_iter, len(x_grid)))
        for k in range(nbr_iter):
            # estimate f_hat
            X_sample = myGenerator.sample(N)
            myReg = myRegularizer(X_sample, x_grid, kernel_or_coeff)
            f_hat_k = myReg(h)
            # save and estimate the ise
            f_hat_all[k,:] += f_hat_k
            ise += np.trapz((f_exact - f_hat_k) ** 2, x_grid)
        # mise
        mise[idx_h] = ise / nbr_iter
        # bias-variance
        f_hat_mean = np.mean(f_hat_all, axis=0)
        error_bias_f_hat[idx_h] = np.trapz((f_hat_mean - f_exact) ** 2, x_grid)
        f_hat_variance = np.var(f_hat_all, axis=0)
        error_var_f_hat[idx_h] = np.trapz(f_hat_variance, x_grid)

    # finally
    df = pd.DataFrame({
        "h": h_interval,
        "mise": mise,
        "error_bias": error_bias_f_hat,
        "error_variance": error_var_f_hat
    })

    return df

def estimate_MSE(myGenerator, myRegularizer, t_val, N, nbr_iter, x_grid, coeffs=""):
    f_exact = myGenerator.pdf(x_grid)
    dx = x_grid[1] - x_grid[0]
    errorL2 = np.zeros((len(x_grid)))
    for n in range(nbr_iter):
        X_sample = myGenerator.sample(N)
        if coeffs == "":
            myReg = myRegularizer(X_sample, x_grid)
        else:
            myReg = myRegularizer(X_sample, x_grid, coeffs)
        f_estimate = myReg(t_val)
        errorL2 += (f_exact - f_estimate) ** 2
    mse = errorL2 / N
    df_mse = pd.DataFrame({"x_grid": x_grid, "mse": mse})
    return df_mse


################################################################################
def estimate_MSE_decompose(
    myGenerator, myRegularizer, t_val, N, nbr_iter, x_grid, coeffs=""
):
    f_exact = myGenerator.pdf(x_grid)
    dx = x_grid[1] - x_grid[0]
    mat_error = np.zeros((nbr_iter, len(x_grid)))
    for n in range(nbr_iter):
        X_sample = myGenerator.sample(N)
        if coeffs == "":
            myReg = myRegularizer(X_sample, x_grid)
        else:
            myReg = myRegularizer(X_sample, x_grid, coeffs)
        f_estimate = myReg(t_val)
        mat_error[n, :] = f_estimate - f_exact
    # stat
    bias_x = np.mean(mat_error, axis=0)
    var_x = np.var(mat_error, axis=0)
    mse_x = bias_x ** 2 + var_x
    df_mse = pd.DataFrame(
        {"x_grid": x_grid, "mse": mse_x, "bias": bias_x, "var": var_x}
    )
    return df_mse


##############################################################################
def estimate_tessellation(X_update, x_grid):
    N = len(X_update)
    X_mid_tp = (X_update[1:] + X_update[:-1]) / 2
    x_left = 2 * X_update[0] - X_mid_tp[0]
    x_right = 2 * X_update[-1] - X_mid_tp[-1]
    X_mid = np.concatenate([[x_left], X_mid_tp, [x_right]])
    # compute f_tess
    f_at_index = np.concatenate([[0], 1 / np.diff(X_mid) * 1 / N, [0]])
    index_voronoi = np.sum(x_grid > X_mid.reshape(-1, 1), axis=0)
    return f_at_index[index_voronoi]

def estimate_mise_tess(myGenerator, myRegularizer, h_interval, N, nbr_iter, x_grid, kernel_or_coeff):
    f_exact = myGenerator.pdf(x_grid)
    mise = np.zeros((nbr_iter, len(h_interval)))
    dt = h_interval[1] - h_interval[0]
    for k in range(nbr_iter):
        X_sample = myGenerator.sample(N)
        myReg = myRegularizer(X_sample, x_grid, kernel_or_coeff)
        A = myReg.matrix_operator(N, X_sample, kernel_or_coeff)
        L = linalg.expm(dt * A)
        X_sol = np.sort(X_sample)
        for (idx_h, h) in enumerate(h_interval):
            f_hat = estimate_tessellation(X_sol, x_grid)
            integrated_squared_error = np.trapz((f_exact - f_hat) ** 2, x_grid)
            mise[k, idx_h] = integrated_squared_error

            X_sol = np.matmul(L, X_sol)

        if k % 10 == 0:
            print(k)

    mean_squared_error = np.mean(mise, axis=0)
    return pd.DataFrame({"h": h_interval, "mise": mean_squared_error})

# def estimate_mise_tess(
#     myGenerator, myRegularizer, h_interval, N, nbr_iter, x_grid, kernel_or_coeff
# ):
#     # init
#     f_exact = myGenerator.pdf(x_grid)

#     # dx = x_grid[1] - x_grid[0]
#     f_hat = np.zeros((nbr_iter, len(h_interval), len(x_grid)))

#     # loop
#     dt = h_interval[1] - h_interval[0]

#     for k in range(nbr_iter):
#         X_sample = myGenerator.sample(N)
#         myReg = myRegularizer(X_sample, x_grid, kernel_or_coeff)

#         A = myReg.matrix_operator(N, X_sample, kernel_or_coeff)
#         L = linalg.expm(dt * A)

#         X_sol = np.sort(X_sample)
#         for (idx_h, h) in enumerate(h_interval):
#             f_hat[k, idx_h, :] = estimate_tessellation(X_sol, x_grid)
#             X_sol = np.matmul(L, X_sol)

#         if k % 10 == 0:
#             print(k)

#     integrated_squared_error = np.trapz((f_exact - f_hat) ** 2, x_grid, axis=2)
#     mean_squared_error = np.mean(integrated_squared_error, axis=0)

#     integrated_variance = np.trapz(np.var(f_hat, axis=0), x_grid, axis=1)

#     squared_bias = (np.mean(f_hat, axis=0) - f_exact) ** 2
#     integrated_squared_bias = np.trapz(squared_bias, x_grid, axis=1)

#     df = pd.DataFrame(
#         {
#             "h": h_interval,
#             "mise": mean_squared_error,
#             "integ_var": integrated_variance,
#             "integ_sq_bias": integrated_squared_bias,
#         }
#     )
#     # print("mise df created")
#     return df
