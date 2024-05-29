import numpy as np
import pandas as pd


def compute_rule_of_thumb_bandwidths(n_grid, myGenerator):
    """ compute the rule-of-thumb bandwidths for given sample sizes """
    h_rot = []  # placeholder for rule-of-thumb bandwidths
    for n in n_grid:
        x_sample = myGenerator.sample(n) # generate sample of size n
        sigma_hat = np.std(x_sample) # standard deviation of the sample
        h_rot_value = 1.06 * sigma_hat * n ** (-1 / 5) # rule-of-thumb h
        h_rot.append(h_rot_value)

    # retur a dataframe with sample sizes and their corresponding bandwidths
    return pd.DataFrame({"n_grid": n_grid, "h_rot": h_rot})



def least_square_cross_validation(h_grid, myKde, X_sample, x_grid, f_true):
    """
    returns a datadrame with the h_grid, lscv(h), approximated ISE values.
    """
    # number of samples
    N = len(X_sample)
    # placeholder for lscv values for each bandwidth
    lscv = np.zeros(len(h_grid))
    ise = np.zeros(len(h_grid))
   
    for idx, h in enumerate(h_grid):
        # compute matrix of differences between every pair of samples
        mat_Xi_Xj = np.outer(X_sample,np.ones(N)) - np.outer(np.ones(N),X_sample)
        # compute the second term for lscv: double sum of kernels
        tp1 = myKde.kde_f(mat_Xi_Xj, 0 * mat_Xi_Xj, h)
        np.fill_diagonal(tp1, 0.0)  # Set diagonal to zero
        double_sum = np.sum(tp1)
        # compute the first term for lscv: sum of squared fhat
        tp2 = myKde.kde_f(mat_Xi_Xj, 0 * mat_Xi_Xj, np.sqrt(2) * h)
        fhat_squared_integral = np.sum(tp2) / N**2
        # compute the lscv value for the current bandwidth
        lscv[idx] = fhat_squared_integral - 2 * double_sum / (N * (N - 1))
        # ise
        f_hat_h = myKde(h)
        ise[idx] = np.trapz((f_true-f_hat_h)**2, x_grid)

    # compute approximate ise by adding the integral to lscv values
    ise_approx = lscv + np.trapz(f_true**2, x_grid)

    return pd.DataFrame({"h_grid": h_grid,
                         "lscv": lscv,
                         "ise_approx": ise_approx,
                         "ise": ise})


def rot_bandwith(myGenerator, n_grid):
    h_rot = []
    for n in n_grid:
        x_sample = myGenerator.sample(n)
        # sigma_hat = np.std(x_sample)
        h_rot.append(np.round(1.06 * np.std(x_sample) * n ** (-1 / 5), 2))
    df = pd.DataFrame({"n_grid": n_grid, "h_rot": np.array(h_rot)})
    return df

def optimal_lscv_rot_mise(file_dir):
    df_mise = pd.read_csv(f"{file_dir}/N1000_normal_kde_mise.csv")
    h_optimal = df_mise.h[np.argmin(df_mise.mise)]
    mise_optimal = df_mise.mise.min()

    df_rot = pd.read_csv(f"{file_dir}/h_rot.csv")
    h_rot = np.round(df_rot.h_rot.iloc[-1], 2)
    mise_h_rot = df_mise[df_mise.h == h_rot]['mise'].iloc[0]

    df_ls = pd.read_csv(f"{file_dir}/lscv_sampleSize_1000.csv")
    h_ls = np.round(df_ls.h_grid[np.argmin(df_ls.lscv)], 2)
    mise_h_ls = df_mise['mise'][df_mise.h == h_ls].iloc[0]

    htype_array = ["optimal", "rule_of_thumb", "lscv"]
    h_vec = [h_optimal, h_rot, h_ls]

    mise_vec = [mise_optimal, mise_h_rot, mise_h_ls]
    mise_array = [format(mise_val, '.4e') for mise_val in mise_vec]

    df_result = pd.DataFrame({"h_type": htype_array,
                              "h_vals": h_vec,
                              "mise_vals": mise_array})

    return df_result




def compute_ise(myGenerator, myReg, h_interval, N, x_grid):
    """compute integrated squared error for kernel density estimation """
    
    X_sample = myGenerator.sample(N)
    f_exact = myGenerator.pdf(x_grid)

    ise_values = []

    # estimate the kde for each bandwidth
    for h in h_interval:
        f_hat = myReg(h)  # Estimate using regularization
        squared_error = (f_hat - f_exact)**2
        ise = np.trapz(squared_error, x_grid)
        ise_values.append(ise)
    # Compile results into a DataFrame
    return pd.DataFrame({"h_grid": h_interval, "ise": np.array(ise_values)})



def compare_mise_vs_bwidth_types(results_path, n_grid):
    h_optimal, h_rot, h_lscv =[], [], []
    min_mise, mise_rot, mise_lscv = [], [], []
    i = 0
    df_hrot_vals = pd.read_csv(
        f"./{results_path}/rule_of_thumb_bandwidhs_vs_Ngrid.csv")
    df_lscv_hvals = pd.read_csv(
        f"./{results_path}/lscv_bandwith_vaules_vs_Ngird.csv")
    for n in n_grid:
        df_mise = pd.read_csv(f"./{results_path}/N{n}_normal_kde_mise.csv")
        h_1 = df_mise.h[np.argmin(df_mise.mise)]
        h_optimal.append(h_1)
        min_mise_val =  format(df_mise.mise.min(), '.4e')
        min_mise.append(min_mise_val)

        h_2 = df_hrot_vals.h_rot.iloc[i]
        h_rot.append(h_2)

        mise_rot_val = format(
            df_mise[np.abs(df_mise['h'] - h_2) < 1e-2]['mise'].iloc[0], '.4e')
        mise_rot.append(mise_rot_val)

        h_3 = df_lscv_hvals.lscv_hvalues.iloc[i]
        h_lscv.append(h_3)

        mise_value = df_mise[np.abs(df_mise['h'] - h_3) < 1e-2]['mise'].iloc[0]
        mise_lscv.append(format(mise_value, '.4e'))

        print(h_1, h_2, h_3)

        i += 1

    df_result = pd.DataFrame({"n_grid": n_grid,
                              "h_optimal": h_optimal,
                              "min_mise": min_mise,
                              "h_lscv": h_lscv,
                              "mise_lscv": mise_lscv,
                              "h_rot": h_rot,
                              "mise_rot": mise_rot,
                              })
    return df_result
