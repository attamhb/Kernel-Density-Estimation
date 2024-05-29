import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import time, datetime, os

import lib.tessellation as ts
from lib.kde import KdeRegularizer

np.random.seed(42)

mpl.rc("axes", labelsize=14)
mpl.rc("xtick", labelsize=12)
mpl.rc("ytick", labelsize=12)


############################################################
########## functions to save figures and dataframes #########
#############################################################

# use the following function to save figures
def savefig(fig_id, fig_dir, fig_extension="pdf"):
    root_dir = "."
    fig_path = os.path.join(root_dir, fig_dir)
    os.makedirs(fig_path, exist_ok=True)
    path = os.path.join(fig_path, fig_id + "." + fig_extension)
    plt.tight_layout()
    print(f"Saving figure to {path}")
    plt.savefig(path, format=fig_extension, dpi=500)

#####use the following function to save pandas dataframes
def save_df(dataframe, file_id, file_dir, file_formate="csv"):
    root_dir = "."
    file_path = os.path.join(root_dir, file_dir)
    os.makedirs(file_path, exist_ok=True)
    path = os.path.join(file_path, file_id + "." + file_formate)
    print(f"Saving dataframe  to {path}")
    dataframe.to_csv(path, index=False)

#############################################################
################ KDE plot functions #########################
#############################################################

# plot all the kernels used in the KDE
def plot_kenels(myKde, cfg, fig_name, fig_dir, fig_save=True):
    x_grid = eval(cfg['X_GRID'])
    plt.figure()
    for ci in cfg['KERNEL_LIST']:
        plt.plot(x_grid, myKde.kernel_f(x_grid, ci), lw=3, label=f"{ci}")
    plt.xlim([-4, 4])
    plt.xlabel("x"); plt.ylabel("K(x)")
    plt.legend(); plt.grid()
    if fig_save:
        savefig(fig_name, fig_dir)
    plt.show()

#############################################################
# plot kde for  three different bandwith values for comaprison 
def compare_kde_plot(cfg, myGenerator, myKde, fig_name, fig_dir, fig_save=True):
    plt.figure()
    x_grid = eval(cfg['X_GRID'])
    plt.plot(x_grid, myGenerator.pdf(x_grid), "k-", lw=3, label="True PDF")
    for hi in cfg['H_COMPARE']:
        plt.plot(x_grid, myKde(hi), lw=3, label=f"Bandwidth={hi}")
    plt.xlabel("x")
    plt.legend()
    plt.grid()
    if fig_save:
        savefig(fig_name, fig_dir)
    plt.show()

#############################################################
# plot KDE with the opimal bandwith values for each kernel
def plot_opimal_kde(myGenerator,X_sample, cfg, fig_name,fig_dir, csv_dir, fig_save=True):
    x_grid = eval(cfg['X_GRID'])
    plt.plot(x_grid, myGenerator.pdf(x_grid), "k-", lw=3, label="True PDF")
    for kr in cfg['KERNEL_LIST']:
        df = pd.read_csv(f"{csv_dir}/{kr}_kde_mise.csv")
        h_optimal = df.loc[df["mise"].idxmin(), "h"]
        print(f"mies for kde with {kr} kernel: {df['mise'].min()}")
        myKde_= KdeRegularizer(X_sample, x_grid, kr)
        plt.plot(x_grid, myKde_(h_optimal), lw=2, label=f"KDE with {kr}")
    plt.legend()
    plt.grid()
    if fig_save:
        savefig(fig_name, fig_dir)
    plt.show()

#############################################################
# plot KDE MISE for all kernels
def plot_kde_mise(cfg, fig_name,fig_dir, csv_dir, fig_save=True):
    for k in cfg['KERNEL_LIST']:
        df = pd.read_csv(f"{csv_dir}/{k}_kde_mise.csv")
        plt.plot(df["h"], df["mise"], lw=3, label=f"{k}")
    plt.xlim([0.02, 3])
    plt.ylim([0.0001, 0.003])
    plt.xlabel("Bandwith")
    plt.ylabel("MISE")
    plt.legend()
    plt.grid()
    if fig_save:
        savefig(fig_name, fig_dir)
    plt.show()

#############################################################
# plot KDE with rule of thumb bandwith as well as with opimal bandwith 
def plot_kde_rule_of_thumb(myKde, X_sample, myGenerator, cfg, fig_name, fig_dir, csv_dir, fig_save=True):
    x_grid = eval(cfg['X_GRID'])
    df = pd.read_csv(f"{csv_dir}/normal_kde_mise.csv")
    h_rot = 1.06*np.std(X_sample)*cfg['N']**(-1/5)  
    h_optimal = df.loc[df["mise"].idxmin(), "h"]
    plt.plot(x_grid, myGenerator.pdf(x_grid), "k-", lw=3, label="True PDF")
    plt.plot(x_grid, myKde(h_optimal),"b", lw=3, label="Optimal h")
    plt.plot(x_grid, myKde(h_rot),"r", lw=3, label="ROT h")
    plt.grid()
    plt.legend()
    print("h_rot:", h_rot)
    print("h_optimal:", h_optimal)
    if fig_save:
        savefig(fig_name, fig_dir)
    plt.show()

#############################################################
# plot KDE using the Gaussian Kernels for different values of bandwith
def plot_kde_mise_N(cfg, fig_name, fig_dir, csv_dir, fig_save=True):
    kernel = "normal"
    for n in cfg['N_GRID']:
        df_mise_N = pd.read_csv(f"{csv_dir}/N{n}_{kernel}_kde_mise.csv")
        plt.plot(df_mise_N['h'],df_mise_N['mise'],lw=3, label=f"{n}")
        print(f"MISE MIEN for N={n:}",df_mise_N["mise"].min())
    plt.ylim([0, 0.01])
    plt.ylabel("MISE")
    plt.xlabel("h")
    plt.legend()
    plt.grid()
    if fig_save:
        savefig(fig_name, fig_dir)
    plt.show()
 
#############################################################
# loglogplot of mise vs sample size using gaussian kernel
def plot_loglog_mise(cfg, fig_name, fig_dir, csv_dir, fig_save=True):
    kernel = "normal"
    mise_vec = []
    for n in cfg['N_GRID']:
        df_mise_N = pd.read_csv(f"{csv_dir}/N{n}_{kernel}_kde_mise.csv")
        mise_vec.append(df_mise_N['mise'].min())
    plt.loglog(cfg['N_GRID'], mise_vec,"b-.",lw=3)
    plt.ylabel("Log-Log of LOWEST MISE"); plt.xlabel("N")
    #plt.legend(); #plt.grid()
    if fig_save:
        savefig(fig_name, fig_dir)
    plt.show()

#############################################################
#

#############################################################
################ Tessellation plot functions w##############
#############################################################


# plot simple tess
def plot_simple_tess(myReg, myGenerator, cfg, fig_name, fig_dir,  fig_save=True):
    # fig_dir = cfg['SUB_DIR_TESS']
    x_grid = eval(cfg['X_GRID'])
    plt.figure()
    plt.plot(x_grid, myReg(0), "c", lw=2, label="f_tess")
    plt.plot(x_grid, myGenerator.pdf(x_grid), "k", lw=3, label="f_true")
    plt.grid()
    plt.xlabel("x")
    plt.legend()
    if fig_save:
        savefig(fig_name, fig_dir)
    plt.show()

# # plot recursive tess
def plot_recursive_tess(myTessRec, myGenerator, cfg, fig_name, fig_dir, fig_save=True):
    # fig_dir = cfg['SUB_DIR_TESS']
    x_grid = eval(cfg['X_GRID'])
    X_sample = myGenerator.sample(cfg['N'])
    myReg = myTessRec(X_sample, x_grid, cfg['WTS'])
    plt.plot(x_grid, myReg(cfg['RECURSE_PRMTR']), "g", lw=2, label="f_recursive")
    plt.plot(x_grid, myGenerator.pdf(x_grid), "k-", lw=3, label="f_true")
    plt.grid()
    plt.xlabel("x")
    plt.legend()
    if fig_save:
        savefig(fig_name, fig_dir)
    plt.show()

# # plot ftess with constant coeffs for different time values




# plot ode tess estimation for all coeff types    
def plot_tess_all_coeffs(myGenerator, myTessConst, myTessGauss, myTessReg, myKdeNorm, cfg, fig_name , fig_dir, fig_save=True):
    # fig_dir = cfg['SUB_DIR_TESS']
    x_grid = eval(cfg['X_GRID'])
    X_sample = myGenerator.sample(cfg['N'])

    plt.figure()
    plt.clf()
    plt.plot(x_grid, myGenerator.pdf(x_grid), "k-", lw=3, label="f_True")
    plt.plot(x_grid, myTessConst(cfg['T_CONSTANT']), "b--", lw=2, label="Constant")
    plt.plot(x_grid, myTessGauss(cfg['T_GAUSSIAN']), "g-.", lw=2, label="Gaussian")
    plt.plot(x_grid, myTessReg(cfg['T_REGULAR']), lw=2, label="Regularized")
    plt.plot(x_grid, myKdeNorm(0.6), lw=2, label="KDE")
    plt.grid()
    plt.xlabel("x")
    plt.legend()
    if fig_save:
        savefig(fig_name, fig_dir)
    plt.show()

# plot f with compact support whicm generates coeffs
def f_cmpc_support(X_sample, cfg,  fig_name, fig_dir, fig_save=True):
    # fig_dir = cfg['SUB_DIR_TESS']
    x_grid = eval(cfg['X_GRID'])
    plt.figure()
    for r in eval(cfg['RADIUS_GRID']):
        f_eq = ts.f_eqlbrm(x_grid, X_sample, radius=r)
        plt.plot(x_grid, f_eq, lw=3, label=f"Radius={np.round(r,1)}")
    plt.grid()
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("h(x)")
    if fig_save:
        savefig(fig_name, fig_dir)
    plt.show()

def plot_ftess_constant_compare(myTessConst, myGenerator , cfg,fig_name, fig_dir, fig_save=True):
    # figdir = cfg['SUB_DIR_TESS']
    x_grid = eval(cfg['X_GRID'])
    t_ = [100, 217, 5000]
    f_true = myGenerator.pdf(x_grid)
    f_tess1 = myTessConst(t_[0])
    f_tess2 = myTessConst(t_[1])
    f_tess3 = myTessConst(t_[2])
    plt.plot(x_grid, f_tess1, lw=3, label=f"t={t_[0]}")
    plt.plot(x_grid, f_tess2, lw=3, label=f"t={t_[1]}")
    plt.plot(x_grid, f_tess3, lw=3, label=f"t={t_[2]}")
    plt.plot(x_grid, f_true, "k", lw=3, label="f_true")
    plt.grid()
    plt.legend()
    plt.xlabel("x")
    if fig_save:
        savefig(fig_name, fig_dir)
    plt.show()


def plot_tess_mise(cfg, fig_name, fig_dir, csv_dir, fig_save=True):
    # fig_dir = cfg['SUB_DIR_TESS']
    x_grid = eval(cfg['X_GRID'])
    for ci in cfg['COEFF_LIST']:
        df = pd.read_csv(f"{csv_dir}/{ci}_tess_mise.csv")
        print(df.mise.min(), "at t =", df.h[df.mise.argmin()])
        plt.plot(df.h, df.mise, lw=3, label=f"{ci}_coeffs")
    plt.ylim([0, 0.01])
    plt.grid()
    plt.xlabel("Time")
    plt.ylabel("MISE")
    plt.legend()
    if fig_save:
        savefig(fig_name, fig_dir)
    plt.show()





def optimal_tess_plot(TessReg, myGenerator, cfg, fig_name, fig_dir, csv_dir, fig_save=True):
    # fig_dir = cfg['SUB_DIR_TESS']
    x_grid = eval(cfg['X_GRID'])
    X_sample = myGenerator.sample(cfg['N'])
    plt.figure()
    plt.plot(x_grid, myGenerator.pdf(x_grid), "k-", lw=3, label="f_true")
    for ci in cfg['COEFF_LIST']:
        myTess = TessReg(X_sample, x_grid, coeff_type=ci)
        df = pd.read_csv(f"{csv_dir}/{ci}_tess_mise.csv")
        # print(df.mise.min(), "at t =", df.h[df.mise.argmin()])
        t_arg = df.mise.argmin()
        # print(t_arg)
        plt.plot(x_grid, myTess(df.h[t_arg]), lw=2, label=f"f_tess_{ci}")

    plt.grid()
    plt.xlabel("Time")
    plt.ylabel("MISE")
    plt.legend()
    if fig_save:
        savefig(fig_name, fig_dir)
    plt.show()




def loglog_plot_tess_mise(cfg,  fig_dir, csv_dir):
    # fig_dir = cfg['SUB_DIR_TESS']
    lowest_mise = []
    optimal_time = []
    for ci in cfg['COEFF_LIST']:
        for n in cfg['N_GRID']:
            df = pd.read_csv(f"{csv_dir}/{ci}_tess_mise_{n}.csv")
            time_ = df.h[df.mise.argmin()]
            mise_ = df.mise.min()
            print(mise_, "at t =", time_)
            optimal_time.append(time_)
            lowest_mise.append(mise_)

    arr_mise = np.array(lowest_mise)
    arr_time = np.array(optimal_time)
    mise = arr_mise.reshape(3, len(cfg['N_GRID']))
    optimal_time = arr_time.reshape(3, len(cfg['N_GRID']))
    plt.figure()
    plt.loglog(cfg['N_GRID'], mise[0, :], lw=3, label="constant")
    plt.loglog(cfg['N_GRID'], mise[1, :], lw=3, label="gaussian")
    plt.loglog(cfg['N_GRID'], mise[2, :], lw=3, label="regular")
    plt.legend()
    plt.xlabel("Sample Size")
    savefig("tess_mise_loglog_plot", fig_dir)
    plt.show()

    plt.figure()
    plt.loglog(cfg['N_GRID'], optimal_time[0, :], lw=3, label="constant")
    plt.loglog(cfg['N_GRID'], optimal_time[1, :], lw=3, label="gaussian")
    plt.loglog(cfg['N_GRID'], optimal_time[2, :], lw=3, label="regular")
    plt.legend()
    plt.xlabel("Sample Size")
    savefig("optimal_time_vs_N_loglog", fig_dir)
    plt.show()



def ftess_equillibrium_sol(myGenerator, cfg, fig_name,fig_dir, fig_save=True):
    X_sample = myGenerator.sample(cfg['N'])
    x_grid = eval(cfg['X_GRID'])
    # fig_dir = cfg['SUB_DIR_TESS']
    coeff = cfg['COEFF_LIST'][0]
    f_equi = ts.equilibrium_pdf(X_sample,  x_grid, coeff)
    plt.plot(x_grid, f_equi,"b", label="equillibrium",lw=3)
    plt.plot(x_grid, myGenerator.pdf(x_grid),"k", label="f_true",lw=3)
    plt.grid()
    plt.legend()
    plt.xlabel('x')    
    if fig_save:
        savefig(fig_name, fig_dir)
    plt.show()

#%#####################################################################

def plot_lscvh(fig_name, fig_dir, csv_dir, fig_save=True):
    df_lscv = pd.read_csv(f"{csv_dir}/least_square_cv_normal_kde_1000_pts.csv")
    df_ise = pd.read_csv(f"{csv_dir}/integrated_sqrue_error_kde_1000_pts.csv")
    plt.plot(df_lscv['h_grid'], df_lscv['ise_approx'], lw=2, label="LSCV(h) + ||f||^2")
    plt.plot(df_ise['h_grid'], df_ise['ise'], lw=2, label="ISE")
    plt.legend()
    plt.xlabel("Bandwith")
    plt.grid()
    if fig_save:
        savefig(fig_name, fig_dir)
    plt.show()
