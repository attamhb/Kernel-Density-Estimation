#!/usr/bin/python3
import json
import resource
import numpy as np
import pandas as pd
# from multiprocessing import Pool,  cpu_count
from multiprocessing import Pool
from itertools import product
import time, datetime, os

# import local modules
from lib.generator import GenerateSample
from lib.kde import KdeRegularizer
from lib.plot_utils import save_df
import lib.error_eval as ee
import lib.plot_utils as pu
import lib.lscv_rot as lr

#-###################################################################
# rlimit = int(0.9 * 64 * 1024**3)
# resource.setrlimit(resource.RLIMIT_AS, (rlimit, rlimit))
# #-####
# with open("myCfg.json", "r") as string:
#     cfg = json.load(string)

# load parameters and variables
with open('myCfg.json','r') as string:
    cfg = json.load(string)

# initialize
np.random.seed(42)
t0 = time.time()
cfg['str_time'] = datetime.datetime.now().replace(microsecond=0).isoformat(sep='_').replace(':', 'h', 1).replace(':', 'm', 1)
# saving folder
kde_results_dir = f"{cfg['RESULTS_DIR']}/{cfg['KDE_DIR']}/simulation_{cfg['str_time']}"
kde_csv = f"{kde_results_dir}/{cfg['CSV_SUB_DIR']}"
kde_figs = f"{kde_results_dir}/{cfg['FIG_SUB_DIR']}"
os.makedirs(f"{kde_figs}", exist_ok=True)
os.makedirs(f"{kde_csv}", exist_ok=True)

# true density (on grid) and sample
x_grid = eval(cfg['X_GRID'])
h_interval = eval(cfg['H_INTERVAL'])
myGenerator = GenerateSample(np.array(cfg['WEIGHT']), cfg['MU'] , cfg['VAR'] )
X_sample = myGenerator.sample(cfg['N'] )
f_exact = myGenerator.pdf(x_grid)

# define Kde regularize
myKde = KdeRegularizer(X_sample, x_grid, kernel_type="normal")


# option 1.1: Simulation
#compute and save kernel mise for all kernels with sample size fixed
if (1==1):
    for kr in cfg['KERNEL_LIST']:
        print(f"computing MISE with {kr}")
        df_mise_kde = ee.estimate_MISE(myGenerator, KdeRegularizer, h_interval, cfg['N'], cfg['NBR_ITER'], x_grid, kr)
        #save_df(df_mise_kde, f"{kr}_kde_mise", cfg['SUB_DIR_KDE'])
        save_df(df_mise_kde, f"{kr}_kde_mise", kde_csv)

# compute mise with all kernels and for different sample sizes
if (1==0):
    for kernel in cfg['KERNEL_LIST']:
        for n in cfg['N_GRID']:
            df_mise_kde_n = ee.estimate_MISE(myGenerator, KdeRegularizer, h_interval, n, cfg['NBR_ITER'], x_grid, kernel)
            save_df(df_mise_kde_n, f"N{n}_{kernel}_kde_mise", kde_csv)


# simulation time, upade cfg.json and save it
time_elapsed = time.time() - t0
cfg['time_simulation'] = '{:.0f}m {:.0f}s'.format( time_elapsed // 60, time_elapsed % 60)
print('--- simulation completed in '+cfg['time_simulation']+' ---')
with open(f"./{kde_results_dir}/myCfg.json",'w') as jsonFile:
    json.dump(cfg, jsonFile, indent=2)


# plot all the six kernels
if (1==1):
    pu.plot_kenels(myKde,cfg, "kernels_plot", kde_figs ,fig_save=True)


#%########## Part 1.2: Plot of imulations results

#%## provide the folder name! "../results/kde/Report_2023-03-12_15h35m29"
# plot KDE with Gaussain Kernel for three different choice of bandwith to comare smoothness
if (1==1):
    pu.compare_kde_plot(cfg, myGenerator, myKde, "kde_bandwidth_compare_plot", kde_figs, fig_save=True)

# plot KDE using all the six kernel  with the opimal bandwith
if (1==1):
    pu.plot_opimal_kde(myGenerator,X_sample, cfg, "kde_with_all_kernels_plot",kde_figs, kde_csv, fig_save=True)

# plot kernel mise  vs the bandwith values for all the six kernels
if (1==1):
    pu.plot_kde_mise(cfg, "kde_mise_compare_plot", kde_figs, kde_csv, fig_save=True)

# plot KDE with the Gaussian kernel with  the opimal bandwith as well as with the ruel-of-thumb bandwith (silverman)

if (1==1):
    pu.plot_kde_rule_of_thumb(myKde, X_sample, myGenerator, cfg, "kde_h_optimal_vs_h_rot", kde_figs, kde_csv, fig_save=True)

# plot KDE mise with the Gaussian kernel vs bandwith values for different sample sizes
if (1==1):
    pu.plot_kde_mise_N(cfg, "kde_gaussian_mise_N", kde_figs, kde_csv, fig_save=True)


# plot KDE loglog of mise with the Gaussian kernel vs sample size
if (1==1):
    pu.plot_loglog_mise(cfg, "loglog_kde_low_mise_vs_h",  kde_figs, kde_csv, fig_save=True)

# ###########################################################################

# ############################################################################
if 1 == 1:
    df_lscv = lr.least_square_cross_validation(h_interval, myKde, X_sample, x_grid, f_exact)
    df_ise = lr.compute_ise(myGenerator, myKde, h_interval, cfg['N'], x_grid)
    save_df(df_lscv, "least_square_cv_normal_kde_1000_pts", kde_csv)
    save_df(df_ise, "integrated_sqrue_error_kde_1000_pts", kde_csv)

if 1 == 1:
    pu.plot_lscvh("lscv_plus_normf_squared", kde_figs, kde_csv, fig_save=True)

if 1 == 1:
    df_h_rot = lr.compute_rule_of_thumb_bandwidths(cfg['N_GRID'], myGenerator)
    save_df(df_h_rot, "rule_of_thumb_bandwidhs_vs_Ngrid", kde_csv)

if 1 == 1:
    lscv_hvalues = []
    nbr_iter = cfg['NBR_ITER']
    for n in cfg['N_GRID']:
        X_sample_n = myGenerator.sample(n)
        df_lscv_n = lr.least_square_cross_validation(h_interval, myKde, X_sample_n, x_grid, f_exact)
        h_val = df_lscv_n['h_grid'][np.argmin(df_lscv_n.lscv)]
        lscv_hvalues.append(h_val)
        save_df(df_lscv_n, f"{n}_pts_least_square_cv_normal_kde", kde_csv)

    df_lscv_bandwiths = pd.DataFrame({"N": cfg['N_GRID'], "lscv_hvalues": lscv_hvalues})
    save_df(df_lscv_bandwiths, "lscv_bandwith_vaules_vs_Ngird", kde_csv)

if 1 == 1:
    df_mise_compare = lr.compare_mise_vs_bwidth_types(kde_csv, cfg['N_GRID'])
    save_df(df_mise_compare, "compare_mise_with_bw_types", kde_csv)
    print(df_mise_compare.head())
