### Import statements ###
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import cvxpy as cp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from src.odece import ODECE
from ML.TorchML import LinearRegressionforSyn
from OptProblems.synthetic.syndataset import syn_dataset
from OptProblems.synthetic.synsolver import syn_solver
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os
import concurrent
import time
import csv
import pickle
import warnings
import torch
warnings.simplefilter('ignore', UserWarning)
warnings.simplefilter('ignore', RuntimeWarning)
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SYNTHETIC EXPERIMENT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

### Main steps of the experiment ###

# 1. Sensitivity analysis

def run_sensitivity_analysis_for_one_setting(setting, N_train, lambda_regs, gammas, alphas, feas_tol, opt_tol, base_dir):

    # Set directories
    setting_dir = os.path.join(base_dir, 'synthetic', f'setting_{setting}')
    model_dir = os.path.join(setting_dir, 'sensitivity_analysis', 'models')
    runtime_dir = os.path.join(setting_dir, 'sensitivity_analysis', 'runtime')
    os.makedirs(model_dir, exist_ok = True)
    os.makedirs(runtime_dir, exist_ok = True)

    # Read in setting data
    n, m, c, A, d, _, xi_train, training_reps, b_train, x_star_train, y_star_train, _, _, _, _, _ = read_setting_data(setting_dir)

    ### Train the models ###

    # 1. Primal-DAL using different values of lambda and gamma
    for lambda_reg in lambda_regs:
        for gamma in gammas:

            # Create runtime csv file for the current set of parameters
            primal_dal_dir = os.path.join(model_dir, f'primal_dal_N_train_{N_train}_lambda_{lambda_reg}_gamma_{gamma}')
            os.makedirs(primal_dal_dir, exist_ok = True)
            primal_dal_runtime_file = os.path.join(runtime_dir, f'primal_dal_N_train_{N_train}_lambda_{lambda_reg}_gamma_{gamma}.csv')
            with open(primal_dal_runtime_file, mode = 'a', newline = '') as file:
                writer = csv.writer(file)
                writer.writerow(['rep', 'N_train', 'soln_method', 'lambda_reg', 'gamma', 'num_iters', 'total_time', 'primal_dal_obj_val', 'obj_val_duality_gap', 'obj_val_reg', 'obj_val_pen'])

            # Run each replication in sequence
            for rep in training_reps:
                train_primal_dal_and_write_out(m, c, A, d, N_train, xi_train, b_train, x_star_train, rep, lambda_reg, gamma, 'acs', feas_tol, opt_tol, primal_dal_dir, primal_dal_runtime_file)

    # 2. Dual-DAL using different values of alpha
    for alpha in alphas:
            
        # Create runtime csv file for the current parameter
        dual_dal_dir = os.path.join(model_dir, f'dual_dal_N_train_{N_train}_alpha_{alpha}')
        os.makedirs(dual_dal_dir, exist_ok = True)
        dual_dal_runtime_file = os.path.join(runtime_dir, f'dual_dal_N_train_{N_train}_alpha_{alpha}.csv')
        with open(dual_dal_runtime_file, mode = 'a', newline = '') as file:
            writer = csv.writer(file)
            writer.writerow(['rep', 'N_train', 'alpha', 'status', 'total_time', 'dual_dal_obj_val', 'avg_duality_gap_w_abs'])

        # Run each replication in sequence
        for rep in training_reps:
            train_dual_dal_and_write_out(n, m, c, A, d, N_train, xi_train, b_train, x_star_train, y_star_train, rep, alpha, dual_dal_dir, dual_dal_runtime_file)
    
    print(f'Finished sensitivity analysis on setting # {setting}')

    return None

def generate_sensitivity_analysis_plots_for_one_setting(setting, N_train, N_test, lambda_regs, gammas, alphas, feas_tol, opt_tol, base_dir):

    model = 'primal_dal_gamma_0'
    params = [(lambda_reg, gammas[0]) for lambda_reg in lambda_regs]
    primal_dal_x_label = '$\lambda$'
    primal_dal_x_tick_labels = [f'{x:.0e}' for x in lambda_regs]
    primal_dal_gamma_0_y_label = None
    primal_dal_gamma_0_y_tick_labels = None
    generate_sensitivity_analysis_plots_for_one_model(setting, N_train, N_test, model, params, primal_dal_x_label, primal_dal_x_tick_labels, primal_dal_gamma_0_y_label, primal_dal_gamma_0_y_tick_labels, feas_tol, opt_tol, base_dir)
    
    model = 'primal_dal_gamma_pos'
    params = [(lambda_reg, gamma) for lambda_reg in lambda_regs for gamma in gammas[1 : ]]
    primal_dal_gamma_pos_y_label = '$\gamma$'
    primal_dal_gamma_pos_y_tick_labels = [f'{x:.0e}' for x in gammas[1 : ]]
    generate_sensitivity_analysis_plots_for_one_model(setting, N_train, N_test, model, params, primal_dal_x_label, primal_dal_x_tick_labels, primal_dal_gamma_pos_y_label, primal_dal_gamma_pos_y_tick_labels, feas_tol, opt_tol, base_dir)

    model = 'dual_dal'
    dual_dal_x_label = r'$\alpha$'
    dual_dal_x_tick_labels = [str(int(x)) if x == int(x) else str(x) for x in alphas]
    dual_dal_y_label = None
    dual_dal_y_tick_labels = None
    generate_sensitivity_analysis_plots_for_one_model(setting, N_train, N_test, model, alphas, dual_dal_x_label, dual_dal_x_tick_labels, dual_dal_y_label, dual_dal_y_tick_labels, feas_tol, opt_tol, base_dir)
    
    print(f'Generated sensitivity analysis plots for setting # {setting}')

def generate_sensitivity_analysis_plots_for_one_model(setting, N_train, N_test, model, params, x_label, x_tick_labels, y_label, y_tick_labels, feas_tol, opt_tol, base_dir):

    # Set directories
    setting_dir = os.path.join(base_dir, 'synthetic', f'setting_{setting}')
    model_dir = os.path.join(setting_dir, 'sensitivity_analysis', 'models')
    runtime_dir = os.path.join(setting_dir, 'sensitivity_analysis', 'runtime')
    os.makedirs(model_dir, exist_ok = True)
    os.makedirs(runtime_dir, exist_ok = True)

    # Read in data for this setting
    n, m, c, A, _, W_gt, xi_train, training_reps, b_train, x_star_train, y_star_train, xi_test, test_reps, b_test, x_star_test, y_star_test = read_setting_data(setting_dir)

    # Instantiate metric dictionaries for in-sample and out-of-sample (oos) data
    W_hat_sparsity, W_hat_minus_W_gt_0_norm, W_hat_minus_W_gt_2_norm, in_sample_prediction_error, in_sample_prediction_error_at_true_binding_constrs, in_sample_prediction_error_at_true_nonbinding_constrs, in_sample_is_underpredicting_at_all_constrs, in_sample_duality_gap, in_sample_proj_dist, in_sample_cost_opt_gap_proj_and_true_soln = [{} for _ in range(10)]
    oos_prediction_error, oos_prediction_error_at_true_binding_constrs, oos_prediction_error_at_true_nonbinding_constrs, oos_is_underpredicting_at_all_constrs, oos_is_underpredicting_at_true_binding_constrs, oos_is_underpredicting_at_true_nonbinding_constrs, is_predicted_feasible_region_nonempty, intersection, constr_satisfaction, feas, oos_duality_gap, oos_proj_dist, oos_cost_opt_gap_proj_and_true_soln = [{} for _ in range(13)]
    
    # Preprocessing in the case of the Primal-DAL with gamma = 0
    if type(params[0]) == tuple and params[0][1] == 0: # i.e., no penalty in the objective
        params = sorted({param[0] for param in params})

    ### Obtain in-sample metrics ###
    W_hat_dict = {}
    for rep in training_reps:
        for param in params:

            # Read in solution
            if model == 'primal_dal_gamma_0':
                W_hat = np.load(os.path.join(model_dir, f'primal_dal_N_train_{N_train}_lambda_{param}_gamma_0', f'W_hat_primal_dal_gamma_0_rep_{rep}.npy'))
            elif model == 'primal_dal_gamma_pos':
                W_hat = np.load(os.path.join(model_dir, f'primal_dal_N_train_{N_train}_lambda_{param[0]}_gamma_{param[1]}', f'W_hat_primal_dal_gamma_pos_rep_{rep}.npy'))
            else:
                W_hat = np.load(os.path.join(model_dir, f'dual_dal_N_train_{N_train}_alpha_{param}', f'W_hat_dual_dal_rep_{rep}.npy'))
            W_hat_dict[rep, param] = W_hat

            # Metrics regarding the obtained model W_hat
            W_hat_sparsity[rep, param] = np.sum(W_hat == 0)
            W_hat_minus_W_gt_0_norm[rep, param] = np.count_nonzero(W_hat - W_gt)
            W_hat_minus_W_gt_2_norm[rep, param] = np.linalg.norm(W_hat - W_gt)

            # Make predictions
            b_hat = make_predictions(W_hat, 'matrix', N_train, xi_train[rep])

            # Prediction errors
            in_sample_prediction_error[rep, param] = [b_hat[i][j] - b_train[rep][i, j] for i in range(N_train) for j in range(m)]
            in_sample_true_binding_constrs = [[j for j in range(m) if np.abs(A[j] @ x_star_train[rep][i] - b_train[rep][i, j]) <= feas_tol] for i in range(N_train)]
            in_sample_prediction_error_at_true_binding_constrs[rep, param] = [b_hat[i][j] - b_train[rep][i, j] for i in range(N_train) for j in in_sample_true_binding_constrs[i]]
            in_sample_prediction_error_at_true_nonbinding_constrs[rep, param] = [b_hat[i][j] - b_train[rep][i, j] for i in range(N_train) for j in range(m) if j not in in_sample_true_binding_constrs[i]]
            in_sample_is_underpredicting_at_all_constrs[rep, param] = [underprediction_indicator(W_hat, xi_train[rep][i], b_train[rep][i], range(m)) for i in range(N_train)]
            # NOTE: in_sample_prediction_error_at_true_binding_constrs might be empty for a given key because in_sample_true_binding_constrs[i] might be empty for all i

            # Optimality gap
            _, _, in_sample_duality_gap_temp = compute_true_soln_metrics(c, A, N_train, x_star_train[rep], y_star_train[rep], b_hat, feas_tol)
            in_sample_duality_gap[rep, param] = list(in_sample_duality_gap_temp.values())

            # Projection
            x_hat = np.zeros((N_train, n))
            for i in range(N_train):
                x_hat[i], _ = solve_downstream_lp(n, c, A, b_hat[i], feas_tol, opt_tol)
            in_sample_proj_dist_temp, in_sample_cost_opt_gap_proj_and_true_soln_temp = compute_predicted_soln_projection_metrics(c, A, N_train, b_train[rep], x_star_train[rep], x_hat, opt_tol)
            in_sample_proj_dist[rep, param] = list(in_sample_proj_dist_temp.values())
            in_sample_cost_opt_gap_proj_and_true_soln[rep, param] = list(in_sample_cost_opt_gap_proj_and_true_soln_temp.values())
    
    ### Obtain out-of-sample metrics ###
    for rep in test_reps:
        for param in params:

            # Grab corresponing model
            W_hat = W_hat_dict[rep, param]

            # Make predictions
            b_hat = make_predictions(W_hat, 'matrix', N_test, xi_test[rep])

            # Prediction errors
            oos_prediction_error[rep, param] = [b_hat[i][j] - b_test[rep][i, j] for i in range(N_test) for j in range(m)]
            oos_true_binding_constrs = [[j for j in range(m) if np.abs(A[j] @ x_star_test[rep][i] - b_test[rep][i, j]) <= feas_tol] for i in range(N_test)]
            oos_prediction_error_at_true_binding_constrs[rep, param] = [b_hat[i][j] - b_test[rep][i, j] for i in range(N_test) for j in oos_true_binding_constrs[i]]
            oos_prediction_error_at_true_nonbinding_constrs[rep, param] = [b_hat[i][j] - b_test[rep][i, j] for i in range(N_test) for j in range(m) if j not in oos_true_binding_constrs[i]]
            oos_is_underpredicting_at_all_constrs[rep, param] = [underprediction_indicator(W_hat, xi_test[rep][i], b_test[rep][i], range(m)) for i in range(N_test)]
            oos_is_underpredicting_at_true_binding_constrs[rep, param] = [underprediction_indicator(W_hat, xi_test[rep][i], b_test[rep][i], oos_true_binding_constrs[i]) for i in range(N_test)]
            oos_is_underpredicting_at_true_nonbinding_constrs[rep, param] = [underprediction_indicator(W_hat, xi_test[rep][i], b_test[rep][i], list(set(range(m)) - set(oos_true_binding_constrs[i]))) for i in range(N_test)]
            # NOTE: oos_prediction_error_at_true_binding_constrs might be empty for a given key because oos_true_binding_constrs[i] might be empty for all i

            # Intersection
            intersection[rep, param] = [intersection_indicator(n, A, b_test[rep][i], b_hat[i]) for i in range(N_test)]

            # Decision metrics
            constr_satisfaction_temp, feas[rep, param], oos_duality_gap_temp = compute_true_soln_metrics(c, A, N_test, x_star_test[rep], y_star_test[rep], b_hat, feas_tol)
            constr_satisfaction[rep, param] = list(constr_satisfaction_temp.values())
            oos_duality_gap[rep, param] = list(oos_duality_gap_temp.values())

            # See if the predicted downstream problems are feasible (in this case, this is equivalent to them having a finite optimal cost); also, we compute the projection metrics here
            is_predicted_feasible_region_nonempty_list = []
            x_hat = np.zeros((N_test, n))
            for i in range(N_test):
                x_hat[i], _ = solve_downstream_lp(n, c, A, b_hat[i], feas_tol, opt_tol)
                if np.isnan(x_hat[i]).any():
                    is_predicted_feasible_region_nonempty_list.append(0)
                else:
                    is_predicted_feasible_region_nonempty_list.append(1)
            is_predicted_feasible_region_nonempty[rep, param] = is_predicted_feasible_region_nonempty_list
            oos_proj_dist_temp, oos_cost_opt_gap_proj_and_true_soln_temp = compute_predicted_soln_projection_metrics(c, A, N_test, b_test[rep], x_star_test[rep], x_hat, opt_tol)
            oos_proj_dist[rep, param] = list(oos_proj_dist_temp.values())
            oos_cost_opt_gap_proj_and_true_soln[rep, param] = list(oos_cost_opt_gap_proj_and_true_soln_temp.values())

    ### Write out plots ###
    in_sample_folder = os.path.join(setting_dir, 'sensitivity_analysis', 'plots', f'{model}', 'in_sample')
    oos_folder = os.path.join(setting_dir, 'sensitivity_analysis', 'plots', f'{model}', 'oos')
    os.makedirs(in_sample_folder, exist_ok = True)
    os.makedirs(oos_folder, exist_ok = True)

    if model == 'primal_dal_gamma_pos':

        lambda_regs = sorted({param[0] for param in params})
        gammas = sorted({param[1] for param in params})

        # 1. In-sample metrics

        # Metrics regarding the obtained model W_hat
        generate_heatmap(lambda_regs, gammas, training_reps, W_hat_sparsity, x_label, x_tick_labels, y_label, y_tick_labels, in_sample_folder + '/W_hat_sparsity.png', for_every_sample = False)
        generate_heatmap(lambda_regs, gammas, training_reps, W_hat_minus_W_gt_0_norm, x_label, x_tick_labels, y_label, y_tick_labels, in_sample_folder + '/W_hat_minus_W_gt_0_norm.png', for_every_sample = False)
        generate_heatmap(lambda_regs, gammas, training_reps, W_hat_minus_W_gt_2_norm, x_label, x_tick_labels, y_label, y_tick_labels, in_sample_folder + '/W_hat_minus_W_gt_2_norm.png', for_every_sample = False)
        obj_val_duality_gap, obj_val_reg, obj_val_pen = generate_primal_dal_component_function_val_lists(N_train, lambda_regs, gammas, runtime_dir)
        generate_component_function_val_heatmap(lambda_regs, gammas, obj_val_duality_gap, x_label, x_tick_labels, y_label, y_tick_labels, in_sample_folder + '/obj_val_duality_gap.png')
        generate_component_function_val_heatmap(lambda_regs, gammas, obj_val_reg, x_label, x_tick_labels, y_label, y_tick_labels, in_sample_folder + '/obj_val_reg.png')
        generate_component_function_val_heatmap(lambda_regs, gammas, obj_val_pen, x_label, x_tick_labels, y_label, y_tick_labels, in_sample_folder + '/obj_val_pen.png')
        
        # Prediction errors
        generate_heatmap(lambda_regs, gammas, training_reps, in_sample_prediction_error, x_label, x_tick_labels, y_label, y_tick_labels, in_sample_folder + '/prediction_error.png', for_every_sample = True)
        generate_heatmap(lambda_regs, gammas, training_reps, in_sample_prediction_error_at_true_binding_constrs, x_label, x_tick_labels, y_label, y_tick_labels, in_sample_folder + '/prediction_error_at_true_binding_constrs.png', for_every_sample = True)
        generate_heatmap(lambda_regs, gammas, training_reps, in_sample_prediction_error_at_true_nonbinding_constrs, x_label, x_tick_labels, y_label, y_tick_labels, in_sample_folder + '/prediction_error_at_true_nonbinding_constrs.png', for_every_sample = True)
        is_underpredicting_all_constrs_dict = get_dict_of_avgs_from_two_params_dict(in_sample_is_underpredicting_at_all_constrs, lambda_regs, gammas, training_reps, N_train)
        generate_indicator_heatmap(lambda_regs, gammas, is_underpredicting_all_constrs_dict, x_label, x_tick_labels, y_label, y_tick_labels, in_sample_folder + '/is_underpredicting_at_all_constrs.png')

        # Optimality gap
        generate_heatmap(lambda_regs, gammas, training_reps, in_sample_duality_gap, x_label, x_tick_labels, y_label, y_tick_labels, in_sample_folder + '/duality_gap.png', for_every_sample = True)
        
        # Projection
        generate_heatmap(lambda_regs, gammas, training_reps, in_sample_proj_dist, x_label, x_tick_labels, y_label, y_tick_labels, in_sample_folder + '/proj_dist.png', for_every_sample = True)
        generate_heatmap(lambda_regs, gammas, training_reps, in_sample_cost_opt_gap_proj_and_true_soln, x_label, x_tick_labels, y_label, y_tick_labels, in_sample_folder + '/cost_opt_gap_proj_and_true_soln.png', for_every_sample = True)

        # 2. OOS metrics

        # Prediction errors
        generate_heatmap(lambda_regs, gammas, training_reps, oos_prediction_error, x_label, x_tick_labels, y_label, y_tick_labels, oos_folder + '/prediction_error.png', for_every_sample = True)
        generate_heatmap(lambda_regs, gammas, training_reps, oos_prediction_error_at_true_binding_constrs, x_label, x_tick_labels, y_label, y_tick_labels, oos_folder + '/prediction_error_at_true_binding_constrs.png', for_every_sample = True)
        generate_heatmap(lambda_regs, gammas, training_reps, oos_prediction_error_at_true_nonbinding_constrs, x_label, x_tick_labels, y_label, y_tick_labels, oos_folder + '/prediction_error_at_true_nonbinding_constrs.png', for_every_sample = True)
        oos_is_underpredicting_at_all_constrs_dict = get_dict_of_avgs_from_two_params_dict(oos_is_underpredicting_at_all_constrs, lambda_regs, gammas, test_reps, N_test)
        generate_indicator_heatmap(lambda_regs, gammas, oos_is_underpredicting_at_all_constrs_dict, x_label, x_tick_labels, y_label, y_tick_labels, oos_folder + '/is_underpredicting_at_all_constrs.png')
        oos_is_underpredicting_at_true_binding_constrs_dict = get_dict_of_avgs_from_two_params_dict(oos_is_underpredicting_at_true_binding_constrs, lambda_regs, gammas, test_reps, N_test)
        generate_indicator_heatmap(lambda_regs, gammas, oos_is_underpredicting_at_true_binding_constrs_dict, x_label, x_tick_labels, y_label, y_tick_labels, oos_folder + '/is_underpredicting_at_true_binding_constrs.png')
        oos_is_underpredicting_at_true_nonbinding_constrs_dict = get_dict_of_avgs_from_two_params_dict(oos_is_underpredicting_at_true_nonbinding_constrs, lambda_regs, gammas, test_reps, N_test)
        generate_indicator_heatmap(lambda_regs, gammas, oos_is_underpredicting_at_true_nonbinding_constrs_dict, x_label, x_tick_labels, y_label, y_tick_labels, oos_folder + '/is_underpredicting_at_true_nonbinding_constrs.png')

        # Predicted and true feasible regions
        is_predicted_feasible_region_nonempty_dict = get_dict_of_avgs_from_two_params_dict(is_predicted_feasible_region_nonempty, lambda_regs, gammas, test_reps, N_test)
        generate_indicator_heatmap(lambda_regs, gammas, is_predicted_feasible_region_nonempty_dict, x_label, x_tick_labels, y_label, y_tick_labels, oos_folder + '/is_predicted_feasible_region_nonempty.png')
        intersection_dict = get_dict_of_avgs_from_two_params_dict(intersection, lambda_regs, gammas, test_reps, N_test)
        generate_indicator_heatmap(lambda_regs, gammas, intersection_dict, x_label, x_tick_labels, y_label, y_tick_labels, oos_folder + '/intersection.png')
        feas_dict = get_dict_of_avgs_from_two_params_dict(feas, lambda_regs, gammas, test_reps, N_test)
        generate_indicator_heatmap(lambda_regs, gammas, feas_dict, x_label, x_tick_labels, y_label, y_tick_labels, oos_folder + '/feas.png')

        # Decision metrics
        generate_heatmap(lambda_regs, gammas, training_reps, constr_satisfaction, x_label, x_tick_labels, y_label, y_tick_labels, oos_folder + '/constr_satisfaction.png', for_every_sample = True)
        generate_heatmap(lambda_regs, gammas, training_reps, oos_duality_gap, x_label, x_tick_labels, y_label, y_tick_labels, oos_folder + '/duality_gap.png', for_every_sample = True)
        
        # Projection metrics
        generate_heatmap(lambda_regs, gammas, training_reps, oos_proj_dist, x_label, x_tick_labels, y_label, y_tick_labels, oos_folder + '/proj_dist.png', for_every_sample = True)
        generate_heatmap(lambda_regs, gammas, training_reps, oos_cost_opt_gap_proj_and_true_soln, x_label, x_tick_labels, y_label, y_tick_labels, oos_folder + '/cost_opt_gap_proj_and_true_soln.png', for_every_sample = True)
    
    else:

        # 1. In-sample metrics

        # Metrics regarding the obtained model W_hat
        generate_boxplots(params, training_reps, W_hat_sparsity, x_label, x_tick_labels, 'Number of zero components', in_sample_folder + '/W_hat_sparsity.png', for_every_sample = False)
        generate_boxplots(params, training_reps, W_hat_minus_W_gt_0_norm, x_label, x_tick_labels, r'$||\widehat{W} - W^\star||_0$', in_sample_folder + '/W_hat_minus_W_gt_0_norm.png', for_every_sample = False)
        generate_boxplots(params, training_reps, W_hat_minus_W_gt_2_norm, x_label, x_tick_labels, r'$||\widehat{W} - W^\star||_2$', in_sample_folder + '/W_hat_minus_W_gt_2_norm.png', for_every_sample = False)
        if model == 'primal_dal_gamma_0':
            obj_val_duality_gap, obj_val_reg, _ = generate_primal_dal_component_function_val_lists(N_train, params, [0], runtime_dir)
            generate_paired_boxplots(obj_val_duality_gap, obj_val_reg, params, x_label, x_tick_labels, [r'$\frac{1}{N}\sum_{i \in [N]}(\langle c, x_i^\star \rangle - \langle \widehat{W}\xi_i, \hat{y}_i \rangle)$', r'$\sum_{j \in [m]}\sum_{k \in [d]}|\widehat{w}_{jk}|$'], in_sample_folder + '/component_function_vals.png')
        else:
            dual_dal_obj_val, avg_duality_gap_w_abs = generate_dual_dal_component_function_val_lists(N_train, params, runtime_dir)
            generate_paired_boxplots(dual_dal_obj_val, avg_duality_gap_w_abs, params, x_label, x_tick_labels, [r'$\frac{1}{N}\sum_{i \in [N]}(\langle c, \hat{x}_i \rangle - \langle \alpha \widehat{W}\xi_i - b_i, y_i^\star \rangle)$', r'$\frac{1}{N}\sum_{i \in [N]}|\langle c, x_i^\star \rangle - \langle \widehat{W}\xi_i, y_i^\star \rangle|$'], in_sample_folder + '/component_function_vals.png')

        # Prediction errors
        generate_tripled_boxplots(in_sample_prediction_error, in_sample_prediction_error_at_true_binding_constrs, in_sample_prediction_error_at_true_nonbinding_constrs, params, x_label, x_tick_labels, [r'$\langle \hat{w}_j, \xi_i \rangle - b_{ij}$', r'$\langle \hat{w}_j, \xi_i \rangle - b_{ij}$ for $j \in J^=(x_i^\star)$', r'$\langle \hat{w}_j, \xi_i \rangle - b_{ij}$ for $j \in [m] \setminus J^=(x_i^\star)$'], in_sample_folder + '/prediction_error.png')
        is_underpredicting_all_constrs_list = get_list_of_avgs_from_dict(in_sample_is_underpredicting_at_all_constrs, params, training_reps, N_train)
        generate_lineplot(params, is_underpredicting_all_constrs_list, x_label, x_tick_labels, r'$\chi\{\langle \hat{w}_j, \xi_i \rangle \leq b_{ij} ~ \forall j \in [m]\} ~ \text{(percentage)}$', in_sample_folder + '/is_underpredicting_at_all_constrs.png')

        # Optimality gap
        generate_boxplots(params, training_reps, in_sample_duality_gap, x_label, x_tick_labels, r'$\langle c, x_i^\star \rangle - \langle \widehat{W}\xi_i, y_i^\star \rangle$', in_sample_folder + '/duality_gap.png', for_every_sample = True)
        
        # Projection
        generate_boxplots(params, training_reps, in_sample_proj_dist, x_label, x_tick_labels, r'$||\hat{x}_i - \tilde{x}_i||_2$', in_sample_folder + '/proj_dist.png', for_every_sample = True)
        in_sample_cost_opt_gap_proj_and_true_soln_median = get_list_of_medians_from_dict(in_sample_cost_opt_gap_proj_and_true_soln, params, training_reps)
        generate_lineplot(params, in_sample_cost_opt_gap_proj_and_true_soln_median, x_label, x_tick_labels, r'$\langle c, \tilde{x}_i \rangle - \langle c, x^\star_i \rangle ~ \text{(normalized)}$', in_sample_folder + '/cost_opt_gap_proj_and_true_soln.png')

        # 2. OOS metrics

        # Prediction errors
        generate_tripled_boxplots(oos_prediction_error, oos_prediction_error_at_true_binding_constrs, oos_prediction_error_at_true_nonbinding_constrs, params, x_label, x_tick_labels, [r'$\langle \hat{w}_j, \xi_i \rangle - b_{ij}$', r'$\langle \hat{w}_j, \xi_i \rangle - b_{ij}$ for $j \in J^=(x_i^\star)$', r'$\langle \hat{w}_j, \xi_i \rangle - b_{ij}$ for $j \in [m] \setminus J^=(x_i^\star)$'], oos_folder + '/prediction_error.png')
        is_underpredicting_at_all_constrs_list = get_list_of_avgs_from_dict(oos_is_underpredicting_at_all_constrs, params, test_reps, N_test)
        is_underpredicting_at_true_binding_constrs_list = get_list_of_avgs_from_dict(oos_is_underpredicting_at_true_binding_constrs, params, test_reps, N_test)
        is_underpredicting_at_true_nonbinding_constrs_list = get_list_of_avgs_from_dict(oos_is_underpredicting_at_true_nonbinding_constrs, params, test_reps, N_test)
        generate_tripled_lineplots(params, is_underpredicting_at_all_constrs_list, is_underpredicting_at_true_binding_constrs_list, is_underpredicting_at_true_nonbinding_constrs_list, r'$\chi\{\langle \hat{w}_j, \xi_i \rangle \leq b_{ij} ~ \forall j \in [m]\} ~ \text{(percentage)}$', r'$\chi\{\langle \hat{w}_j, \xi_i \rangle \leq b_{ij} ~ \forall j \in J^=(x_i^\star)\} ~ \text{(percentage)}$', r'$\chi\{\langle \hat{w}_j, \xi_i \rangle \leq b_{ij} ~ \forall j \in [m] \setminus J^=(x_i^\star)\} ~ \text{(percentage)}$', x_label, x_tick_labels, oos_folder + '/predicted_and_true_feasible_regions.png')

        # Predicted and true feasible regions
        is_predicted_feasible_region_nonempty_list = get_list_of_avgs_from_dict(is_predicted_feasible_region_nonempty, params, test_reps, N_test)
        intersection_list = get_list_of_avgs_from_dict(intersection, params, test_reps, N_test)
        feas_list = get_list_of_avgs_from_dict(feas, params, test_reps, N_test)
        generate_tripled_lineplots(params, is_predicted_feasible_region_nonempty_list, intersection_list, feas_list, r'$\chi\{X(\widehat{W}\xi_i) \neq \emptyset\} ~ \text{(percentage)}$', r'$\chi\{X(b_i) \cap X(\widehat{W}\xi_i) \neq \emptyset\} ~ \text{(percentage)}$', r'$\chi\{x_i^\star \in X(\widehat{W}\xi_i)\} ~ \text{(percentage)}$', x_label, x_tick_labels, oos_folder + '/predicted_and_true_feasible_regions.png')

        # Decision metrics
        generate_boxplots_and_stripplots(params, test_reps, constr_satisfaction, x_label, x_tick_labels, r'$\sum_{j \in [m]}\chi\{\langle a_j, x_i^\star \rangle \geq \langle \hat{w}_j, \xi_i \rangle\}$', oos_folder + '/constr_satisfaction.png')
        generate_boxplots(params, test_reps, oos_duality_gap, x_label, x_tick_labels, r'$\langle c, x_i^\star \rangle - \langle \widehat{W}\xi_i, y_i^\star \rangle$', oos_folder + '/duality_gap.png', for_every_sample = True)

        # Projection metrics
        generate_boxplots(params, test_reps, oos_proj_dist, x_label, x_tick_labels, r'$||\hat{x}_i - \tilde{x}_i||_2$', oos_folder + '/proj_dist.png', for_every_sample = True)
        oos_cost_opt_gap_proj_and_true_soln_median = get_list_of_medians_from_dict(oos_cost_opt_gap_proj_and_true_soln, params, test_reps)
        generate_lineplot(params, oos_cost_opt_gap_proj_and_true_soln_median, x_label, x_tick_labels, r'$\langle c, \tilde{x}_i \rangle - \langle c, x^\star_i \rangle ~ \text{(normalized)}$', oos_folder + '/cost_opt_gap_proj_and_true_soln.png')

    return None

def get_list_of_avgs_from_dict(input_dict, params, reps, N):

    avgs_by_param = []
    for param in params:
        fixed_param_list = [input_dict[rep, param][i] for rep in reps for i in range(N)]
        avgs_by_param.append(100 * np.mean(fixed_param_list))

    return avgs_by_param

def get_list_of_medians_from_dict(input_dict, params, reps):

    medians_by_param = []
    for param in params:
        fixed_param_list = [x for rep in reps for x in input_dict[rep, param]]
        medians_by_param.append(np.median(fixed_param_list))

    return medians_by_param

def get_dict_of_avgs_from_two_params_dict(input_dict, lambda_regs, gammas, reps, N):

    avgs_by_param = {}
    for lambda_reg in lambda_regs:
        for gamma in gammas:
            fixed_param_list = [input_dict[rep, (lambda_reg, gamma)][i] for rep in reps for i in range(N)]
            avgs_by_param[lambda_reg, gamma] = 100 * np.mean(fixed_param_list)

    return avgs_by_param

def underprediction_indicator(W, xi, b, constr_indices):

    if all(W[j] @ xi <= b[j] for j in constr_indices): # NOTE: this condition holds vacuously if constr_indices is empty
        return 1
    else:
        return 0

def intersection_indicator(n, A, b, b_hat):

    model = gp.Model()
    x = model.addMVar(n, lb = 0, name = 'x')
    model.setObjective(0, GRB.MINIMIZE)
    model.addConstr(A @ x >= b)
    model.addConstr(A @ x >= b_hat)
    model.update()
    model.optimize()

    if model.status == GRB.OPTIMAL:
        return 1
    else:
        return 0

def get_normalized_cost_opt_gap(c, x, x_star, opt_tol):

    if np.abs(c @ x_star) > opt_tol:
        return (c @ x - c @ x_star)/np.abs(c @ x_star) # normalized cost optimality gap
    else:
        return c @ x - c @ x_star

def convert_dict_to_df(input_dict):

    rows = []
    for k, values in input_dict.items():
        for v in values:
            rows.append({'param': k, 'value': v})
    df = pd.DataFrame(rows)

    return df
    
def generate_primal_dal_component_function_val_lists(N_train, lambda_regs, gammas, runtime_dir):

    obj_val_duality_gap = {}
    obj_val_reg = {}
    obj_val_pen = {}
    for lambda_reg in lambda_regs:
        for gamma in gammas:
            df = pd.read_csv(os.path.join(runtime_dir, f'primal_dal_N_train_{N_train}_lambda_{lambda_reg}_gamma_{gamma}.csv'))
            obj_val_duality_gap[lambda_reg, gamma] = df['obj_val_duality_gap'].values
            obj_val_reg[lambda_reg, gamma] = df['obj_val_reg'].values
            obj_val_pen[lambda_reg, gamma] = df['obj_val_pen'].values

    return obj_val_duality_gap, obj_val_reg, obj_val_pen

def generate_dual_dal_component_function_val_lists(N_train, alphas, runtime_dir):

    dual_dal_obj_val = {}
    avg_duality_gap_w_abs = {}
    for alpha in alphas:
        df = pd.read_csv(os.path.join(runtime_dir, f'dual_dal_N_train_{N_train}_alpha_{alpha}.csv'))
        dual_dal_obj_val[alpha] = df['dual_dal_obj_val'].values
        avg_duality_gap_w_abs[alpha] = df['avg_duality_gap_w_abs'].values

    return dual_dal_obj_val, avg_duality_gap_w_abs

def generate_boxplots(params, reps, input_dict, x_label, x_tick_labels, y_label, output_file, for_every_sample):

    # Gather data into pandas dataframe
    metric = {}
    for param in params:
        fixed_param = []
        for rep in reps:
            if for_every_sample:
                fixed_param += input_dict[rep, param]
            else:
                fixed_param.append(input_dict[rep, param])
        metric[param] = fixed_param
    df = convert_dict_to_df(metric)

    if not df.empty:

        # Plot the boxplots
        plt.figure(figsize = (10, 5))
        sns.boxplot(data = df, x = 'param', y = 'value')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.tight_layout()
        plt.xticks(ticks = range(len(x_tick_labels)), labels = x_tick_labels)
        plt.savefig(output_file, dpi = 300, bbox_inches = 'tight')
        plt.close()

    return None

def generate_boxplots_and_stripplots(params, reps, input_dict, x_label, x_tick_labels, y_label, output_file):

    # Gather the data into a pandas dataframe
    metric = {}
    for param in params:
        fixed_param = []
        for rep in reps:
            fixed_param += input_dict[rep, param]
        metric[param] = fixed_param
    df = convert_dict_to_df(metric)
    
    if not df.empty:
        # Plot the boxplot and stripplot
        plt.figure(figsize = (10, 5))
        sns.boxplot(data = df, x = 'param', y = 'value', whis = 1.5, fliersize = 0)
        sns.stripplot(data = df, x = 'param', y = 'value', color = 'black', alpha = 0.5, jitter = 0.25)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.tight_layout()
        plt.xticks(ticks = range(len(x_tick_labels)), labels = x_tick_labels)
        plt.savefig(output_file, dpi = 300, bbox_inches = 'tight')
        plt.close()

    return None

def generate_paired_boxplots(input_dict_1, input_dict_2, labels, x_label, x_tick_labels, legend_labels, output_file):
    
    # Gather data into pandas dataframe
    keys = list(input_dict_1.keys())
    rows = []
    for key, label in zip(keys, labels):
        rows += [(label, 'metric_1', v) for v in input_dict_1[key]]
        rows += [(label, 'metric_2', v) for v in input_dict_2[key]]
    df = pd.DataFrame(rows, columns = ['param', 'metric', 'value'])

    if not df.empty:

        # Plot paired boxplots
        plt.figure(figsize = (10, 5))
        ax = sns.boxplot(data = df, x = 'param', y = 'value', hue = 'metric')
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, legend_labels, title = None)
        plt.tight_layout()
        plt.xlabel(x_label)
        plt.ylabel('')
        plt.xticks(ticks = range(len(x_tick_labels)), labels = x_tick_labels)
        plt.savefig(output_file, dpi = 300, bbox_inches = 'tight')
        plt.close()

    return None

def generate_tripled_boxplots(input_dict_1, input_dict_2, input_dict_3, labels, x_label, x_tick_labels, legend_labels, output_file):
    
    # Gather data into pandas dataframe
    keys = list(input_dict_1.keys())
    rows = []
    for key, label in zip(keys, labels):
        rows += [(label, 'metric_1', v) for v in input_dict_1[key]]
        rows += [(label, 'metric_2', v) for v in input_dict_2[key]]
        rows += [(label, 'metric_3', v) for v in input_dict_3[key]]
    df = pd.DataFrame(rows, columns = ['param', 'metric', 'value'])

    if not df.empty:

        # Plot paired boxplots
        plt.figure(figsize = (10, 5))
        ax = sns.boxplot(data = df, x = 'param', y = 'value', hue = 'metric')
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, legend_labels, title = None)
        plt.tight_layout()
        plt.xlabel(x_label)
        plt.ylabel('')
        plt.xticks(ticks = range(len(x_tick_labels)), labels = x_tick_labels)
        plt.savefig(output_file, dpi = 300, bbox_inches = 'tight')
        plt.close()

    return None

def generate_lineplot(params, input_list, x_label, x_tick_labels, y_label, output_file):

    plt.figure(figsize = (10, 5))
    plt.plot(range(len(params)), input_list, marker = 'o')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    plt.xticks(ticks = range(len(x_tick_labels)), labels = x_tick_labels)
    plt.tight_layout()
    plt.savefig(output_file, dpi = 300, bbox_inches = 'tight')
    plt.close()

    return None

def generate_tripled_lineplots(params, input_list_1, input_list_2, input_list_3, input_list_1_legend, input_list_2_legend, input_list_3_legend, x_label, x_tick_labels, output_file):

    plt.figure(figsize = (10, 5))
    plt.plot(range(len(params)), input_list_1, marker = 'o', label = input_list_1_legend)
    plt.plot(range(len(params)), input_list_2, marker = 'o', label = input_list_2_legend)
    plt.plot(range(len(params)), input_list_3, marker = 'o', label = input_list_3_legend)
    plt.xlabel(x_label)
    plt.legend()
    plt.grid(True)
    plt.xticks(ticks = range(len(x_tick_labels)), labels = x_tick_labels)
    plt.tight_layout()
    plt.savefig(output_file, dpi = 300, bbox_inches = 'tight')
    plt.close()

    return None

def plot_heatmap_data(heatmap_data, x_label, x_tick_labels, y_label, y_tick_labels, output_file):

    plt.imshow(heatmap_data, origin = 'lower', cmap = 'viridis')
    plt.colorbar()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(range(len(x_tick_labels)), x_tick_labels)
    plt.yticks(range(len(y_tick_labels)), y_tick_labels)
    plt.savefig(output_file, dpi = 300, bbox_inches = 'tight')
    plt.close()

    return None

def generate_heatmap(lambda_regs, gammas, reps, input_dict, x_label, x_tick_labels, y_label, y_tick_labels, output_file, for_every_sample):

    # Gather data into a pandas dataframe
    metric = {}
    for lambda_reg in lambda_regs:
        for gamma in gammas:
            fixed_param_list = []
            for rep in reps:
                if for_every_sample:
                    fixed_param_list += input_dict[rep, (lambda_reg, gamma)]
                else:
                    fixed_param_list.append(input_dict[rep, (lambda_reg, gamma)])
            metric[lambda_reg, gamma] = fixed_param_list
    df = convert_dict_to_df(metric)
    
    if not df.empty:
    
        # Plot the heatmap
        heatmap_data = np.array([[np.median(df.loc[df['param'] == (lambda_reg, gamma), 'value'].values) for lambda_reg in lambda_regs] for gamma in gammas])
        plot_heatmap_data(heatmap_data, x_label, x_tick_labels, y_label, y_tick_labels, output_file)

    return None

def generate_component_function_val_heatmap(lambda_regs, gammas, input_dict, x_label, x_tick_labels, y_label, y_tick_labels, output_file):

    # Gather data into pandas dataframe
    df = convert_dict_to_df(input_dict)

    # Plot the heatmap
    heatmap_data = np.array([[np.median(df.loc[df['param'] == (lambda_reg, gamma), 'value'].values) for lambda_reg in lambda_regs] for gamma in gammas])
    plot_heatmap_data(heatmap_data, x_label, x_tick_labels, y_label, y_tick_labels, output_file)

    return None

def generate_indicator_heatmap(lambda_regs, gammas, input_dict, x_label, x_tick_labels, y_label, y_tick_labels, output_file):
    
    heatmap_data = np.array([[input_dict[lambda_reg, gamma] for lambda_reg in lambda_regs] for gamma in gammas])
    plot_heatmap_data(heatmap_data, x_label, x_tick_labels, y_label, y_tick_labels, output_file)

    return None

# 2. Solution method comparison

def run_primal_dal_soln_method_comparison_for_one_setting(setting, N_train, lambda_reg, gamma, feas_tol, opt_tol, base_dir):

    # Set directories and create runtime csv file
    setting_dir = os.path.join(base_dir, 'synthetic', f'setting_{setting}')
    primal_dal_soln_method_comparison_dir = os.path.join(setting_dir, 'primal_dal_soln_method_comparison')
    os.makedirs(primal_dal_soln_method_comparison_dir, exist_ok = True)
    primal_dal_runtime_file = os.path.join(primal_dal_soln_method_comparison_dir, f'primal_dal_N_train_{N_train}_runtime.csv')
    with open(primal_dal_runtime_file, mode = 'a', newline = '') as file:
        writer = csv.writer(file)
        writer.writerow(['rep', 'N_train', 'soln_method', 'lambda_reg', 'gamma', 'num_iters_or_status', 'total_time', 'primal_dal_obj_val', 'obj_val_duality_gap', 'obj_val_reg', 'obj_val_pen'])

    # Read in setting data
    _, m, c, A, d, _, xi_train, training_reps, b_train, x_star_train, _, _, _, _, _, _ = read_setting_data(setting_dir)
    
    # Solve the Primal-DAL problem using different solution methods
    for soln_method in ['acs', 'ccp', 'gurobi']:

        # Set model directory for the current solution method
        primal_dal_dir = os.path.join(primal_dal_soln_method_comparison_dir, 'models', f'primal_dal_N_train_{N_train}_lambda_{lambda_reg}_gamma_{gamma}_{soln_method}')
        os.makedirs(primal_dal_dir, exist_ok = True)
        
        # Train over different replications in sequence
        for rep in training_reps:
            train_primal_dal_and_write_out(m, c, A, d, N_train, xi_train, b_train, x_star_train, rep, lambda_reg, gamma, soln_method, feas_tol, opt_tol, primal_dal_dir, primal_dal_runtime_file)
    
    print(f'Finished comparison on setting # {setting}')

    return None

# 3. Synthetic experiment

def run_synthetic_experiment_for_one_setting(setting, N_train, N_test, lambda_regs, gammas, alphas, infeasibility_aversion_coefs, deltas, feas_tol, opt_tol, base_dir):

    # Grab setting directory
    setting_dir = os.path.join(base_dir, 'synthetic', f'setting_{setting}')

    # Read in setting data
    n, m, c, A, d, _, xi_train, training_reps, b_train, x_star_train, y_star_train, xi_test, test_reps, b_test, x_star_test, y_star_test = read_setting_data(setting_dir)

    ### Tune models ###
    synthetic_experiment_dir = os.path.join(setting_dir, 'synthetic_experiment')
    tuned_params_dir = os.path.join(synthetic_experiment_dir, 'tuned_params')
    os.makedirs(tuned_params_dir, exist_ok = True)
    
    # 1a. Primal-DAL (gamma = 0)
    params = [(lambda_reg, gammas[0]) for lambda_reg in lambda_regs]
    tuned_lambda_regs_gamma_0 = tune_model(n, m, c, A, d, N_train, xi_train, b_train, x_star_train, y_star_train, 'primal_dal', params, training_reps, feas_tol, opt_tol)
    tuned_lambda_regs_gamma_0 = {key : value[0] for key, value in tuned_lambda_regs_gamma_0.items()} # grab the lambda_reg parameter
    tuned_lambda_regs_gamma_0_file = os.path.join(tuned_params_dir, 'tuned_lambda_regs_gamma_0.pkl')
    write_pickle_file(tuned_lambda_regs_gamma_0_file, tuned_lambda_regs_gamma_0)

    print('Tuned Primal-DAL (gamma = 0)')

    # 1b. Primal-DAL (gamma > 0)
    params = [(lambda_reg, gamma) for lambda_reg in lambda_regs for gamma in gammas[1 : ]] # here, we do not train with gamma = 0
    tuned_lambda_regs_and_gammas = tune_model(n, m, c, A, d, N_train, xi_train, b_train, x_star_train, y_star_train, 'primal_dal', params, training_reps, feas_tol, opt_tol)
    tuned_lambda_regs_and_gammas_file = os.path.join(tuned_params_dir, 'tuned_lambda_regs_and_gammas.pkl')
    write_pickle_file(tuned_lambda_regs_and_gammas_file, tuned_lambda_regs_and_gammas)

    print('Tuned Primal-DAL (gamma > 0)')

    # 2. Dual-DAL
    tuned_alphas = tune_model(n, m, c, A, d, N_train, xi_train, b_train, x_star_train, y_star_train, 'dual_dal', alphas, training_reps, feas_tol, opt_tol)
    tuned_alphas_file = os.path.join(tuned_params_dir, 'tuned_alphas.pkl')
    write_pickle_file(tuned_alphas_file, tuned_alphas)

    print('Tuned Dual-DAL')

    # 3. ODECE
    tuned_infeasibility_aversion_coefs = tune_model(n, m, c, A, d, N_train, xi_train, b_train, x_star_train, y_star_train, 'odece', infeasibility_aversion_coefs, training_reps, feas_tol, opt_tol)
    tuned_infeasibility_aversion_coefs_file = os.path.join(tuned_params_dir, 'tuned_infeasibility_aversion_coefs.pkl')
    write_pickle_file(tuned_infeasibility_aversion_coefs_file, tuned_infeasibility_aversion_coefs)

    print('Tuned ODECE')

    # 4. Lasso
    tuned_deltas = tune_model(n, m, c, A, d, N_train, xi_train, b_train, x_star_train, y_star_train, 'lasso', deltas, training_reps, feas_tol, opt_tol)
    tuned_deltas_file = os.path.join(tuned_params_dir, 'tuned_deltas.pkl')
    write_pickle_file(tuned_deltas_file, tuned_deltas)

    print('Tuned Lasso')

    print('Finished hyperparameter tuning')

    ### Train models over various subsets of the training dataset ###
    for N_train_frac in [int((i/4) * N_train) for i in range(1, 5)]:

        # Create model and runtime directories for the models at the current level of training data considered
        model_dir = os.path.join(synthetic_experiment_dir, 'models', f'N_train_{N_train_frac}')
        runtime_dir = os.path.join(synthetic_experiment_dir, 'runtime', f'N_train_{N_train_frac}')
        os.makedirs(model_dir, exist_ok = True)
        os.makedirs(runtime_dir, exist_ok = True)

        # Create runtime csv files
        optimistic_dal_runtime_file = create_runtime_file_for_a_model('optimistic_dal', ['rep', 'N_train', 'status', 'total_time', 'optimistic_dal_obj_val'], runtime_dir)
        primal_dal_gamma_0_runtime_file = create_runtime_file_for_a_model('primal_dal_gamma_0', ['rep', 'N_train', 'soln_method', 'lambda_reg', 'gamma', 'num_iters', 'total_time', 'primal_dal_obj_val', 'obj_val_duality_gap', 'obj_val_reg', 'obj_val_pen'], runtime_dir)
        primal_dal_gamma_pos_runtime_file = create_runtime_file_for_a_model('primal_dal_gamma_pos', ['rep', 'N_train', 'soln_method', 'lambda_reg', 'gamma', 'num_iters', 'total_time', 'primal_dal_obj_val', 'obj_val_duality_gap', 'obj_val_reg', 'obj_val_pen'], runtime_dir)
        dual_dal_runtime_file = create_runtime_file_for_a_model('dual_dal', ['rep', 'N_train', 'alpha', 'status', 'total_time', 'dual_dal_obj_val', 'avg_duality_gap_w_abs'], runtime_dir)
        odece_runtime_file = create_runtime_file_for_a_model('odece', ['rep', 'N_train', 'infeasibility_aversion_coef', 'total_time'], runtime_dir)
        lr_runtime_file = create_runtime_file_for_a_model('lr', ['rep', 'N_train', 'total_time'], runtime_dir)
        lasso_runtime_file = create_runtime_file_for_a_model('lasso', ['rep', 'N_train', 'delta', 'total_time'], runtime_dir)
        rf_runtime_file = create_runtime_file_for_a_model('rf', ['rep', 'N_train', 'total_time'], runtime_dir)

        for rep in training_reps:
            
            print('(N_train_frac, rep) = ', (N_train_frac, rep))

            train_optimistic_dal_and_write_out(m, c, A, d, N_train_frac, xi_train, x_star_train, y_star_train, rep, model_dir, optimistic_dal_runtime_file)
            train_primal_dal_and_write_out(m, c, A, d, N_train_frac, xi_train, b_train, x_star_train, rep, tuned_lambda_regs_gamma_0[N_train_frac], 0, 'acs', feas_tol, opt_tol, model_dir, primal_dal_gamma_0_runtime_file)
            train_primal_dal_and_write_out(m, c, A, d, N_train_frac, xi_train, b_train, x_star_train, rep, tuned_lambda_regs_and_gammas[N_train_frac][0], tuned_lambda_regs_and_gammas[N_train_frac][1], 'acs', feas_tol, opt_tol, model_dir, primal_dal_gamma_pos_runtime_file)
            train_dual_dal_and_write_out(n, m, c, A, d, N_train_frac, xi_train, b_train, x_star_train, y_star_train, rep, tuned_alphas[N_train_frac], model_dir, dual_dal_runtime_file)
            train_odece_and_write_out(n, m, c, A, d, N_train_frac, xi_train, b_train, rep, tuned_infeasibility_aversion_coefs[N_train_frac], model_dir, odece_runtime_file)
            train_regression_model_and_write_out(N_train_frac, xi_train, b_train, rep, 'lr', model_dir, lr_runtime_file)
            train_regression_model_and_write_out(N_train_frac, xi_train, b_train, rep, 'lasso', model_dir, lasso_runtime_file, tuned_deltas = tuned_deltas)
            train_regression_model_and_write_out(N_train_frac, xi_train, b_train, rep, 'rf', model_dir, rf_runtime_file, d = d)
            # NOTE: the above 'train_regression_model_and_write_out' function is only well-defined when the model_name is 'lr', 'lasso', or 'rf'
    
    print('Trained all of the models')

    ### Test models ###
    for N_train_frac in [int((i/4) * N_train) for i in range(1, 5)]:

        # Set results directory
        results_dir = os.path.join(synthetic_experiment_dir, 'results', f'N_train_{N_train_frac}')
        os.makedirs(results_dir, exist_ok = True)

        for rep in test_reps:

            print('(N_train_frac, rep) = ', (N_train_frac, rep))

            # Read in models
            optimistic_dal_file = os.path.join(synthetic_experiment_dir, 'models', f'N_train_{N_train_frac}', f'W_hat_optimistic_dal_rep_{rep}.npy')
            W_hat_optimistic_dal = np.load(optimistic_dal_file)

            primal_dal_gamma_0_file = os.path.join(synthetic_experiment_dir, 'models', f'N_train_{N_train_frac}', f'W_hat_primal_dal_gamma_0_rep_{rep}.npy')
            W_hat_primal_dal_gamma_0 = np.load(primal_dal_gamma_0_file)

            primal_dal_gamma_pos_file = os.path.join(synthetic_experiment_dir, 'models', f'N_train_{N_train_frac}', f'W_hat_primal_dal_gamma_pos_rep_{rep}.npy')
            W_hat_primal_dal_gamma_pos = np.load(primal_dal_gamma_pos_file)

            dual_dal_file = os.path.join(synthetic_experiment_dir, 'models', f'N_train_{N_train_frac}', f'W_hat_dual_dal_rep_{rep}.npy')
            W_hat_dual_dal = np.load(dual_dal_file)

            odece_file = os.path.join(synthetic_experiment_dir, 'models', f'N_train_{N_train_frac}', f'odece_rep_{rep}.pkl')
            odece = read_pickle_file(odece_file)

            lr_file = os.path.join(synthetic_experiment_dir, 'models', f'N_train_{N_train_frac}', f'W_hat_lr_rep_{rep}.pkl')
            W_hat_lr = read_pickle_file(lr_file)

            lasso_file = os.path.join(synthetic_experiment_dir, 'models', f'N_train_{N_train_frac}', f'W_hat_lasso_rep_{rep}.pkl')
            W_hat_lasso = read_pickle_file(lasso_file)

            rf_file = os.path.join(synthetic_experiment_dir, 'models', f'N_train_{N_train_frac}', f'rf_rep_{rep}.pkl')
            rf = read_pickle_file(rf_file)

            evaluate_model(n, m, c, A, N_test, xi_test, b_test, x_star_test, y_star_test, rep, 'optimistic_dal', W_hat_optimistic_dal, feas_tol, opt_tol, results_dir)
            evaluate_model(n, m, c, A, N_test, xi_test, b_test, x_star_test, y_star_test, rep, 'primal_dal_gamma_0', W_hat_primal_dal_gamma_0, feas_tol, opt_tol, results_dir)
            evaluate_model(n, m, c, A, N_test, xi_test, b_test, x_star_test, y_star_test, rep, 'primal_dal_gamma_pos', W_hat_primal_dal_gamma_pos, feas_tol, opt_tol, results_dir)
            evaluate_model(n, m, c, A, N_test, xi_test, b_test, x_star_test, y_star_test, rep, 'dual_dal', W_hat_dual_dal, feas_tol, opt_tol, results_dir)
            evaluate_model(n, m, c, A, N_test, xi_test, b_test, x_star_test, y_star_test, rep, 'odece', odece, feas_tol, opt_tol, results_dir)
            evaluate_model(n, m, c, A, N_test, xi_test, b_test, x_star_test, y_star_test, rep, 'lr', W_hat_lr, feas_tol, opt_tol, results_dir)
            evaluate_model(n, m, c, A, N_test, xi_test, b_test, x_star_test, y_star_test, rep, 'lasso', W_hat_lasso, feas_tol, opt_tol, results_dir)
            evaluate_model(n, m, c, A, N_test, xi_test, b_test, x_star_test, y_star_test, rep, 'rf', rf, feas_tol, opt_tol, results_dir)

    # print('Tested all of the models')
    print(f'Finished synthetic experiment on setting # {setting}')

    return None

def tune_model(n, m, c, A, d, N, xi, b, x_star, y_star, model_name, params, training_reps, feas_tol, opt_tol):

    cv_loss = {}
    pooled_cv_loss = {}
    tuned_param = {}

    for N_frac in [int((i/4) * N) for i in range(1, 5)]:
        for param in params:
            for rep in training_reps:

                # Perform cross-validation
                fold_loss = {}
                kf = KFold(n_splits = 5, shuffle = True, random_state = 0)
                for fold, (train_idx, holdout_idx) in enumerate(kf.split(xi[rep][ : N_frac]), 1):

                    # Split the training data into cross-validation training and cross-validation holdout datasets
                    xi_cv_train, xi_cv_holdout = xi[rep][train_idx], xi[rep][holdout_idx]
                    b_cv_train, b_cv_holdout = b[rep][train_idx], b[rep][holdout_idx]
                    x_star_cv_train, x_star_cv_holdout = x_star[rep][train_idx], x_star[rep][holdout_idx]
                    y_star_cv_train, y_star_cv_holdout = y_star[rep][train_idx], y_star[rep][holdout_idx]

                    # Train the model on the current fold's training dataset
                    if model_name == 'primal_dal':
                        W_0_acs, y_0_acs, _ = generate_primal_dal_initial_solution(m, c, A, d, len(train_idx), xi_cv_train, x_star_cv_train, feas_tol, opt_tol)
                        W_hat, _, _ = primal_dal_acs(m, c, A, d, W_0_acs, len(train_idx), y_0_acs, xi_cv_train, b_cv_train, x_star_cv_train, param[0], param[1], max_iters = 100, eps = 1e-2)
                    elif model_name == 'dual_dal':
                        W_hat, _, _ = dual_dal(n, m, c, A, d, len(train_idx), xi_cv_train, b_cv_train, y_star_cv_train, param)
                    elif model_name == 'lasso':
                        W_hat = Lasso(alpha = param, fit_intercept = False).fit(xi_cv_train, b_cv_train)
                    else: # model_name == 'odece
                        odece = train_odece(n, m, c, A, d, N, xi, b, rep, param)

                    # Evaluate model on the current fold's holdout dataset
                    if model_name in ['primal_dal', 'dual_dal']:
                        b_hat = make_predictions(W_hat, 'matrix', len(holdout_idx), xi_cv_holdout)
                    elif model_name == 'lasso':
                        b_hat = make_predictions(W_hat, 'regression', len(holdout_idx), xi_cv_holdout)
                    else: # model_name == 'odece'
                        b_hat = make_predictions(odece, 'odece', len(holdout_idx), xi_cv_holdout)
                    _, feas, _ = compute_true_soln_metrics(c, A, len(holdout_idx), x_star_cv_holdout, y_star_cv_holdout, b_hat, feas_tol)

                    # Loss for the current fold
                    if model_name == 'lasso':
                        b_hat_arr = np.zeros((len(holdout_idx), m))
                        for i in range(len(holdout_idx)):
                            b_hat_arr[i] = b_hat[i]
                        fold_loss[fold] = mean_squared_error(b_hat_arr, b_cv_holdout) # prediction loss
                    else:
                        fold_loss[fold] = sum(feas.values()) # decision loss (feasibility)

                # Mean CV loss across 5 folds
                cv_loss[N_frac, param, rep] = (1/len(fold_loss)) * sum([fold_loss[fold] for fold, _ in enumerate(kf.split(xi[rep][ : N_frac]), 1)])

            # Pooling CV loss over all replications
            pooled_cv_loss[N_frac, param] = (1/len(training_reps)) * sum([cv_loss[N_frac, param, rep] for rep in training_reps])

        # For a fixed value of N_frac, find the parameter 'param' that optimizes pooled_cv_loss[N_frac, param]
        if model_name == 'lasso':
            minimizing_key = min((k for k in pooled_cv_loss if k[0] == N_frac), key = pooled_cv_loss.get)
            tuned_param[N_frac] = minimizing_key[1]
        else:
            maximizing_key = max((k for k in pooled_cv_loss if k[0] == N_frac), key = pooled_cv_loss.get)
            tuned_param[N_frac] = maximizing_key[1]

    return tuned_param

def create_runtime_file_for_a_model(model_name, column_names, runtime_dir):

    runtime_for_model_file = os.path.join(runtime_dir, f'{model_name}.csv')
    with open(runtime_for_model_file, mode = 'a', newline = '') as file:
        writer = csv.writer(file)
        writer.writerow(column_names)

    return runtime_for_model_file

def evaluate_model(n, m, c, A, N, xi, b, x_star, y_star, rep, model_name, model, feas_tol, opt_tol, results_dir):

    # Create directory that will store the results for the given model
    results_for_model_dir = os.path.join(results_dir, model_name)
    os.makedirs(results_for_model_dir, exist_ok = True)

    # Make predictions of RHS vectors
    if model_name in ['optimistic_dal', 'primal_dal_gamma_0', 'primal_dal_gamma_pos', 'dual_dal']:
        b_hat = make_predictions(model, 'matrix', N, xi[rep])
    elif model_name in ['lr', 'lasso', 'rf']:
        b_hat = make_predictions(model, 'regression', N, xi[rep])
    else: # model_name == 'odece
        b_hat = make_predictions(model, 'odece', N, xi[rep])
    
    # Compute metrics
    prediction_error = {(i, j) : b_hat[i][j] - b[rep][i, j] for i in range(N) for j in range(m)} # prediction minus truth
    constr_satisfaction_true_soln, feas_true_soln, duality_gap_true_soln = compute_true_soln_metrics(c, A, N, x_star[rep], y_star[rep], b_hat, feas_tol)
    x_hat = np.zeros((N, n))
    y_hat = np.zeros((N, m))
    recover_x_hat = {}
    for i in range(N):
        x_hat[i], z_hat_i = solve_downstream_lp(n, c, A, b_hat[i], feas_tol, opt_tol)
        if z_hat_i is None:
            recover_x_hat[i] = 0
        else:
            recover_x_hat[i] = 1
        y_hat[i], _ = solve_dual_downstream_lp(m, c, A, b_hat[i], feas_tol, opt_tol)
    constr_satisfaction_predicted_soln, feas_predicted_soln, duality_gap_predicted_soln = compute_predicted_soln_feas_duality_gap_metrics(c, A, N, b[rep], x_hat, y_hat, feas_tol)
    proj_dist, cost_opt_gap_proj_and_true_soln = compute_predicted_soln_projection_metrics(c, A, N, b[rep], x_star[rep], x_hat, opt_tol)

    # Write out data
    write_pickle_file(os.path.join(results_for_model_dir, f'prediction_error_rep_{rep}.pkl'), prediction_error)
    write_pickle_file(os.path.join(results_for_model_dir, f'constr_satisfaction_true_soln_rep_{rep}.pkl'), constr_satisfaction_true_soln)
    write_pickle_file(os.path.join(results_for_model_dir, f'feas_true_soln_rep_{rep}.pkl'), feas_true_soln)
    write_pickle_file(os.path.join(results_for_model_dir, f'duality_gap_true_soln_rep_{rep}.pkl'), duality_gap_true_soln)
    write_pickle_file(os.path.join(results_for_model_dir, f'recover_x_hat_rep_{rep}.pkl'), recover_x_hat)
    write_pickle_file(os.path.join(results_for_model_dir, f'constr_satisfaction_predicted_soln_rep_{rep}.pkl'), constr_satisfaction_predicted_soln)
    write_pickle_file(os.path.join(results_for_model_dir, f'feas_predicted_soln_rep_{rep}.pkl'), feas_predicted_soln)
    write_pickle_file(os.path.join(results_for_model_dir, f'duality_gap_predicted_soln_rep_{rep}.pkl'), duality_gap_predicted_soln)
    write_pickle_file(os.path.join(results_for_model_dir, f'proj_dist_rep_{rep}.pkl'), proj_dist)
    write_pickle_file(os.path.join(results_for_model_dir, f'cost_opt_gap_proj_and_true_soln_rep_{rep}.pkl'), cost_opt_gap_proj_and_true_soln)

    return None

def analyze_synthetic_experiment_results_for_one_setting(setting, N_train, N_test, base_dir):

    # Grab/set different values to analyze synthetic experiment results
    xi_test = read_pickle_file(os.path.join(base_dir, 'synthetic', f'setting_{setting}', 'test', 'xi.pkl'))
    test_reps = list(xi_test.keys())
    results_dir = os.path.join(base_dir, 'synthetic', f'setting_{setting}', 'synthetic_experiment', 'results')
    model_names = ['optimistic_dal', 'primal_dal_gamma_0', 'primal_dal_gamma_pos', 'dual_dal', 'odece', 'lr', 'lasso', 'rf']
    metrics = ['prediction_error', 'constr_satisfaction_true_soln', 'feas_true_soln', 'duality_gap_true_soln', 'recover_x_hat', 'constr_satisfaction_predicted_soln', 'feas_predicted_soln', 'duality_gap_predicted_soln', 'proj_dist', 'cost_opt_gap_proj_and_true_soln']
    
    for metric in metrics:

        # Create metric csv file and obtain results for the current metric
        if metric.startswith('feas') or metric.startswith('recover'):
            output_dict = get_indicator_results(N_train, N_test, metric, test_reps, model_names, results_dir)
        else:
            output_dict = get_numerical_results(N_train, metric, test_reps, model_names, results_dir)

        # Write out results to the above-created metric csv file
        output_file = os.path.join(results_dir, f'{metric}.csv')
        for N_train_frac in [int((i/4) * N_train) for i in range(1, 5)]:
            with open(output_file, 'a', newline = '') as file:
                writer = csv.writer(file)
                writer.writerow([N_train_frac] + [output_dict[N_train_frac, model] for model in model_names])

    return None

def get_indicator_results(N_train, N_test, metric, test_reps, model_names, results_dir):

    create_metric_file(metric, ['N_train_frac'] + model_names, results_dir)

    # Gather results
    output_dict = {}
    for N_train_frac in [int((i/4) * N_train) for i in range(1, 5)]:
        for model_name in model_names:
            output_dict[N_train_frac, model_name] = 0
            for rep in test_reps:
                file = read_pickle_file(os.path.join(results_dir, f'N_train_{N_train_frac}', f'{model_name}', f'{metric}_rep_{rep}.pkl'))
                output_dict[N_train_frac, model_name] += sum(file.values())
    output_dict = {key : 100 * (value/(len(test_reps) * N_test)) for key, value in output_dict.items()} # reporting values as percentages

    return output_dict

def get_numerical_results(N_train, metric, test_reps, model_names, results_dir):

    create_metric_file(metric, ['N_train_frac'] + model_names, results_dir)

    # Gather results
    output_dict = {}
    for N_train_frac in [int((i/4) * N_train) for i in range(1, 5)]:
        for model_name in model_names:
            output_dict[N_train_frac, model_name] = []
            for rep in test_reps:
                file = read_pickle_file(os.path.join(results_dir, f'N_train_{N_train_frac}', f'{model_name}', f'{metric}_rep_{rep}.pkl'))
                output_dict[N_train_frac, model_name] += list(file.values())
            output_dict[N_train_frac, model_name] = np.median(np.array(output_dict[N_train_frac, model_name]))

    return output_dict

def create_metric_file(metric, column_names, results_dir):

    with open(os.path.join(results_dir, f'{metric}.csv'), mode = 'a', newline = '') as file:
        writer = csv.writer(file)
        writer.writerow(column_names)

    return None

### Solving the downstream linear program and its dual ###

def solve_downstream_lp(n, c, A, b, feas_tol, opt_tol):
    
    model = gp.Model()
    model.params.FeasibilityTol = feas_tol
    model.params.OptimalityTol = opt_tol
    x = model.addMVar(n, lb = 0, name = 'x')
    model.setObjective(c @ x, GRB.MINIMIZE)
    model.addConstr(A @ x >= b)
    model.update()
    model.optimize()

    if model.status == GRB.OPTIMAL:
        return model.X, model.ObjVal
    else:
        return None, None

def solve_dual_downstream_lp(m, c, A, b, feas_tol, opt_tol):
    
    model = gp.Model()
    model.params.FeasibilityTol = feas_tol
    model.params.OptimalityTol = opt_tol
    y = model.addMVar(m, lb = 0, name = 'y')
    model.setObjective(b @ y, GRB.MAXIMIZE)
    model.addConstr(A.T @ y <= c)
    model.update()
    model.optimize()

    if model.status == GRB.OPTIMAL:
        return model.X, model.ObjVal
    else:
        return None, None

### Learning models ###

# 1. Optimistic-DAL
def optimistic_dal(m, c, A, d, N, xi, x_star, y_star):

    model = gp.Model()
    W = model.addMVar((m, d), lb = -np.inf, name = 'W')
    model.setObjective((1/N) * gp.quicksum(c @ x_star[i] - W @ xi[i] @ y_star[i] for i in range(N)), GRB.MINIMIZE)
    model.addConstrs(A @ x_star[i] >= W @ xi[i] for i in range(N))
    model.update()
    model.optimize()

    if model.status == GRB.OPTIMAL:
        return W.X, model.status
    else:
        print('The status of the Optimistic-DAL problem is not \'optimal\'')
        return None, model.status

# 2. Primal-DAL (note we use a L1 regularizer on the model and a penalty on overprediction)
def generate_primal_dal_initial_solution(m, c, A, d, N, xi, x_star, feas_tol, opt_tol):
    
    model = gp.Model()
    model.params.FeasibilityTol = feas_tol
    model.params.OptimalityTol = opt_tol
    W = model.addMVar((m, d), lb = -np.inf, name = 'W')
    y = {i : model.addMVar(shape = (m, ), lb = 0, name = f'y[{i}]') for i in range(N)}
    model.setObjective(0, GRB.MINIMIZE)
    model.addConstrs(A @ x_star[i] >= W @ xi[i] for i in range(N))
    model.addConstrs(A.T @ y[i] <= c for i in range(N))
    model.update()
    model.optimize()

    if model.status == GRB.OPTIMAL:
        return W.X, {i : y[i].X for i in range(N)}, np.array(model.X)
    else:
        print('The Primal-DAL problem is infeasible')
        return None, None, None

# 2.1: ACS - alternate convex search (biconvex algorithm)
def primal_dal_acs(m, c, A, d, W, N, y, xi, b, x_star, lambda_reg, gamma, max_iters, eps):

    # Start a timer for the algorithm
    start_time_algorithm = time.time()

    # Compute objective function value of initial iterate
    prev_obj_value = compute_primal_dal_obj_val(m, c, d, W, N, xi, b, x_star, y, lambda_reg, gamma)

    for k_iter in range(max_iters):

        # Check if the code has hit the maximum time limit
        if time.time() - start_time_algorithm >= 3600:
            return W, y, k_iter + 1

        # Step 1: solve for W while keeping y fixed
        W_var = cp.Variable((m, d))
        W_subproblem_objective = (1/N) * cp.sum([c @ x_star[i] - W_var @ xi[i] @ y[i] for i in range(N)]) + \
              lambda_reg * cp.norm1(W_var) + \
                gamma * cp.sum([cp.maximum(0, b[i, j] - W_var[j] @ xi[i]) for i in range(N) for j in range(m)])
        W_subproblem_constrs = [A @ x_star[i] >= W_var @ xi[i] for i in range(N)]
        W_subproblem = cp.Problem(cp.Minimize(W_subproblem_objective), W_subproblem_constrs)
        W_subproblem.solve(cp.GUROBI)
        W = W_var.value

        # Step 2: solve for y while keeping W fixed
        y_var = cp.Variable((N, m))
        y_subproblem_objective = (1/N) * cp.sum([c @ x_star[i] - W @ xi[i] @ y_var[i] for i in range(N)])
        y_subproblem_constrs = [A.T @ y_var[i] <= c for i in range(N)]
        y_subproblem_constrs.append(y_var >= 0)
        y_subproblem = cp.Problem(cp.Minimize(y_subproblem_objective), y_subproblem_constrs)
        y_subproblem.solve(cp.GUROBI)
        y = y_var.value
        y = {i : y[i] for i in range(N)}

        # Step 3: compute new objective value
        new_obj_value = compute_primal_dal_obj_val(m, c, d, W, N, xi, b, x_star, y, lambda_reg, gamma)

        # Step 4: check for convergence and update previous objective value, if needed
        if new_obj_value == prev_obj_value or (prev_obj_value - new_obj_value)/prev_obj_value < eps: # the first condition takes care of the case where new_obj_value == prev_obj_value == 0
            break
        prev_obj_value = new_obj_value
    
    return W, y, k_iter + 1

# 2.2: CCP - convex concave procedure (difference-of-convex algorithm)
def primal_dal_ccp(m, c, A, d, x, N, xi, b, x_star, lambda_reg, gamma, max_iters, epsilon):

    # Step 0: start the timer for the function and store the initial objective function value
    start_time_algorithm = time.time()
    prev_obj_value = h_0_primal_dal(x, m, c, d, N, xi, b, x_star, lambda_reg, gamma)
 
    for k_iter in range(max_iters):

        # Checking if the code has hit the maximum time limit
        if time.time() - start_time_algorithm >= 3600:
            return x, k_iter + 1

        # Step 1: solve convexified subproblem
        x_var = cp.Variable(x.shape)
        objective = f_0_symbolic_primal_dal(x_var, m, c, d, N, xi, b, x_star, lambda_reg, gamma) - \
            (g_0_primal_dal(x, m, d, N, xi) + grad_g_0_primal_dal(x, m, d, N, xi).T @ (x_var - x))
        constrs = get_ccp_constrs_primal_dal(x_var, m, c, A, d, N, xi, x_star)
        convexified_problem = cp.Problem(cp.Minimize(objective), constrs)
        convexified_problem.solve(solver = cp.CLARABEL)
        x = x_var.value
        new_obj_value = h_0_primal_dal(x, m, c, d, N, xi, b, x_star, lambda_reg, gamma)

        # Step 2: check for convergence
        if k_iter >= 1:
            if new_obj_value == prev_obj_value: # this takes care of the case where new_obj_value == prev_obj_value == 0
                break
            elif (prev_obj_value - new_obj_value)/prev_obj_value < epsilon:
                break
        
        # Step 3: update objective function value
        prev_obj_value = new_obj_value

    return x, k_iter + 1

# ~~~ Start of CCP-related functions ~~~ #
def K_plus(i, d, xi):

    return [k for k in range(d) if xi[i, k] > 0]

def K_minus(i, d, xi):

    return [k for k in range(d) if xi[i, k] < 0]

def f_0_symbolic_primal_dal(x, m, c, d, N, xi, b, x_star, lambda_reg, gamma):
    
    return (1/N) * cp.sum([c @ x_star[i] + \
                           (1/2) * cp.sum([cp.sum([xi[i, k] * (x[d * j + k] ** 2 + x[m * d + m * i + j] ** 2) for k in K_plus(i, d, xi)]) - \
                                           cp.sum([xi[i, k] * (x[d * j + k] + x[m * d + m * i + j]) ** 2 for k in K_minus(i, d, xi)]) for j in range(m)]) for i in range(N)]) + \
                                           lambda_reg * cp.sum([cp.abs(x[d * j + k]) for j in range(m) for k in range(d)]) + \
                                            gamma * cp.sum([cp.maximum(0, b[i, j] - x[d * j : d * j + d] @ xi[i]) for i in range(N) for j in range(m)])

def f_0_primal_dal(x, m, c, d, N, xi, b, x_star, lambda_reg, gamma):

    return (1/N) * sum([c @ x_star[i] + \
                        (1/2) * sum([sum([xi[i, k] * (x[d * j + k] ** 2 + x[m * d + m * i + j] ** 2) for k in K_plus(i, d, xi)]) - \
                                     sum([xi[i, k] * (x[d * j + k] + x[m * d + m * i + j]) ** 2 for k in K_minus(i, d, xi)]) for j in range(m)]) for i in range(N)]) + \
                                     lambda_reg * sum([abs(x[d * j + k]) for j in range(m) for k in range(d)]) + \
                                        gamma * sum([max(0, b[i, j] - x[d * j : d * j + d] @ xi[i]) for i in range(N) for j in range(m)])

def g_0_primal_dal(x, m, d, N, xi):

    return (1/(2 * N)) * sum([(sum([xi[i, k] * (x[d * j + k] + x[m * d + m * i + j]) ** 2 for k in K_plus(i, d, xi)]) - 
                               sum([xi[i, k] * (x[d * j + k] ** 2 + x[m * d + m * i + j] ** 2) for k in K_minus(i, d, xi)])) for j in range(m) for i in range(N)])

def grad_g_0_primal_dal(x, m, d, N, xi):
    
    grad_g_0_W = np.zeros(m * d, dtype = float)
    grad_g_0_y = np.zeros(N * m, dtype = float)

    for j in range(m):
        for k in range(d):
            grad_g_0_W[d * j + k] = (1/N) * sum([xi[i, k] * (x[d * j + k] + x[m * d + m * i + j]) if k in K_plus(i, d, xi) else -xi[i, k] * x[d * j + k] for i in range(N)])

    for i in range(N):
        for j in range(m):
            grad_g_0_y[m * i + j] = (1/N) * (x[m * d + m * i + j] * sum([abs(xi[i, k]) for k in range(d)]) + sum([xi[i, k] * (x[d * j + k]) for k in K_plus(i, d, xi)]))

    grad_g_0_final = np.concatenate((grad_g_0_W, grad_g_0_y))

    return grad_g_0_final

def h_0_primal_dal(x, m, c, d, N, xi, b, x_star, lambda_reg, gamma):

    return f_0_primal_dal(x, m, c, d, N, xi, b, x_star, lambda_reg, gamma) - g_0_primal_dal(x, m, d, N, xi)

def get_ccp_constrs_primal_dal(x, m, c, A, d, N, xi, x_star):
    
    constrs = []
    
    for i in range(N):
        for j in range(m):
            constrs.append(A[j] @ x_star[i] >= x[d * j : d * j + d] @ xi[i])
    for i in range(N):
        constrs.append(A.T @ x[m * d + m * i : m * d + m * i + m] <= c)
        constrs.append(x[m * d + m * i : m * d + m * i + m] >= 0)

    return constrs
# ~~~ End of CCP-related functions ~~~ #

# 2.3: Gurobi
def primal_dal_gurobi(m, c, A, d, N, xi, b, x_star, lambda_reg, gamma):

    model = gp.Model()
    model.params.NonConvex = 2
    model.params.TimeLimit = 3600
    W = model.addMVar((m, d), lb = -np.inf, name = 'W')
    y = {i : model.addMVar(shape = (m, ), lb = 0, name = f'y[{i}]') for i in range(N)}
    aux_reg = model.addMVar((m, d), lb = -np.inf, name = 'aux_reg')
    aux_pen = model.addMVar((N, m), lb = 0, name = 'aux_pen')
    model.setObjective((1/N) * gp.quicksum(c @ x_star[i] - (W @ xi[i]) @ y[i] for i in range(N)) + \
                       lambda_reg * gp.quicksum(aux_reg[j, k] for j in range(m) for k in range(d)) + \
                        gamma * gp.quicksum(aux_pen[i, j] for i in range(N) for j in range(m)), GRB.MINIMIZE)
    model.addConstrs(A @ x_star[i] >= W @ xi[i] for i in range(N))
    model.addConstrs(A.T @ y[i] <= c for i in range(N))
    model.addConstrs(aux_reg[j, k] >= W[j, k] for j in range(m) for k in range(d))
    model.addConstrs(aux_reg[j, k] >= -W[j, k] for j in range(m) for k in range(d))
    model.addConstrs(aux_pen[i, j] >= b[i, j] - W[j] @ xi[i] for i in range(N) for j in range(m))
    model.update()
    model.optimize()

    return W.X, {i : y[i].X for i in range(N)}, model.status

# 3. Dual-DAL
def dual_dal(n, m, c, A, d, N, xi, b, y_star, alpha):
    
    model = gp.Model()
    W = model.addMVar(shape = (m, d), lb = -np.inf, name = 'W')
    x = {i : model.addMVar(shape = (n, ), lb = 0, name = f'x[{i}]') for i in range(N)}
    model.setObjective((1/N) * gp.quicksum(c @ x[i] - (alpha * (W @ xi[i]) - b[i]) @ y_star[i] for i in range(N)), GRB.MINIMIZE)
    model.addConstrs(A @ x[i] >= alpha * (W @ xi[i]) - b[i] for i in range(N))
    model.update()
    model.optimize()

    if model.status == GRB.OPTIMAL:
        return W.X, {i : x[i].X for i in range(N)}, model.status
    else:
        print('The status of the Dual-DAL problem is not \'optimal\'')
        return None, None, model.status

# 4. ODECE
def train_odece(n, m, c, A, d, N, xi, b, rep, infeasibility_aversion_coeff):

    # ODECE settings
    margin_threshold = 2
    learning_rate = 0.05
    batch_size = 32
    max_epochs = 20

    dataset_train = syn_dataset(xi[rep][ : N], c, A, b[rep][ : N])
    loader_train = DataLoader(dataset_train, batch_size = batch_size, shuffle = True, num_workers = 0) # DataLoader is a pytorch-specific class
    solver = syn_solver()
    reg = LinearRegressionforSyn(d, n, m)
    odece = ODECE([reg], predict_indices = 1, optsolver = solver, num_predconstrsvar = 1, infeasibility_aversion_coeff = infeasibility_aversion_coeff, margin_threshold = margin_threshold, predict_cost = False, lr = learning_rate, max_epochs = max_epochs)
    trainer = pl.Trainer(max_epochs = max_epochs, check_val_every_n_epoch = 1, logger = False, enable_progress_bar = False, enable_model_summary = False, enable_checkpointing = False)
    trainer.fit(odece, train_dataloaders = loader_train)

    return odece

### Evaluating the objective function of the different learning problems for a given solution ###

def compute_avg_duality_gap(c, W, N, xi, x, y):
    
    return (1/N) * sum([c @ x[i] - W @ xi[i] @ y[i] for i in range(N)])

def compute_avg_duality_gap_w_abs(c, W, N, xi, x, y):

    return (1/N) * sum([abs(c @ x[i] - W @ xi[i] @ y[i]) for i in range(N)])

def compute_primal_dal_obj_val(m, c, d, W, N, xi, b, x, y, lambda_reg, gamma):
    
    return (1/N) * sum([c @ x[i] - W @ xi[i] @ y[i] for i in range(N)]) + \
        lambda_reg * sum([abs(W[j, k]) for j in range(m) for k in range(d)]) + \
          gamma * sum([max(0, b[i, j] - W[j] @ xi[i]) for i in range(N) for j in range(m)])

def compute_primal_dal_obj_val_by_term(m, c, d, W, N, xi, b, x, y):
    
    return (1/N) * sum([c @ x[i] - (W @ xi[i]) @ y[i] for i in range(N)]), sum([abs(W[j, k]) for j in range(m) for k in range(d)]), sum([max(0, b[i, j] - W[j] @ xi[i]) for i in range(N) for j in range(m)])

def compute_dual_dal_obj_val(c, W, N, xi, b, x, y, alpha):

    return (1/N) * sum([c @ x[i] - (alpha * (W @ xi[i]) - b[i]) @ y[i] for i in range(N)])

### Generate synthetic data ###

def generate_synthetic_data_for_one_setting(settings, setting, num_batches, N_train, N_test, num_reps, feas_tol, opt_tol, base_dir):

    # Grab setting-specific data from settings dictionary
    n = settings[setting][0]
    m = settings[setting][1]
    d = settings[setting][2]
    P_Xi = settings[setting][3]
    P_Xi_param_1 = settings[setting][4]
    P_Xi_param_2 = settings[setting][5]
    xi_b_relationship = settings[setting][6]

    # Set training and test data directories
    setting_dir = os.path.join(base_dir, 'synthetic', f'setting_{setting}')
    train_data_dir = os.path.join(setting_dir, 'train')
    test_data_dir = os.path.join(setting_dir, 'test')
    os.makedirs(train_data_dir, exist_ok = True)
    os.makedirs(test_data_dir, exist_ok = True)

    # Generate and write out the cost vector c, the constraint matrix A, and the ground-truth linear model W_gt
    seed = setting // 4 # NOTE: this is important and ensures that (c, A, W_gt) are the same for every consecutive group of four settings
    np.random.seed(seed)
    c = np.random.uniform(-10, 10, (n,))
    A = np.random.uniform(-10, 10, (m, n))
    W_gt = np.random.binomial(1, 0.5, (m, d))
    b_temp = np.random.uniform(-20, 100, (m,)) # proxy RHS vector; this is to see if (c, A) is suitable for generating instances with finite optimal solutions

    # Check if the current (c, A, b_temp) triplet generates an unbounded or infeasible downstream problem; if so, update (c, A) until this no longer happens
    inf_or_unbd_flag = True
    while inf_or_unbd_flag:
        model = gp.Model()
        model.params.FeasibilityTol = feas_tol
        model.params.OptimalityTol = opt_tol
        x = model.addMVar(n, lb = 0, name = 'x')
        model.setObjective(c @ x, GRB.MINIMIZE)
        model.addConstr(A @ x >= b_temp)
        model.update()
        model.optimize()

        if model.status in [GRB.INFEASIBLE, GRB.UNBOUNDED, GRB.INF_OR_UNBD]:
            seed += num_batches # this ensures there is no overlap of (c, A, W_gt) between the 3 batches
            np.random.seed(seed)
            c = np.random.uniform(-10, 10, (n,))
            A = np.random.uniform(-10, 10, (m, n))
            W_gt = np.random.binomial(1, 0.5, (m, d))
        else:
            inf_or_unbd_flag = False

    # Save setting-specific data
    np.save(os.path.join(setting_dir, 'c.npy'), c)
    np.save(os.path.join(setting_dir, 'A.npy'), A)
    np.save(os.path.join(setting_dir, 'W_gt.npy'), W_gt)

    # Generate and write out training data
    xi_train, b_train, x_star_train, v_primal_star_train, y_star_train, v_dual_star_train, overall_idx_train = generate_samples(n, m, d, c, A, W_gt, P_Xi, P_Xi_param_1, P_Xi_param_2, xi_b_relationship, num_batches, N_train, num_reps, feas_tol, opt_tol, overall_idx = seed + num_batches)
    write_out_synthetic_data(xi_train, b_train, x_star_train, v_primal_star_train, y_star_train, v_dual_star_train, train_data_dir)

    # Generate and write out test data
    xi_test, b_test, x_star_test, v_primal_star_test, y_star_test, v_dual_star_test, _ = generate_samples(n, m, d, c, A, W_gt, P_Xi, P_Xi_param_1, P_Xi_param_2, xi_b_relationship, num_batches, N_test, num_reps, feas_tol, opt_tol, overall_idx = overall_idx_train)
    write_out_synthetic_data(xi_test, b_test, x_star_test, v_primal_star_test, y_star_test, v_dual_star_test, test_data_dir)

    print(f'Wrote out training and test data for setting # {setting}')

    return None

def generate_samples(n, m, d, c, A, W_gt, P_Xi, P_Xi_param_1, P_Xi_param_2, xi_b_relationship, num_batches, N, num_reps, feas_tol, opt_tol, overall_idx):

    # Instantiate data dictionaries
    xi = {rep : np.zeros((N, d)) for rep in range(num_reps)}
    b = {rep : np.zeros((N, m)) for rep in range(num_reps)}
    x_star = {rep : np.zeros((N, n)) for rep in range(num_reps)}
    v_primal_star = {rep : np.zeros(N) for rep in range(num_reps)}
    y_star = {rep : np.zeros((N, m)) for rep in range(num_reps)}
    v_dual_star = {rep : np.zeros(N) for rep in range(num_reps)}

    # Iterate over each replication
    for rep in range(num_reps):

        i = 0
        while i != N:

            # Generate context vector
            np.random.seed(overall_idx)
            if P_Xi == 'uniform':
                xi_i = np.random.uniform(P_Xi_param_1, P_Xi_param_2, (d,))
            else: # P_Xi is a normal distribution
                xi_i = np.random.normal(P_Xi_param_1, P_Xi_param_2, (d,))
            xi_i[0] = abs(xi_i[0]) + 0.5 # give the first feature the same sign (+) across the training datset to guarantee feasibility of certain DAL problems

            # Generate corresponding RHS vector
            if xi_b_relationship == 'linear':
                b_i = np.array([(1/np.sqrt(d)) * (W_gt[j] @ xi_i) + np.random.normal(0, 1) for j in range(m)])
            else: # relationship is quadratic
                b_i = np.array([((1/np.sqrt(d)) * (W_gt[j] @ xi_i)) ** 2 + np.random.normal(0, 1) for j in range(m)])

            # Compute optimal solution/cost of associated downstream problem (and its dual)
            x_star_i, v_primal_star_i = solve_downstream_lp(n, c, A, b_i, feas_tol, opt_tol)
            y_star_i, v_dual_star_i = solve_dual_downstream_lp(m, c, A, b_i, feas_tol, opt_tol)
            
            # Only add datapoints where the associated downstream problem has a finite optimal cost
            if v_primal_star_i is not None:
                xi[rep][i] = xi_i
                b[rep][i] = b_i
                x_star[rep][i] = x_star_i
                v_primal_star[rep][i] = v_primal_star_i
                y_star[rep][i] = y_star_i
                v_dual_star[rep][i] = v_dual_star_i
                i += 1
            
            # Increment "overall index" for data generation
            overall_idx += num_batches
    
    return xi, b, x_star, v_primal_star, y_star, v_dual_star, overall_idx

### Functions for reading and writing data ###

def read_pickle_file(file):
    
    with open(file, 'rb') as f:
        loaded_data = pickle.load(f)

    return loaded_data

def write_pickle_file(file, data):

    with open(file, 'wb') as f:
        pickle.dump(data, f)

    return None

def write_out_synthetic_data(xi, b, x_star, v_primal_star, y_star, v_dual_star, data_dir):
    
    xi_file = os.path.join(data_dir, 'xi.pkl')
    b_file = os.path.join(data_dir, 'b.pkl')
    x_star_file = os.path.join(data_dir, 'x_star.pkl')
    v_primal_star_file = os.path.join(data_dir, 'v_primal_star.pkl')
    y_star_file = os.path.join(data_dir, 'y_star.pkl')
    v_dual_star_file = os.path.join(data_dir, 'v_dual_star.pkl')

    write_pickle_file(xi_file, xi)
    write_pickle_file(b_file, b)
    write_pickle_file(x_star_file, x_star)
    write_pickle_file(v_primal_star_file, v_primal_star)
    write_pickle_file(y_star_file, y_star)
    write_pickle_file(v_dual_star_file, v_dual_star)

    return None

def read_setting_data(setting_dir):

    # Global setting data, i.e., not per replication
    c = np.load(os.path.join(setting_dir, 'c.npy'))
    A = np.load(os.path.join(setting_dir, 'A.npy'))
    W_gt = np.load(os.path.join(setting_dir, 'W_gt.npy'))

    # Downstream problem values
    n = len(c)
    m = A.shape[0]
    d = W_gt.shape[1]

    # Training data
    train_data_dir = os.path.join(setting_dir, 'train')
    xi_train = read_pickle_file(os.path.join(train_data_dir, 'xi.pkl'))
    b_train = read_pickle_file(os.path.join(train_data_dir, 'b.pkl'))
    x_star_train = read_pickle_file(os.path.join(train_data_dir, 'x_star.pkl'))
    y_star_train = read_pickle_file(os.path.join(train_data_dir, 'y_star.pkl'))

    # Test data
    test_data_dir = os.path.join(setting_dir, 'test')
    xi_test = read_pickle_file(os.path.join(test_data_dir, 'xi.pkl'))
    b_test = read_pickle_file(os.path.join(test_data_dir, 'b.pkl'))
    x_star_test = read_pickle_file(os.path.join(test_data_dir, 'x_star.pkl'))
    y_star_test = read_pickle_file(os.path.join(test_data_dir, 'y_star.pkl'))

    return n, m, c, A, d, W_gt, xi_train, list(xi_train.keys()), b_train, x_star_train, y_star_train, xi_test, list(xi_test.keys()), b_test, x_star_test, y_star_test

def write_model(model_name, model, model_dir, to_pickle):

    if to_pickle:
        write_pickle_file(os.path.join(model_dir, f'{model_name}.pkl'), model)
    else:
        np.save(os.path.join(model_dir, f'{model_name}.npy'), model)

    return None

### Functions for training and writing out the models ###

def train_optimistic_dal_and_write_out(m, c, A, d, N, xi, x_star, y_star, rep, model_dir, runtime_file):

    # Train model
    start_time = time.time()
    W_hat, status = optimistic_dal(m, c, A, d, N, xi[rep], x_star[rep], y_star[rep])
    end_time = time.time()
    total_time = end_time - start_time
    optimistic_dal_obj_val = compute_avg_duality_gap(c, W_hat, N, xi[rep], x_star[rep], y_star[rep])
    write_model(f'W_hat_optimistic_dal_rep_{rep}', W_hat, model_dir, to_pickle = False)
    
    # Write out runtime results
    with open(runtime_file, 'a', newline = '') as file:
        writer = csv.writer(file)
        writer.writerow([rep, N, status, total_time, optimistic_dal_obj_val])

    return None

def train_primal_dal_and_write_out(m, c, A, d, N, xi, b, x_star, rep, lambda_reg, gamma, soln_method, feas_tol, opt_tol, model_dir, runtime_file):

    # NOTE: the solution method must either be 'acs', 'ccp', or 'gurobi'

    # Generate initial solution, if necessary
    if soln_method in ['acs', 'ccp']:
        W_0_acs, y_0_acs, x_0_ccp = generate_primal_dal_initial_solution(m, c, A, d, N, xi[rep], x_star[rep], feas_tol, opt_tol)
    
    # Train model
    if soln_method == 'acs':
        start_time = time.time()
        W_hat, y_hat, num_iters = primal_dal_acs(m, c, A, d, W_0_acs, N, y_0_acs, xi[rep], b[rep], x_star[rep], lambda_reg, gamma, max_iters = 100, eps = 1e-2)
        end_time = time.time()
    elif soln_method == 'ccp':
        start_time = time.time()
        x_hat_ccp, num_iters = primal_dal_ccp(m, c, A, d, x_0_ccp, N, xi[rep], b[rep], x_star[rep], lambda_reg, gamma, max_iters = 100, epsilon = 1e-2)
        W_hat = np.zeros((m, d))
        for j in range(m):
            for k in range(d):
                W_hat[j, k] = x_hat_ccp[j * d + k]
        y_hat = {i : x_hat_ccp[m * d + m * i : m * d + m * (i + 1)] for i in range(N)}
        end_time = time.time()
    else:
        start_time = time.time()
        W_hat, y_hat, num_iters = primal_dal_gurobi(m, c, A, d, N, xi[rep], b[rep], x_star[rep], lambda_reg, gamma)
        end_time = time.time()
        # NOTE: the above "num_iters" is really the optimization status code produced by Gurobi at termination of the solve, but we refer to it as "num_iters" for consistency within the scope of this function
    total_time = end_time - start_time
    primal_dal_obj_val = compute_primal_dal_obj_val(m, c, d, W_hat, N, xi[rep], b[rep], x_star[rep], y_hat, lambda_reg, gamma)
    obj_val_duality_gap, obj_val_reg, obj_val_pen = compute_primal_dal_obj_val_by_term(m, c, d, W_hat, N, xi[rep], b[rep], x_star[rep], y_hat)
    if gamma == 0:
        write_model(f'W_hat_primal_dal_gamma_0_rep_{rep}', W_hat, model_dir, to_pickle = False)
        write_model(f'y_hat_primal_dal_gamma_0_rep_{rep}', y_hat, model_dir, to_pickle = True)
    else:
        write_model(f'W_hat_primal_dal_gamma_pos_rep_{rep}', W_hat, model_dir, to_pickle = False)
        write_model(f'y_hat_primal_dal_gamma_pos_rep_{rep}', y_hat, model_dir, to_pickle = True)
    
    # Write out runtime results
    with open(runtime_file, 'a', newline = '') as file:
        writer = csv.writer(file)
        writer.writerow([rep, N, soln_method, lambda_reg, gamma, num_iters, total_time, primal_dal_obj_val, obj_val_duality_gap, obj_val_reg, obj_val_pen])

    return None

def train_dual_dal_and_write_out(n, m, c, A, d, N, xi, b, x_star, y_star, rep, alpha, model_dir, runtime_file):

    # Train model
    start_time = time.time()
    W_hat, x_hat, status = dual_dal(n, m, c, A, d, N, xi[rep], b[rep], y_star[rep], alpha)
    end_time = time.time()
    total_time = end_time - start_time
    if W_hat is None:
        dual_dal_obj_val, avg_duality_gap_w_abs = None, None
        write_model(f'W_hat_dual_dal_rep_{rep}', W_hat, model_dir, to_pickle = True)
    else:
        dual_dal_obj_val = compute_dual_dal_obj_val(c, W_hat, N, xi[rep], b[rep], x_hat, y_star[rep], alpha)
        avg_duality_gap_w_abs = compute_avg_duality_gap_w_abs(c, W_hat, N, xi[rep], x_star[rep], y_star[rep]) # NOTE: since the Dual-DAL solution (W_hat, (x_hat)_i) might not satisfy the predicted feasibility constraints in the primal space, we evaluate the average optimality gap objective function with absolute values
        write_model(f'W_hat_dual_dal_rep_{rep}', W_hat, model_dir, to_pickle = False)
    write_model(f'x_hat_dual_dal_rep_{rep}', x_hat, model_dir, to_pickle = True)

    # Write out runtime results
    with open(runtime_file, 'a', newline = '') as file:
        writer = csv.writer(file)
        writer.writerow([rep, N, alpha, status, total_time, dual_dal_obj_val, avg_duality_gap_w_abs])

    return None

def train_odece_and_write_out(n, m, c, A, d, N, xi, b, rep, infeasibility_aversion_coeff, model_dir, runtime_file):

    # Train model
    start_time = time.time()
    odece = train_odece(n, m, c, A, d, N, xi, b, rep, infeasibility_aversion_coeff)
    end_time = time.time()
    total_time = end_time - start_time
    write_model(f'odece_rep_{rep}', odece, model_dir, to_pickle = True)

    # Write out runtime results
    with open(runtime_file, 'a', newline = '') as file:
        writer = csv.writer(file)
        writer.writerow([rep, N, infeasibility_aversion_coeff, total_time])

    return None

def train_regression_model_and_write_out(N, xi, b, rep, model_name, model_dir, runtime_file, tuned_deltas = None, d = None):

    # Train model
    if model_name == 'lr':
        start_time = time.time()
        model = LinearRegression(fit_intercept = False).fit(xi[rep][ : N], b[rep][ : N])
        end_time = time.time()
    elif model_name == 'lasso':
        start_time = time.time()
        model = Lasso(alpha = tuned_deltas[N], fit_intercept = False).fit(xi[rep][ : N], b[rep][ : N])
        end_time = time.time()
    else:
        start_time = time.time()
        model = RandomForestRegressor(max_features = int(np.ceil(d/3)), random_state = 0).fit(xi[rep][ : N], b[rep][ : N])
        end_time = time.time()
    total_time = end_time - start_time

    # Write out model
    if model_name in ['lr', 'lasso']:
        write_model('W_hat_' + model_name + f'_rep_{rep}', model, model_dir, to_pickle = True)
    else:
        write_model(model_name + f'_rep_{rep}', model, model_dir, to_pickle = True)
    with open(runtime_file, 'a', newline = '') as file:
        writer = csv.writer(file)
        if model_name in ['lr', 'rf']:
            writer.writerow([rep, N, total_time])
        else:
            writer.writerow([rep, N, tuned_deltas[N], total_time])

    return None

### Evaluation functions for after the models are trained ###

def make_predictions(model, model_type, N, xi):

    if model_type == 'matrix':
        return {i : model @ xi[i] for i in range(N)}
    elif model_type == 'regression':
        return {i : model.predict(xi[i].reshape(1, -1))[0] for i in range(N)}
    else: # model_type == 'odece'
        output_dict = {}
        for i in range(N):
            feature_tensor = torch.tensor(xi[i], dtype = torch.float32)
            model.eval()
            with torch.no_grad():
                b_hat = model(feature_tensor)
            b_hat_tensor = b_hat[0]
            b_hat_np = b_hat_tensor.numpy()
            output_dict[i] = b_hat_np

        return output_dict

def compute_true_soln_metrics(c, A, N, x_star, y_star, b_hat, feas_tol):

    constr_satisfaction = {}
    feas = {}
    duality_gap = {}

    # Looping over true solutions
    for i in range(N):
        eval = (A @ np.array(x_star[i]) >= b_hat[i] - feas_tol)
        constr_satisfaction[i] = sum(eval)
        if all(eval):
            feas[i] = 1
            duality_gap[i] = c @ x_star[i] - b_hat[i] @ y_star[i]
        else:
            feas[i] = 0

    return constr_satisfaction, feas, duality_gap

def compute_predicted_soln_feas_duality_gap_metrics(c, A, N, b, x_hat, y_hat, feas_tol):
    
    constr_satisfaction = {}
    feas = {}
    duality_gap = {}

    # Looping over predicted solutions for a given model
    for i in range(N):
        if not np.isnan(x_hat[i]).any():
            eval = (A @ np.array(x_hat[i]) >= b[i] - feas_tol)
            constr_satisfaction[i] = sum(eval)
            if all(eval):
                feas[i] = 1
                duality_gap[i] = c @ x_hat[i] - b[i] @ y_hat[i]
            else:
                feas[i] = 0

    return constr_satisfaction, feas, duality_gap

def compute_predicted_soln_projection_metrics(c, A, N, b, x_star, x_hat, opt_tol):

    proj_dist = {}
    cost_opt_gap_proj_and_true_soln = {}

    # Looping over predicted solutions for a given model
    for i in range(N):
        if not np.isnan(x_hat[i]).any():

            # Project x_hat onto the true feasible region
            x_tilde_i = project_onto_polyhedron(A, b[i], x_hat[i])

            # Compute the distance from the optimal solution of the predicted problem to the true feasible region
            proj_dist[i] = np.linalg.norm(x_hat[i] - x_tilde_i)

            # Compute the cost optimality gap between the projected and true solutions
            cost_opt_gap_proj_and_true_soln[i] = get_normalized_cost_opt_gap(c, x_tilde_i, x_star[i], opt_tol)

    return proj_dist, cost_opt_gap_proj_and_true_soln

def project_onto_polyhedron(A, b, x_hat):

    n = len(x_hat)
    x_tilde = cp.Variable(n)
    objective = cp.Minimize(cp.sum_squares(x_tilde - x_hat))
    constrs = [A @ x_tilde >= b, x_tilde >= 0]
    problem = cp.Problem(objective, constrs)
    problem.solve(solver = cp.GUROBI)

    if problem.status == 'optimal':
        x_tilde = x_tilde.value
        x_tilde.shape = (n,)
        return x_tilde
    else:
        return None

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ BEER EXPERIMENT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

### Main step of the experiment ###

def run_all_replications_of_beer_experiment(lambda_regs, gammas, deltas, base_dir):

    ### Setup ###

    # Grab data from the beer network optimization problem
    beer_lp = gp.read(os.path.join(base_dir, 'beer', 'beer.lp'))
    c = np.array(beer_lp.getAttr('Obj'))
    n = len(c) # number of variables
    A = beer_lp.getA().toarray()
    b = np.array(beer_lp.getAttr('RHS'))
    m_u, m_v, m_w = 5, 7, 12 # we predict 5 RHS values; the remaining 19 deterministic RHS values are split between 7 equality constraints and 12 inequality constraints
    A_u, A_v, A_w = A[ : m_u , :], A[m_u : m_u + m_v , :], A[m_u + m_v : , :] # row-subsets of the constraint matrix A
    v, w = b[m_u : m_u + m_v], b[m_u + m_v : ] # deterministic RHS subvectors

    # Generate feature-target dataset
    xi, u = create_feature_target_vectors_beer(pd.read_csv(os.path.join(base_dir, 'beer', 'datasets', 'beer_final_df.csv')))
    d = xi.shape[1] # number of contextual features

    # Remove any records with nulls and prepend feature vectors with a 1
    mask_xi, mask_u = np.isnan(xi).any(axis = 1), np.isnan(u).any(axis = 1)
    mask = mask_xi | mask_u
    xi = xi[~mask]
    xi = np.hstack((np.ones((len(xi), 1)), xi))
    u = u[~mask]

    # Create training and test data directories
    train_data_dir = os.path.join(base_dir, 'beer', 'train')
    test_data_dir = os.path.join(base_dir, 'beer', 'test')
    os.makedirs(train_data_dir, exist_ok = True)
    os.makedirs(test_data_dir, exist_ok = True)

    # Generate one split of the dataset for each replication and write out to corresponding directory
    reps = range(30) # the number here indicates the number of replications
    xi_train, xi_test, u_train, u_test = get_train_test_splits_beer(xi, u, reps)
    write_pickle_file(os.path.join(train_data_dir, 'xi.pkl'), xi_train)
    write_pickle_file(os.path.join(train_data_dir, 'u.pkl'), u_train)
    write_pickle_file(os.path.join(test_data_dir, 'xi.pkl'), xi_test)
    write_pickle_file(os.path.join(test_data_dir, 'u.pkl'), u_test)

    # Set relevant parameters
    N_train, N_test = len(xi_train[0]), len(xi_test[0])
    feas_tol, opt_tol = 1e-6, 1e-6

    # Solve primal and dual beer network optimization problems using the RHS vectors in the training and test datasets
    x_star_train, y_u_star_train, y_v_star_train, y_w_star_train, x_star_test, y_u_star_test, y_v_star_test, y_w_star_test = [{} for _ in range(8)]
    for rep in reps:
        x_star_train_rep, y_u_star_train_rep, y_v_star_train_rep, y_w_star_train_rep, x_star_test_rep, y_u_star_test_rep, y_v_star_test_rep, y_w_star_test_rep = np.zeros((N_train, n)), np.zeros((N_train, m_u)), np.zeros((N_train, m_v)), np.zeros((N_train, m_w)), np.zeros((N_test, n)), np.zeros((N_test, m_u)), np.zeros((N_test, m_v)), np.zeros((N_test, m_w))
        for i in range(N_train):
            x_star_train_rep[i], _ = solve_downstream_lp_beer(n, c, A_u, A_v, A_w, u_train[rep][i], v, w, feas_tol, opt_tol)
            y_u_star_train_rep[i], y_v_star_train_rep[i], y_w_star_train_rep[i], _ = solve_dual_downstream_lp_beer(m_u, m_v, m_w, c, A_u, A_v, A_w, u_train[rep][i], v, w, feas_tol, opt_tol)
        for i in range(N_test):
            x_star_test_rep[i], _ = solve_downstream_lp_beer(n, c, A_u, A_v, A_w, u_test[rep][i], v, w, feas_tol, opt_tol)
            y_u_star_test_rep[i], y_v_star_test_rep[i], y_w_star_test_rep[i], _= solve_dual_downstream_lp_beer(m_u, m_v, m_w, c, A_u, A_v, A_w, u_test[rep][i], v, w, feas_tol, opt_tol)
        x_star_train[rep], y_u_star_train[rep], y_v_star_train[rep], y_w_star_train[rep], x_star_test[rep], y_u_star_test[rep], y_v_star_test[rep], y_w_star_test[rep] = x_star_train_rep, y_u_star_train_rep, y_v_star_train_rep, y_w_star_train_rep, x_star_test_rep, y_u_star_test_rep, y_v_star_test_rep, y_w_star_test_rep
    write_pickle_file(os.path.join(train_data_dir, 'x_star.pkl'), x_star_train)
    write_pickle_file(os.path.join(train_data_dir, 'y_u_star.pkl'), y_u_star_train)
    write_pickle_file(os.path.join(train_data_dir, 'y_v_star.pkl'), y_v_star_train)
    write_pickle_file(os.path.join(train_data_dir, 'y_w_star.pkl'), y_w_star_train)
    write_pickle_file(os.path.join(test_data_dir, 'x_star.pkl'), x_star_test)
    write_pickle_file(os.path.join(test_data_dir, 'y_u_star.pkl'), y_u_star_test)
    write_pickle_file(os.path.join(test_data_dir, 'y_v_star.pkl'), y_v_star_test)
    write_pickle_file(os.path.join(test_data_dir, 'y_w_star.pkl'), y_w_star_test)

    ### Tune models ###
    tuned_params_dir = os.path.join(base_dir, 'beer', 'tuned_params')
    os.makedirs(tuned_params_dir, exist_ok = True)

    # 1a. Primal-DAL (gamma = 0)
    params = [(lambda_reg, gammas[0]) for lambda_reg in lambda_regs]
    tuned_lambda_regs_gamma_0 = tune_model_beer(m_u, m_v, m_w, c, A_u, A_v, A_w, d, v, w, N_train, xi_train, u_train, x_star_train, y_u_star_train, y_v_star_train, y_w_star_train, 'primal_dal', params, reps, feas_tol, opt_tol)
    tuned_lambda_regs_gamma_0 = {key : value[0] for key, value in tuned_lambda_regs_gamma_0.items()}
    tuned_lambda_regs_gamma_0_file = os.path.join(tuned_params_dir, 'tuned_lambda_regs_gamma_0.pkl')
    write_pickle_file(tuned_lambda_regs_gamma_0_file, tuned_lambda_regs_gamma_0)

    # 1b. Primal-DAL (gamma > 0)
    params = [(lambda_reg, gamma) for lambda_reg in lambda_regs for gamma in gammas[1 : ]]
    tuned_lambda_regs_and_gammas = tune_model_beer(m_u, m_v, m_w, c, A_u, A_v, A_w, d, v, w, N_train, xi_train, u_train, x_star_train, y_u_star_train, y_v_star_train, y_w_star_train, 'primal_dal', params, reps, feas_tol, opt_tol)
    tuned_lambda_regs_and_gammas_file = os.path.join(tuned_params_dir, 'tuned_lambda_regs_and_gammas.pkl')
    write_pickle_file(tuned_lambda_regs_and_gammas_file, tuned_lambda_regs_and_gammas)

    # 2. Lasso
    tuned_deltas = tune_model_beer(m_u, m_v, m_w, c, A_u, A_v, A_w, d, v, w, N_train, xi_train, u_train, x_star_train, y_u_star_train, y_v_star_train, y_w_star_train, 'lasso', deltas, reps, feas_tol, opt_tol)
    tuned_deltas_file = os.path.join(tuned_params_dir, 'tuned_deltas.pkl')
    write_pickle_file(tuned_deltas_file, tuned_deltas)

    print('Finished hyperparameter tuning')

    ### Train models ###
    for N_train_frac in [int((i/4) * N_train) for i in range(1, 5)]:

        # Set directories for models
        model_dir = os.path.join(base_dir, 'beer', 'models', f'N_train_{N_train_frac}')
        runtime_dir = os.path.join(base_dir, 'beer', 'runtime', f'N_train_{N_train_frac}')
        os.makedirs(model_dir, exist_ok = True)
        os.makedirs(runtime_dir, exist_ok = True)

        # Create runtime csv files
        optimistic_dal_runtime_file = create_runtime_file_for_a_model('optimistic_dal', ['rep', 'N_train', 'status', 'total_time', 'optimistic_dal_obj_val'], runtime_dir)
        primal_dal_gamma_0_runtime_file = create_runtime_file_for_a_model('primal_dal_gamma_0', ['rep', 'N_train', 'soln_method', 'lambda_reg', 'gamma', 'num_iters', 'total_time', 'primal_dal_obj_val', 'obj_val_duality_gap', 'obj_val_reg', 'obj_val_pen'], runtime_dir)
        primal_dal_gamma_pos_runtime_file = create_runtime_file_for_a_model('primal_dal_gamma_pos', ['rep', 'N_train', 'soln_method', 'lambda_reg', 'gamma', 'num_iters', 'total_time', 'primal_dal_obj_val', 'obj_val_duality_gap', 'obj_val_reg', 'obj_val_pen'], runtime_dir)
        dual_dal_runtime_file = create_runtime_file_for_a_model('dual_dal', ['rep', 'N_train', 'alpha', 'status', 'total_time', 'dual_dal_obj_val', 'avg_duality_gap_w_abs'], runtime_dir)
        lr_runtime_file = create_runtime_file_for_a_model('lr', ['rep', 'N_train', 'total_time'], runtime_dir)
        lasso_runtime_file = create_runtime_file_for_a_model('lasso', ['rep', 'N_train', 'delta', 'total_time'], runtime_dir)
        rf_runtime_file = create_runtime_file_for_a_model('rf', ['rep', 'N_train', 'total_time'], runtime_dir)

        with concurrent.futures.ProcessPoolExecutor(max_workers = os.cpu_count() - 1) as executor:
            futures = [executor.submit(train_optimistic_dal_beer_and_write_out, m_u, c, A_u, v, w, d, N_train_frac, xi_train, x_star_train, y_u_star_train, y_v_star_train, y_w_star_train, rep, model_dir, optimistic_dal_runtime_file) for rep in reps]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f'Task failed with exception {e}')

        with concurrent.futures.ProcessPoolExecutor(max_workers = os.cpu_count() - 1) as executor:
            futures = [executor.submit(train_primal_dal_beer_and_write_out, m_u, m_v, m_w, c, A_u, A_v, A_w, v, w, d, N_train_frac, xi_train, u_train, x_star_train, rep, tuned_lambda_regs_gamma_0[N_train_frac], 0, 'acs', feas_tol, opt_tol, model_dir, primal_dal_gamma_0_runtime_file) for rep in reps]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f'Task failed with exception {e}')

        with concurrent.futures.ProcessPoolExecutor(max_workers = os.cpu_count() - 1) as executor:
            futures = [executor.submit(train_primal_dal_beer_and_write_out, m_u, m_v, m_w, c, A_u, A_v, A_w, v, w, d, N_train_frac, xi_train, u_train, x_star_train, rep, tuned_lambda_regs_and_gammas[N_train_frac][0], tuned_lambda_regs_and_gammas[N_train_frac][1], 'acs', feas_tol, opt_tol, model_dir, primal_dal_gamma_pos_runtime_file) for rep in reps]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f'Task failed with exception {e}')
        
        with concurrent.futures.ProcessPoolExecutor(max_workers = os.cpu_count() - 1) as executor:
            futures = [executor.submit(train_dual_dal_beer_and_write_out, n, m_u, c, A_u, A_v, A_w, v, w, d, N_train_frac, xi_train, u_train, x_star_train, y_u_star_train, y_v_star_train, y_w_star_train, rep, 2, model_dir, dual_dal_runtime_file) for rep in reps]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f'Task failed with exception {e}')

        with concurrent.futures.ProcessPoolExecutor(max_workers = os.cpu_count() - 1) as executor:
            futures = [executor.submit(train_regression_model_beer_and_write_out, m_u, d, N_train_frac, xi_train, u_train, rep, 'lr', model_dir, lr_runtime_file) for rep in reps]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f'Task failed with exception {e}')

        with concurrent.futures.ProcessPoolExecutor(max_workers = os.cpu_count() - 1) as executor:
            futures = [executor.submit(train_regression_model_beer_and_write_out, m_u, d, N_train_frac, xi_train, u_train, rep, 'lasso', model_dir, lasso_runtime_file, tuned_deltas = tuned_deltas) for rep in reps]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f'Task failed with exception {e}')

        with concurrent.futures.ProcessPoolExecutor(max_workers = os.cpu_count() - 1) as executor:
            futures = [executor.submit(train_regression_model_beer_and_write_out, m_u, d, N_train_frac, xi_train, u_train, rep, 'rf', model_dir, rf_runtime_file) for rep in reps]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f'Task failed with exception {e}')

    print('Trained all models')

    ### Test models ###
    for N_train_frac in [int((i/4) * N_train) for i in range(1, 5)]:

        # Set results directory
        results_dir = os.path.join(base_dir, 'beer', 'results', f'N_train_{N_train_frac}')
        os.makedirs(results_dir, exist_ok = True)

        with concurrent.futures.ProcessPoolExecutor(max_workers = os.cpu_count() - 1) as executor:
            futures = [executor.submit(read_and_evaluate_models_beer_at_rep_level, n, m_u, m_v, m_w, c, A_u, A_v, A_w, v, w, N_test, xi_test, u_test, x_star_test, y_u_star_test, y_v_star_test, y_w_star_test, N_train_frac, rep, feas_tol, opt_tol, base_dir, results_dir) for rep in reps]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f'Task failed with exception {e}')

    print('Tested all models')

    return None

def read_and_evaluate_models_beer_at_rep_level(n, m_u, m_v, m_w, c, A_u, A_v, A_w, v, w, N_test, xi_test, u_test, x_star_test, y_u_star_test, y_v_star_test, y_w_star_test, N_train_frac, rep, feas_tol, opt_tol, base_dir, results_dir):
    
    # Read in models
    optimistic_dal_file = os.path.join(base_dir, 'beer', 'models', f'N_train_{N_train_frac}', f'W_hat_optimistic_dal_rep_{rep}.npy')
    W_hat_optimistic_dal = np.load(optimistic_dal_file)

    primal_dal_gamma_0_file = os.path.join(base_dir, 'beer', 'models', f'N_train_{N_train_frac}', f'W_hat_primal_dal_gamma_0_rep_{rep}.npy')
    W_hat_primal_dal_gamma_0 = np.load(primal_dal_gamma_0_file)

    primal_dal_gamma_pos_file = os.path.join(base_dir, 'beer', 'models', f'N_train_{N_train_frac}', f'W_hat_primal_dal_gamma_pos_rep_{rep}.npy')
    W_hat_primal_dal_gamma_pos = np.load(primal_dal_gamma_pos_file)

    dual_dal_file = os.path.join(base_dir, 'beer', 'models', f'N_train_{N_train_frac}', f'W_hat_dual_dal_rep_{rep}.npy')
    W_hat_dual_dal = np.load(dual_dal_file)

    lr_file = os.path.join(base_dir, 'beer', 'models', f'N_train_{N_train_frac}', f'W_hat_lr_rep_{rep}.npy')
    W_hat_lr = np.load(lr_file)

    lasso_file = os.path.join(base_dir, 'beer', 'models', f'N_train_{N_train_frac}', f'W_hat_lasso_rep_{rep}.npy')
    W_hat_lasso = np.load(lasso_file)

    rf_file = os.path.join(base_dir, 'beer', 'models', f'N_train_{N_train_frac}', f'rf_rep_{rep}.pkl')
    rf = read_pickle_file(rf_file)

    # Evaluate models
    evaluate_model_beer(n, m_u, m_v, m_w, c, A_u, A_v, A_w, v, w, N_test, xi_test, u_test, x_star_test, y_u_star_test, y_v_star_test, y_w_star_test, rep, 'optimistic_dal', W_hat_optimistic_dal, feas_tol, opt_tol, results_dir)
    evaluate_model_beer(n, m_u, m_v, m_w, c, A_u, A_v, A_w, v, w, N_test, xi_test, u_test, x_star_test, y_u_star_test, y_v_star_test, y_w_star_test, rep, 'primal_dal_gamma_0', W_hat_primal_dal_gamma_0, feas_tol, opt_tol, results_dir)
    evaluate_model_beer(n, m_u, m_v, m_w, c, A_u, A_v, A_w, v, w, N_test, xi_test, u_test, x_star_test, y_u_star_test, y_v_star_test, y_w_star_test, rep, 'primal_dal_gamma_pos', W_hat_primal_dal_gamma_pos, feas_tol, opt_tol, results_dir)
    evaluate_model_beer(n, m_u, m_v, m_w, c, A_u, A_v, A_w, v, w, N_test, xi_test, u_test, x_star_test, y_u_star_test, y_v_star_test, y_w_star_test, rep, 'dual_dal', W_hat_dual_dal, feas_tol, opt_tol, results_dir)
    evaluate_model_beer(n, m_u, m_v, m_w, c, A_u, A_v, A_w, v, w, N_test, xi_test, u_test, x_star_test, y_u_star_test, y_v_star_test, y_w_star_test, rep, 'lr', W_hat_lr, feas_tol, opt_tol, results_dir)
    evaluate_model_beer(n, m_u, m_v, m_w, c, A_u, A_v, A_w, v, w, N_test, xi_test, u_test, x_star_test, y_u_star_test, y_v_star_test, y_w_star_test, rep, 'lasso', W_hat_lasso, feas_tol, opt_tol, results_dir)
    evaluate_model_beer(n, m_u, m_v, m_w, c, A_u, A_v, A_w, v, w, N_test, xi_test, u_test, x_star_test, y_u_star_test, y_v_star_test, y_w_star_test, rep, 'rf', rf, feas_tol, opt_tol, results_dir)

    return None

### Solving the downstream linear program and its dual ###

def solve_downstream_lp_beer(n, c, A_u, A_v, A_w, u, v, w, feas_tol, opt_tol):
    
    model = gp.Model()
    model.params.FeasibilityTol = feas_tol
    model.params.OptimalityTol = opt_tol
    x = model.addMVar(n, lb = 0, name = 'x')
    model.setObjective(c @ x, GRB.MINIMIZE)
    model.addConstr(A_u @ x >= u)
    model.addConstr(A_v @ x == v)
    model.addConstr(A_w @ x >= w)
    model.update()
    model.optimize()

    if model.status == GRB.OPTIMAL:
        return model.X, model.ObjVal
    else:
        return None, None

def solve_dual_downstream_lp_beer(m_u, m_v, m_w, c, A_u, A_v, A_w, u, v, w, feas_tol, opt_tol):
    
    model = gp.Model()
    model.params.FeasibilityTol = feas_tol
    model.params.OptimalityTol = opt_tol
    y_u = model.addMVar(m_u, lb = 0, name = 'y_u')
    y_v = model.addMVar(m_v, lb = -np.inf, name = 'y_v')
    y_w = model.addMVar(m_w, lb = 0, name = 'y_w')
    model.setObjective(u @ y_u + v @ y_v + w @ y_w, GRB.MAXIMIZE)
    model.addConstr(A_u.T @ y_u + A_v.T @ y_v + A_w.T @ y_w <= c)
    model.update()
    model.optimize()

    if model.status == GRB.OPTIMAL:
        return y_u.X, y_v.X, y_w.X, model.ObjVal
    else:
        return None, None, None, None

### Learning models ###

def optimistic_dal_beer(m_u, c, A_u, v, w, d, N, xi, x_star, y_u_star, y_v_star, y_w_star):

    model = gp.Model()
    W = model.addMVar((m_u, d + 1), lb = -np.inf, name = 'W')
    model.setObjective((1/N) * gp.quicksum(c @ x_star[i] - (W @ xi[i] @ y_u_star[i] + v @ y_v_star[i] + w @ y_w_star[i]) for i in range(N)), GRB.MINIMIZE)
    model.addConstrs(A_u @ x_star[i] >= W @ xi[i] for i in range(N))
    zero_indices = get_zero_indices_of_model_beer(m_u)
    model.addConstrs(W[i, j] == 0 for (i, j) in zero_indices)
    model.update()
    model.optimize()

    if model.status == GRB.OPTIMAL:
        return W.X, model.status
    else:
        print('The status of the Optimistic-DAL problem is not \'optimal\'')
        return None, model.status

def get_zero_indices_of_model_beer(m_u):

    return [(i, j) for i in range(m_u) for j in range (1, m_u + 1) if j != i + 1]

def primal_dal_beer_acs(m_u, m_v, m_w, c, A_u, A_v, A_w, v, w, d, W, N, xi, u, x_star, y_u, y_v, y_w, lambda_reg, gamma, max_iters, eps):

    start_time_algorithm = time.time()
    prev_obj_value = compute_primal_dal_obj_val_beer(m_u, c, v, w, d, W, N, xi, u, x_star, y_u, y_v, y_w, lambda_reg, gamma)

    for k_iter in range(max_iters):

        # Check if the code has hit the maximum time limit
        if time.time() - start_time_algorithm >= 3600:
            return W, y_u, y_v, y_w, k_iter + 1

        # Step 1: solve for W while keeping y_u, y_v, and y_w fixed
        W_var = cp.Variable((m_u, d + 1))
        W_subproblem_objective = (1/N) * cp.sum([c @ x_star[i] - (W_var @ xi[i] @ y_u[i] + v @ y_v[i] + w @ y_w[i]) for i in range(N)]) + \
            lambda_reg * cp.norm1(W_var) + \
                gamma * cp.sum([cp.maximum(0, u[i, j] - W_var[j] @ xi[i]) for i in range(N) for j in range(m_u)])
        zero_indices = get_zero_indices_of_model_beer(m_u)
        W_subproblem_constrs_zero = [W_var[i, j] == 0 for (i, j) in zero_indices]
        W_subproblem_constrs_feas = [A_u @ x_star[i] >= W_var @ xi[i] for i in range(N)]
        W_subproblem_constrs = W_subproblem_constrs_zero + W_subproblem_constrs_feas
        W_subproblem = cp.Problem(cp.Minimize(W_subproblem_objective), W_subproblem_constrs)
        W_subproblem.solve(cp.GUROBI)
        W = W_var.value

        # Step 2: solve for y_u, y_v, and y_w while keeping W fixed
        y_u_var = cp.Variable((N, m_u))
        y_v_var = cp.Variable((N, m_v))
        y_w_var = cp.Variable((N, m_w))
        y_subproblem_objective = (1/N) * cp.sum([c @ x_star[i] - (W @ xi[i] @ y_u_var[i] + v @ y_v_var[i] + w @ y_w_var[i]) for i in range(N)]) 
        y_subproblem_constrs = [A_u.T @ y_u_var[i] + A_v.T @ y_v_var[i] + A_w.T @ y_w_var[i] <= c for i in range(N)]
        y_subproblem_constrs.append(y_u_var >= 0)
        y_subproblem_constrs.append(y_w_var >= 0)
        y_subproblem = cp.Problem(cp.Minimize(y_subproblem_objective), y_subproblem_constrs)
        y_subproblem.solve(cp.GUROBI)
        y_u = y_u_var.value
        y_v = y_v_var.value
        y_w = y_w_var.value
        y_u = {i : y_u[i] for i in range(N)}
        y_v = {i : y_v[i] for i in range(N)}
        y_w = {i : y_w[i] for i in range(N)}

        # Step 3: compute new objective value
        new_obj_value = compute_primal_dal_obj_val_beer(m_u, c, v, w, d, W, N, xi, u, x_star, y_u, y_v, y_w, lambda_reg, gamma)

        # Step 4: check for convergence and update previous objective value, if needed
        if new_obj_value == prev_obj_value or (prev_obj_value - new_obj_value)/prev_obj_value < eps: # the first condition takes care of the case where new_obj_value == prev_obj_value == 0
            break
        prev_obj_value = new_obj_value
    
    return W, y_u, y_v, y_w, k_iter + 1

def dual_dal_beer(n, m_u, c, A_u, A_v, A_w, v, w, d, N, xi, u, y_u_star, y_v_star, y_w_star):
    
    model = gp.Model()
    alpha_dual_dal = 2
    W = model.addMVar(shape = (m_u, d + 1), lb = -np.inf, name = 'W')
    x = {i : model.addMVar(shape = (n, ), lb = 0, name = f'x[{i}]') for i in range(N)}
    model.setObjective((1/N) * gp.quicksum(c @ x[i] - ((alpha_dual_dal * (W @ xi[i]) - u[i]) @ y_u_star[i] + ((alpha_dual_dal - 1) * v) @ y_v_star[i] + ((alpha_dual_dal - 1) * w) @ y_w_star[i]) for i in range(N)), GRB.MINIMIZE)
    model.addConstrs(A_u @ x[i] >= alpha_dual_dal * (W @ xi[i]) - u[i] for i in range(N))
    model.addConstrs(A_v @ x[i] == (alpha_dual_dal - 1) * v for i in range(N))
    model.addConstrs(A_w @ x[i] >= (alpha_dual_dal - 1) * w for i in range(N))
    zero_indices = get_zero_indices_of_model_beer(m_u)
    model.addConstrs(W[i, j] == 0 for (i, j) in zero_indices)
    model.update()
    model.optimize()

    if model.status == GRB.OPTIMAL:
        return W.X, {i : x[i].X for i in range(N)}, model.status
    else:
        return None, None, model.status

def lr_beer(m_u, d, xi, u):

    W_var = cp.Variable((m_u, d + 1))
    objective = cp.Minimize(cp.sum_squares(xi @ W_var.T - u))
    zero_indices = get_zero_indices_of_model_beer(m_u)
    constrs = [W_var[i, j] == 0 for (i, j) in zero_indices]
    problem = cp.Problem(objective, constrs)
    problem.solve()

    return W_var.value

def lasso_beer(m_u, d, xi, u, delta):

    W_var = cp.Variable((m_u, d + 1))
    objective = cp.Minimize(cp.sum_squares(xi @ W_var.T - u) + delta * cp.norm1(W_var))
    zero_indices = get_zero_indices_of_model_beer(m_u)
    constrs = [W_var[i, j] == 0 for (i, j) in zero_indices]
    problem = cp.Problem(objective, constrs)
    problem.solve()

    return W_var.value

### Evaluating the objective function of the different learning problems for a given solution ###

def compute_avg_duality_gap_beer(c, v, w, W, N, xi, x, y_u, y_v, y_w):
    
    return (1/N) * sum([c @ x[i] - (W @ xi[i] @ y_u[i] + v @ y_v[i] + w @ y_w[i]) for i in range(N)])

def compute_avg_duality_gap_w_abs_beer(c, v, w, W, N, xi, u, x, y_u, y_v, y_w):
    
    return (1/N) * sum([abs(c @ x[i] - (W @ xi[i] @ y_u[i] + v @ y_v[i] + w @ y_w[i])) for i in range(N)])

def compute_primal_dal_obj_val_beer(m_u, c, v, w, d, W, N, xi, u, x, y_u, y_v, y_w, lambda_reg, gamma):
    
    return (1/N) * sum([c @ x[i] - (W @ xi[i] @ y_u[i] + v @ y_v[i] + w @ y_w[i]) for i in range(N)]) + lambda_reg * sum([abs(W[j, k]) for j in range(m_u) for k in range(d + 1)]) + gamma * sum([max(0, u[i, j] - W[j] @ xi[i]) for i in range(N) for j in range(m_u)])

def compute_primal_dal_obj_val_beer_by_term(m_u, c, v, w, d, W, N, xi, u, x, y_u, y_v, y_w, lambda_reg, gamma):
    
    return (1/N) * sum([c @ x[i] - (W @ xi[i] @ y_u[i] + v @ y_v[i] + w @ y_w[i]) for i in range(N)]), lambda_reg * sum([abs(W[j, k]) for j in range(m_u) for k in range(d + 1)]), gamma * sum([max(0, u[i, j] - W[j] @ xi[i]) for i in range(N) for j in range(m_u)])

def compute_dual_dal_obj_val_beer(c, v, w, W, N, xi, u, x, y_u, y_v, y_w):

    alpha = 2
    return (1/N) * sum([c @ x[i] - ((alpha * (W @ xi[i]) - u[i]) @ y_u[i] + ((alpha - 1) * v) @ y_v[i] + ((alpha - 1) * w) @ y_w[i]) for i in range(N)])

### Oragnizing data for different replications ###

def create_feature_target_vectors_beer(df):
    
    # Instantiate data structures
    xi = []
    b = []

    # Iterate over the dates in the dataset
    for date in df['date'].unique():

        # Collect feature data from the current date
        daily_data = df[df['date'] == date]
        daily_data = daily_data.sort_values(by = 'name')
        tavg_vector = daily_data[['tavg']].values.flatten()
        month_2 = np.array([[daily_data['month_2'].values[0]]])
        month_3 = np.array([[daily_data['month_3'].values[0]]])
        month_4 = np.array([[daily_data['month_4'].values[0]]])
        month_5 = np.array([[daily_data['month_5'].values[0]]])
        month_6 = np.array([[daily_data['month_6'].values[0]]])
        month_7 = np.array([[daily_data['month_7'].values[0]]])
        month_8 = np.array([[daily_data['month_8'].values[0]]])
        month_9 = np.array([[daily_data['month_9'].values[0]]])
        month_10 = np.array([[daily_data['month_10'].values[0]]])
        month_11 = np.array([[daily_data['month_11'].values[0]]])
        month_12 = np.array([[daily_data['month_12'].values[0]]])
        day_of_week_Monday = np.array([[daily_data['day_of_week_Monday'].values[0]]])
        day_of_week_Tuesday = np.array([[daily_data['day_of_week_Tuesday'].values[0]]])
        day_of_week_Wednesday = np.array([[daily_data['day_of_week_Wednesday'].values[0]]])
        day_of_week_Thursday = np.array([[daily_data['day_of_week_Thursday'].values[0]]])

        # Store feature data in a vector
        feature_vector = np.concatenate(
            (
                tavg_vector,
                month_2.flatten(),
                month_3.flatten(),
                month_4.flatten(),
                month_5.flatten(),
                month_6.flatten(),
                month_7.flatten(),
                month_8.flatten(),
                month_9.flatten(),
                month_10.flatten(),
                month_11.flatten(),
                month_12.flatten(),
                day_of_week_Monday.flatten(),
                day_of_week_Tuesday.flatten(),
                day_of_week_Wednesday.flatten(),
                day_of_week_Thursday.flatten()
             )
            )
        
        # Form final (xi, b) pair for the current date
        xi.append(feature_vector)
        b.append(daily_data['SALES'].values)
    
    return np.array(xi), np.array(b)

def get_train_test_splits_beer(xi, u, reps):

    # Set training and test dataset sizes
    N_train = int(np.floor(0.75 * len(xi)))
    N_test = len(xi) - N_train

    # Instantiate dictionaries for the training and test datasets over the different replications
    xi_train = {}
    xi_test = {}
    u_train = {}
    u_test = {}

    for rep in reps:
        xi_train[rep], xi_test[rep], u_train[rep], u_test[rep] = train_test_split(xi, u, train_size = N_train, test_size = N_test, random_state = rep)

    return xi_train, xi_test, u_train, u_test

### Training and tuning functions ###

def generate_primal_dal_beer_initial_solution(m_u, m_v, m_w, c, A_u, A_v, A_w, d, N, xi, x_star, feas_tol, opt_tol):
    
    model = gp.Model()
    model.params.FeasibilityTol = feas_tol
    model.params.OptimalityTol = opt_tol
    W = model.addMVar((m_u, d + 1), lb = -np.inf, name = 'W')
    y_u = {i : model.addMVar(shape = (m_u, ), lb = 0, name = f'y_u[{i}]') for i in range(N)}
    y_v = {i : model.addMVar(shape = (m_v, ), lb = -np.inf, name = f'y_v[{i}]') for i in range(N)}
    y_w = {i : model.addMVar(shape = (m_w, ), lb = 0, name = f'y_w[{i}]') for i in range(N)}
    model.setObjective(0, GRB.MINIMIZE)
    model.addConstrs(A_u @ x_star[i] >= W @ xi[i] for i in range(N))
    model.addConstrs(A_u.T @ y_u[i] + A_v.T @ y_v[i] + A_w.T @ y_w[i] <= c for i in range(N))
    zero_indices = get_zero_indices_of_model_beer(m_u)
    model.addConstrs(W[i, j] == 0 for (i, j) in zero_indices)
    model.update()
    model.optimize()

    if model.status == GRB.OPTIMAL:
        return W.X, {i : y_u[i].X for i in range(N)}, {i : y_v[i].X for i in range(N)}, {i : y_w[i].X for i in range(N)}
    else:
        print('The Primal-DAL problem is infeasible')
        return None, None, None, None

def tune_model_beer(m_u, m_v, m_w, c, A_u, A_v, A_w, v, w, d, N, xi, u, x_star, y_u_star, y_v_star, y_w_star, model_name, params, training_reps, feas_tol, opt_tol):

    cv_loss = {}
    pooled_cv_loss = {}
    tuned_param = {}

    for N_frac in [int((i/4) * N) for i in range(1, 5)]:
        for param in params:

            # Conduct each replications's tuning in parallel
            with concurrent.futures.ProcessPoolExecutor(max_workers = os.cpu_count() - 1) as executor:
                futures = [executor.submit(tune_model_beer_at_rep_level, m_u, m_v, m_w, c, A_u, A_v, A_w, v, w, d, xi, u, x_star, y_u_star, y_v_star, y_w_star, model_name, N_frac, param, rep, feas_tol, opt_tol) for rep in training_reps]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        rep, loss_val = future.result()
                        cv_loss[N_frac, param, rep] = loss_val
                    except Exception as e:
                        print(f'Task failed with exception {e}')

            # Pooling over all replications
            pooled_cv_loss[N_frac, param] = (1/len(training_reps)) * sum([cv_loss[N_frac, param, rep] for rep in training_reps])

        # For a fixed value of N_frac, find param such that pooled_cv_loss[N_frac, param] is minimal
        minimizing_key = min((k for k in pooled_cv_loss if k[0] == N_frac), key = pooled_cv_loss.get)
        tuned_param[N_frac] = minimizing_key[1]

    return tuned_param

def tune_model_beer_at_rep_level(m_u, m_v, m_w, c, A_u, A_v, A_w, v, w, d, xi, u, x_star, y_u_star, y_v_star, y_w_star, model_name, N_frac, param, rep, feas_tol, opt_tol):

    # Perform cross-validation
    fold_loss = {}
    fold_feas_count = 0
    kf = KFold(n_splits = 5, shuffle = True, random_state = 0)
    for fold, (train_idx, holdout_idx) in enumerate(kf.split(xi[rep][ : N_frac]), 1):

        # Split the training data into cross-validation training and cross-validation holdout datasets
        xi_cv_train, xi_cv_holdout = xi[rep][train_idx], xi[rep][holdout_idx]
        u_cv_train, u_cv_holdout = u[rep][train_idx], u[rep][holdout_idx]
        x_star_cv_train, x_star_cv_holdout = x_star[rep][train_idx], x_star[rep][holdout_idx]
        _, y_u_star_cv_holdout = y_u_star[rep][train_idx], y_u_star[rep][holdout_idx]
        _, y_v_star_cv_holdout = y_v_star[rep][train_idx], y_v_star[rep][holdout_idx]
        _, y_w_star_cv_holdout = y_w_star[rep][train_idx], y_w_star[rep][holdout_idx]

        # Train the model
        if model_name == 'primal_dal':
            W_0_acs, y_u_0_acs, y_v_0_acs, y_w_0_acs = generate_primal_dal_beer_initial_solution(m_u, m_v, m_w, c, A_u, A_v, A_w, d, len(train_idx), xi_cv_train, x_star_cv_train, feas_tol, opt_tol)
            W_hat, _, _, _, _ = primal_dal_beer_acs(m_u, m_v, m_w, c, A_u, A_v, A_w, v, w, d, W_0_acs, len(train_idx), xi_cv_train, u_cv_train, x_star_cv_train, y_u_0_acs, y_v_0_acs, y_w_0_acs, param[0], param[1], max_iters = 100, eps = 1e-2)
        elif model_name == 'lasso':
            W_hat = lasso_beer(m_u, d, xi_cv_train, u_cv_train, param)
        else:
            # add code to train odece here!!!
            pass

        # Evaluate model on holdout dataset
        if model_name in ['primal_dal', 'lasso']:
            u_hat = make_predictions(W_hat, 'matrix', len(holdout_idx), xi_cv_holdout)
        else: # model_name == 'odece
            u_hat = make_predictions(W_hat, 'odece', len(holdout_idx), xi_cv_holdout)
        _, feas, duality_gap = compute_true_soln_metrics_beer(c, A_u, v, w, len(holdout_idx), u_hat, x_star_cv_holdout, y_u_star_cv_holdout, y_v_star_cv_holdout, y_w_star_cv_holdout, feas_tol)

        # Loss for a specific fold
        if model_name != 'lasso':
            fold_loss[fold] = sum(duality_gap.values()) # decision loss (optimality gap)
            fold_feas_count += sum(feas.values())
        else:
            u_hat_arr = np.zeros((len(holdout_idx), m_u))
            for i in range(len(holdout_idx)):
                u_hat_arr[i] = u_hat[i]
            fold_loss[fold] = mean_squared_error(u_hat_arr, u_cv_holdout) # prediction loss

    # Mean CV loss across 5 folds
    if model_name != 'lasso':
        return rep, (1/fold_feas_count) * sum([fold_loss[fold] for fold, _ in enumerate(kf.split(xi[rep][ : N_frac]), 1)])
    else:
        return rep, (1/len(fold_loss)) * sum([fold_loss[fold] for fold, _ in enumerate(kf.split(xi[rep][ : N_frac]), 1)])

def train_optimistic_dal_beer_and_write_out(m_u, c, A_u, v, w, d, N, xi, x_star, y_u_star, y_v_star, y_w_star, rep, model_dir, runtime_file):

    # Train model
    start_time = time.time()
    W_hat, status = optimistic_dal_beer(m_u, c, A_u, v, w, d, N, xi[rep], x_star[rep], y_u_star[rep], y_v_star[rep], y_w_star[rep])
    end_time = time.time()
    total_time = end_time - start_time
    if W_hat is None:
        optimistic_dal_obj_val = None
        write_model(f'W_hat_optimistic_dal_rep_{rep}', W_hat, model_dir, to_pickle = True)
    else:
        optimistic_dal_obj_val = compute_avg_duality_gap_beer(c, v, w, W_hat, N, xi[rep], x_star[rep], y_u_star[rep], y_v_star[rep], y_w_star[rep])
        write_model(f'W_hat_optimistic_dal_rep_{rep}', W_hat, model_dir, to_pickle = False)
    
    # Write out runtime results
    with open(runtime_file, 'a', newline = '') as file:
        writer = csv.writer(file)
        writer.writerow([rep, N, status, total_time, optimistic_dal_obj_val])

    return None

def train_primal_dal_beer_and_write_out(m_u, m_v, m_w, c, A_u, A_v, A_w, v, w, d, N, xi, u, x_star, rep, lambda_reg, gamma, soln_method, feas_tol, opt_tol, model_dir, runtime_file):

    # Generate initial solution
    W_0_acs, y_u_0_acs, y_v_0_acs, y_w_0_acs = generate_primal_dal_beer_initial_solution(m_u, m_v, m_w, c, A_u, A_v, A_w, d, N, xi[rep], x_star[rep], feas_tol, opt_tol)
    
    # Train model
    start_time = time.time()
    W_hat, y_u_hat, y_v_hat, y_w_hat, num_iters = primal_dal_beer_acs(m_u, m_v, m_w, c, A_u, A_v, A_w, v, w, d, W_0_acs, N, xi[rep], u[rep], x_star[rep], y_u_0_acs, y_v_0_acs, y_w_0_acs, lambda_reg, gamma, max_iters = 100, eps = 1e-2)
    end_time = time.time()
    total_time = end_time - start_time
    primal_dal_obj_val = compute_primal_dal_obj_val_beer(m_u, c, v, w, d, W_hat, N, xi[rep], u[rep], x_star[rep], y_u_hat, y_v_hat, y_w_hat, lambda_reg, gamma)
    obj_val_duality_gap, obj_val_reg, obj_val_pen = compute_primal_dal_obj_val_beer_by_term(m_u, c, v, w, d, W_hat, N, xi[rep], u[rep], x_star[rep], y_u_hat, y_v_hat, y_w_hat, lambda_reg, gamma)
    if gamma == 0:
        write_model(f'W_hat_primal_dal_gamma_0_rep_{rep}', W_hat, model_dir, to_pickle = False)
        write_model(f'y_u_hat_primal_dal_gamma_0_rep_{rep}', y_u_hat, model_dir, to_pickle = True)
        write_model(f'y_v_hat_primal_dal_gamma_0_rep_{rep}', y_v_hat, model_dir, to_pickle = True)
        write_model(f'y_w_hat_primal_dal_gamma_0_rep_{rep}', y_w_hat, model_dir, to_pickle = True)
    else:
        write_model(f'W_hat_primal_dal_gamma_pos_rep_{rep}', W_hat, model_dir, to_pickle = False)
        write_model(f'y_u_hat_primal_dal_gamma_pos_rep_{rep}', y_u_hat, model_dir, to_pickle = True)
        write_model(f'y_v_hat_primal_dal_gamma_pos_rep_{rep}', y_v_hat, model_dir, to_pickle = True)
        write_model(f'y_w_hat_primal_dal_gamma_pos_rep_{rep}', y_w_hat, model_dir, to_pickle = True)
    
    # Write out runtime results
    with open(runtime_file, 'a', newline = '') as file:
        writer = csv.writer(file)
        writer.writerow([rep, N, soln_method, lambda_reg, gamma, num_iters, total_time, primal_dal_obj_val, obj_val_duality_gap, obj_val_reg, obj_val_pen])

    return None

def train_dual_dal_beer_and_write_out(n, m_u, c, A_u, A_v, A_w, v, w, d, N, xi, u, x_star, y_u_star, y_v_star, y_w_star, rep, alpha, model_dir, runtime_file):

    # Train model
    start_time = time.time()
    W_hat, x_hat, status = dual_dal_beer(n, m_u, c, A_u, A_v, A_w, v, w, d, N, xi[rep], u[rep], y_u_star[rep], y_v_star[rep], y_w_star[rep])
    end_time = time.time()
    total_time = end_time - start_time
    if W_hat is None:
        dual_dal_obj_val, avg_duality_gap_w_abs = None, None
        write_model(f'W_hat_dual_dal_rep_{rep}', W_hat, model_dir, to_pickle = True)
    else:
        dual_dal_obj_val = compute_dual_dal_obj_val_beer(c, v, w, W_hat, N, xi[rep], u[rep], x_hat, y_u_star[rep], y_v_star[rep], y_w_star[rep])
        avg_duality_gap_w_abs = compute_avg_duality_gap_w_abs_beer(c, v, w, W_hat, N, xi[rep], u[rep], x_star[rep], y_u_star[rep], y_v_star[rep], y_w_star[rep]) # NOTE: since the Dual-DAL solution (W_hat, (x_hat)_i) might not satisfy predicted feasibility in the primal space over the datapoints i, we must evaluate the average optimality gap objective function with absolute values
        write_model(f'W_hat_dual_dal_rep_{rep}', W_hat, model_dir, to_pickle = False)
    write_model(f'x_hat_dual_dal_rep_{rep}', x_hat, model_dir, to_pickle = True)

    # Write out runtime results
    with open(runtime_file, 'a', newline = '') as file:
        writer = csv.writer(file)
        writer.writerow([rep, N, alpha, status, total_time, dual_dal_obj_val, avg_duality_gap_w_abs])

    return None

def train_regression_model_beer_and_write_out(m_u, d, N, xi, u, rep, model_name, model_dir, runtime_file, tuned_deltas = None):

    # Train model
    if model_name == 'lr':
        start_time = time.time()
        model = lr_beer(m_u, d, xi[rep][ : N], u[rep][ : N])
        end_time = time.time()
    elif model_name == 'lasso':
        start_time = time.time()
        model = lasso_beer(m_u, d, xi[rep][ : N], u[rep][ : N], tuned_deltas[N])
        end_time = time.time()
    else:
        start_time = time.time()
        model = RandomForestRegressor(max_features = int(np.ceil(d/3)), random_state = 0).fit(xi[rep][ : N], u[rep][ : N])
        end_time = time.time()
    total_time = end_time - start_time

    # Write out model
    if model_name in ['lr', 'lasso']:
        write_model('W_hat_' + model_name + f'_rep_{rep}', model, model_dir, to_pickle = False)
    else:
        write_model(model_name + f'_rep_{rep}', model, model_dir, to_pickle = True)
    
    # Write out runtime data
    with open(runtime_file, 'a', newline = '') as file:
        writer = csv.writer(file)
        if model_name in ['lr', 'rf']:
            writer.writerow([rep, N, total_time])
        else:
            writer.writerow([rep, N, tuned_deltas[N], total_time])

    return None

### Analyzing results ###

def compute_true_soln_metrics_beer(c, A_u, v, w, N, u_hat, x_star, y_u_star, y_v_star, y_w_star, feas_tol):

    constr_satisfaction = {}
    feas = {}
    duality_gap = {}

    # Looping over true solutions
    for i in range(N):
        eval = (A_u @ np.array(x_star[i]) >= u_hat[i] - feas_tol)
        constr_satisfaction[i] = sum(eval)
        if all(eval):
            feas[i] = 1
            duality_gap[i] = c @ x_star[i] - (u_hat[i] @ y_u_star[i] + v @ y_v_star[i] + w @ y_w_star[i])
        else:
            feas[i] = 0

    return constr_satisfaction, feas, duality_gap

def compute_predicted_soln_feas_duality_gap_metrics_beer(c, A_u, u, v, w, N, x_hat, y_u_hat, y_v_hat, y_w_hat, feas_tol):
    
    constr_satisfaction = {}
    feas = {}
    duality_gap = {}

    # Looping over predicted solutions for a given model
    for i in range(N):
        if not np.isnan(x_hat[i]).any():
            eval = (A_u @ np.array(x_hat[i]) >= u[i] - feas_tol)
            constr_satisfaction[i] = sum(eval)
            if all(eval):
                feas[i] = 1
                duality_gap[i] = c @ x_hat[i] - (u[i] @ y_u_hat[i] + v @ y_v_hat[i] + w @ y_w_hat[i])
            else:
                feas[i] = 0

    return constr_satisfaction, feas, duality_gap

def compute_predicted_soln_projection_metrics_beer(c, A_u, A_v, A_w, u, v, w, N, x_star, x_hat, opt_tol):

    proj_dist = {}
    cost_opt_gap_proj_and_true_soln = {}

    # Looping over predicted solutions for a given model
    for i in range(N):
        if not np.isnan(x_hat[i]).any():

            # Project x_hat onto the true feasible region
            x_tilde_i = project_onto_polyhedron_beer(A_u, A_v, A_w, u[i], v, w, x_hat[i])

            # Compute the distance from x_hat_i to x_tilde_i
            proj_dist[i] = np.linalg.norm(x_hat[i] - x_tilde_i)

            # Compute the cost optimality gap between the projected and true solutions
            cost_opt_gap_proj_and_true_soln[i] = get_normalized_cost_opt_gap(c, x_tilde_i, x_star[i], opt_tol)

    return proj_dist, cost_opt_gap_proj_and_true_soln

def project_onto_polyhedron_beer(A_u, A_v, A_w, u, v, w, x_hat):

    # Define model, then solve
    n = len(x_hat)
    x_tilde = cp.Variable(n)
    objective = cp.Minimize(cp.sum_squares(x_tilde - x_hat))
    constrs = [A_u @ x_tilde >= u, A_v @ x_tilde == v, A_w @ x_tilde >= w, x_tilde >= 0]
    problem = cp.Problem(objective, constrs)
    problem.solve(solver = cp.GUROBI)

    # Return results
    if problem.status == 'optimal':
        x_tilde = x_tilde.value
        x_tilde.shape = (n,)
        return x_tilde
    else:
        return None

def evaluate_model_beer(n, m_u, m_v, m_w, c, A_u, A_v, A_w, v, w, N, xi, u, x_star, y_u_star, y_v_star, y_w_star, rep, model_name, model, feas_tol, opt_tol, results_dir):
    
    # Create directory that will store the results for the given model
    results_for_model_dir = os.path.join(results_dir, model_name)
    os.makedirs(results_for_model_dir, exist_ok = True)

    # Evaluate models; # TODO: add odece code here!!!
    if model_name != 'rf':
        u_hat = make_predictions(model, N, xi[rep], is_sklearn_model = False)
    else:
        u_hat = make_predictions(model, N, xi[rep], is_sklearn_model = True)
    prediction_error = {(i, j) : u_hat[i][j] - u[rep][i, j] for i in range(N) for j in range(m_u)} # prediction minus truth
    constr_satisfaction_true_soln, feas_true_soln, duality_gap_true_soln = compute_true_soln_metrics_beer(c, A_u, v, w, N, u_hat, x_star[rep], y_u_star[rep], y_v_star[rep], y_w_star[rep], feas_tol)
    x_hat, y_u_hat, y_v_hat, y_w_hat = np.zeros((N, n)), np.zeros((N, m_u)), np.zeros((N, m_v)), np.zeros((N, m_w))
    recover_x_hat = {}
    for i in range(N):
        x_hat[i], z_hat_i = solve_downstream_lp_beer(n, c, A_u, A_v, A_w, u_hat[i], v, w, feas_tol, opt_tol)
        if z_hat_i is None:
            recover_x_hat[i] = 0
        else:
            recover_x_hat[i] = 1
        y_u_hat[i], y_v_hat[i], y_w_hat[i], _ = solve_dual_downstream_lp_beer(m_u, m_v, m_w, c, A_u, A_v, A_w, u_hat[i], v, w, feas_tol, opt_tol)
    constr_satisfaction_predicted_soln, feas_predicted_soln, duality_gap_predicted_soln = compute_predicted_soln_feas_duality_gap_metrics_beer(c, A_u, u[rep], v, w, N, x_hat, y_u_hat, y_v_hat, y_w_hat, feas_tol)
    proj_dist, cost_opt_gap_proj_and_true_soln = compute_predicted_soln_projection_metrics_beer(c, A_u, A_v, A_w, u[rep], v, w, N, x_star[rep], x_hat, opt_tol)

    # Write out data
    write_pickle_file(os.path.join(results_for_model_dir, f'prediction_error_rep_{rep}.pkl'), prediction_error)
    write_pickle_file(os.path.join(results_for_model_dir, f'constr_satisfaction_true_soln_rep_{rep}.pkl'), constr_satisfaction_true_soln)
    write_pickle_file(os.path.join(results_for_model_dir, f'feas_true_soln_rep_{rep}.pkl'), feas_true_soln)
    write_pickle_file(os.path.join(results_for_model_dir, f'duality_gap_true_soln_rep_{rep}.pkl'), duality_gap_true_soln)
    write_pickle_file(os.path.join(results_for_model_dir, f'recover_x_hat_rep_{rep}.pkl'), recover_x_hat)
    write_pickle_file(os.path.join(results_for_model_dir, f'constr_satisfaction_predicted_soln_rep_{rep}.pkl'), constr_satisfaction_predicted_soln)
    write_pickle_file(os.path.join(results_for_model_dir, f'feas_predicted_soln_rep_{rep}.pkl'), feas_predicted_soln)
    write_pickle_file(os.path.join(results_for_model_dir, f'duality_gap_predicted_soln_rep_{rep}.pkl'), duality_gap_predicted_soln)
    write_pickle_file(os.path.join(results_for_model_dir, f'proj_dist_rep_{rep}.pkl'), proj_dist)
    write_pickle_file(os.path.join(results_for_model_dir, f'cost_opt_gap_proj_and_true_soln_rep_{rep}.pkl'), cost_opt_gap_proj_and_true_soln)

    return None

def analyze_beer_experiment_results_for_all_reps(N_train, N_test, base_dir):

    # Grab/set different values to analyze synthetic experiment results
    xi_test = read_pickle_file(os.path.join(base_dir, 'beer', 'test', 'xi.pkl'))
    test_reps = list(xi_test.keys())
    results_dir = os.path.join(base_dir, 'beer', 'results')
    model_names = ['optimistic_dal', 'primal_dal_gamma_0', 'primal_dal_gamma_pos', 'dual_dal', 'lr', 'lasso', 'rf']
    metrics = ['prediction_error', 'constr_satisfaction_true_soln', 'feas_true_soln', 'duality_gap_true_soln', 'recover_x_hat', 'constr_satisfaction_predicted_soln', 'feas_predicted_soln', 'duality_gap_predicted_soln', 'proj_dist', 'cost_opt_gap_proj_and_true_soln']
    
    for metric in metrics:

        # Create metric csv file and obtain results for the current metric
        if metric.startswith('feas') or metric.startswith('recover'):
            output_dict = get_indicator_results(N_train, N_test, metric, test_reps, model_names, results_dir)
        else:
            output_dict = get_numerical_results(N_train, metric, test_reps, model_names, results_dir)

        # Write out results to the above-created metric csv
        output_file = os.path.join(results_dir, f'{metric}.csv')
        for N_train_frac in [int((i/4) * N_train) for i in range(1, 5)]:
            with open(output_file, 'a', newline = '') as file:
                writer = csv.writer(file)
                writer.writerow([N_train_frac] + [output_dict[N_train_frac, model] for model in model_names])
    
    return None