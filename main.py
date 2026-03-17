# !!!! NOTE: be careful with which flags are set to true so as to not overwrite previous results !!!!

# Import statements
import sys
sys.dont_write_bytecode = True
from subroutines import *

# Synthetic experiment flags
set_global_values_for_synthetic_experiment = True # always set this to true when running any part of the synthetic experiment
generate_synthetic_experiment_data = True
run_sensitivity_analysis = False
generate_sensitivity_analysis_plots = False
run_primal_dal_soln_method_comparison = False
analyze_primal_dal_soln_method_comparison_results = False
run_synthetic_experiment = True
analyze_synthetic_experiment_results = True

# Beer network optimization experiment flags
run_beer_experiment = False
analyze_beer_experiment_results = False

def main():
    
    # Set the base directory, i.e., the one within which this file resides 
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    if set_global_values_for_synthetic_experiment:

        print('Setting global values for synthetic experiment \n')
        
        settings = {
            0 : [2, 3, 6, 'uniform', -10, 10, 'linear'], # Batch # 1
            1 : [2, 3, 6, 'uniform', -10, 10, 'quadratic'], # Batch # 1
            2 : [2, 3, 6, 'normal', 0, 3, 'linear'], # Batch # 1
            3 : [2, 3, 6, 'normal', 0, 3, 'quadratic'], # Batch # 1
            4 : [2, 3, 6, 'uniform', -10, 10, 'linear'], # Batch # 2
            5 : [2, 3, 6, 'uniform', -10, 10, 'quadratic'], # Batch # 2
            6 : [2, 3, 6, 'normal', 0, 3, 'linear'], # Batch # 2
            7 : [2, 3, 6, 'normal', 0, 3, 'quadratic'], # Batch # 2
            8 : [2, 3, 6, 'uniform', -10, 10, 'linear'], # Batch # 3
            9 : [2, 3, 6, 'uniform', -10, 10, 'quadratic'], # Batch # 3
            10 : [2, 3, 6, 'normal', 0, 3, 'linear'], # Batch # 3
            11 : [2, 3, 6, 'normal', 0, 3, 'quadratic'] # Batch # 3
        } # dictionary of setting-specific values (each entry specifies a setting)
        # NOTE: the values are as follows:
        # n = number of decision variables
        # m = number of constraints
        # d = number of contextual features
        # P_{\Xi} = distribution of the context vectors
        # P_{\Xi} parameter 1 = either the lower bound of a uniform distribution or the mean of a normal distribution
        # P_{\Xi} parameter 2 = either the upper bound of a uniform distribution or the standard deviation of a normal distribution
        # xi_b_relationship = type of relationship between xi and b, i.e., linear or quadratic

        # NOTE: each batch has the same cost vector c, constraint matrix A, and ground-truth linear model W_gt
        # NOTE: ensure that the above keys are 0-indexed

        num_batches = 3 # number of batches, i.e., groups of 4 settings with similar data generation parameters
        num_reps = 2 # TODO: 30 # number of replications to perform per setting
        N_train = 200 # number of training datapoints
        N_test = 50 # number of test datapoints
        feas_tol = 1e-6 # feasibility tolerance for certain optimization solves
        opt_tol = 1e-6 # optimality tolerance for certain optimization solves

        print(f'There are a total of {len(settings)} settings evenly-divided between {num_batches} batches.')
        print(f'Each of the {len(settings)} settings has the following: \n')
        print(f'   {num_reps} replications')
        print(f'   {N_train} training datapoints per replication')
        print(f'   {N_test} test datapoints per replication \n')
        print('~' * 100)

    if generate_synthetic_experiment_data:
        
        print('\n Generating data for synthetic experiment \n')

        # Generate each setting's data in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers = os.cpu_count() - 1) as executor:
            futures = [executor.submit(generate_synthetic_data_for_one_setting, settings, setting, num_batches, N_train, N_test, num_reps, feas_tol, opt_tol, base_dir) for setting in settings]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f'Task failed with exception {e}')
        print('Finished generating data for synthetic experiment \n')
        print('~' * 100)

    if run_sensitivity_analysis:

        print('\n Conducting sensitivity analysis for each setting \n')

        # Set hyperparameter candidates (NOTE: each of the below parameter lists should be in ascending order and it should be that gammas[0] = 0)
        lambda_regs = [1e-8, 1e-4, 1e0, 1e4, 1e8]
        gammas = [0, 1e-8, 1e-4, 1e0, 1e4, 1e8]
        alphas = np.arange(0.5, 5.5, 0.5)

        # Conduct each setting's sensitivity analysis in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers = os.cpu_count() - 1) as executor:
            futures = [executor.submit(run_sensitivity_analysis_for_one_setting, setting, N_train, lambda_regs, gammas, alphas, feas_tol, opt_tol, base_dir) for setting in settings]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f'Task failed with exception {e}')
        print('Finished all sensitivity analyses \n')
        print('~' * 100)

    if generate_sensitivity_analysis_plots:

        print('\n Generating sensitivity analysis plots')

        # Set hyperparameter candidates (NOTE: these lists must be the same as the parameter value lists in the above 'run_sensitivity_analysis' chunk of code or there will be an error)
        lambda_regs = [1e-8, 1e-4, 1e0, 1e4, 1e8]
        gammas = [0, 1e-8, 1e-4, 1e0, 1e4, 1e8]
        alphas = np.arange(0.5, 5.5, 0.5)

        # Analyze each setting's sensitivity analysis results in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers = os.cpu_count() - 1) as executor:
            futures = [executor.submit(generate_sensitivity_analysis_plots_for_one_setting, setting, N_train, N_test, lambda_regs, gammas, alphas, feas_tol, opt_tol, base_dir) for setting in settings]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f'Task failed with exception {e}')
        print('Finished generating all sensitivity analysis plots \n')
        print('~' * 100)

    if run_primal_dal_soln_method_comparison:

        print('\n Comparing solution methods in solving the Primal-DAL problem \n')

        # Fix hyperparameter values
        lambda_reg = 1e-4
        gamma = 1e-4

        # Run each setting's solution method comparison in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers = os.cpu_count() - 1) as executor:
            futures = [executor.submit(run_primal_dal_soln_method_comparison_for_one_setting, setting, N_train, lambda_reg, gamma, feas_tol, opt_tol, base_dir) for setting in settings]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f'Task failed with exception {e}')
        print('Finished solution method comparison for the Primal-DAL problem \n')
        print('~' * 100)

    if analyze_primal_dal_soln_method_comparison_results:

        print('\n Analyzing results from the solution method comparison \n')
        
        for setting in settings:

            print(f'setting # {setting}')

            # Grab median objective value/runtime
            primal_dal_runtime_path = os.path.join(base_dir, 'synthetic', f'setting_{setting}', 'primal_dal_soln_method_comparison', f'primal_dal_N_train_{N_train}_runtime.csv')
            primal_dal_runtime_df = pd.read_csv(primal_dal_runtime_path)
            primal_dal_obj_val_medians = primal_dal_runtime_df.groupby(['soln_method'])['primal_dal_obj_val'].median()
            total_time_medians = primal_dal_runtime_df.groupby(['soln_method'])['total_time'].median()
            print('Median objective function value = \n ', primal_dal_obj_val_medians)
            print('Median runtime = \n ', total_time_medians)
            print('\n')
        
        print('Finished analyzing all results \n')
        print('~' * 100)

    if run_synthetic_experiment:
        
        print('\n Conducting synthetic experiment \n')

        # Set hyperparameter candidates (NOTE: each of the below parameter lists should be in ascending order and it should be that gammas[0] = 0)
        lambda_regs = [1e4] # TODO: [1e-8, 1e-4, 1e0, 1e4, 1e8] or maybe something different???
        gammas = [0, 1e4] # TODO: [0, 1e-8, 1e-4, 1e0, 1e4, 1e8] or maybe something different???
        alphas = np.arange(0.5, 5.5, 0.5) # TODO: maybe something different???
        infeasibility_aversion_coefs = [0, 0.2, 0.4, 0.6, 0.8, 1] # TODO: maybe something different???
        deltas = [1, 3, 5, 7] # TODO: maybe something different??? # L1 hyperparameter in the Lasso problem

        # Conduct each setting's experiment in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers = os.cpu_count() - 1) as executor:
            futures = [executor.submit(run_synthetic_experiment_for_one_setting, setting, N_train, N_test, lambda_regs, gammas, alphas, infeasibility_aversion_coefs, deltas, feas_tol, opt_tol, base_dir) for setting in settings]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f'Task failed with exception {e}')
        print('Finished synthetic experiment \n')
        print('~' * 100)
    
    if analyze_synthetic_experiment_results:

        print('\n Analyzing results from the synthetic experiment')

        # Analyze each setting's synthetic experiment results in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers = os.cpu_count() - 1) as executor:
            futures = [executor.submit(analyze_synthetic_experiment_results_for_one_setting, setting, N_train, N_test, base_dir) for setting in settings]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f'Task failed with exception {e}')
        print('Finished analyzing synthetic experiment results \n')
        print('~' * 100)
        
    if run_beer_experiment:

        print('\n Conducting beer experiment \n')

        # Set hyperparameter candidates
        lambda_regs = [1e-8, 1e-4, 1e0, 1e4, 1e8]
        gammas = [0, 1e-8, 1e-4, 1e0, 1e4, 1e8]
        deltas = [1, 3, 5, 7]

        # Run the beer experiment
        run_all_replications_of_beer_experiment(lambda_regs, gammas, deltas, base_dir)
        print('Finished beer experiment \n')
        print('~' * 100)

    if analyze_beer_experiment_results:

        print('\n Analyzing results from the beer experiment')

        # Get sizes of the training and test datasets
        xi_train_temp = read_pickle_file(os.path.join(base_dir, 'beer', 'train', 'xi.pkl'))
        xi_test_temp = read_pickle_file(os.path.join(base_dir, 'beer', 'test', 'xi.pkl'))
        N_train, N_test = len(xi_train_temp[0]), len(xi_test_temp[0])
        
        # Analyze results of the beer experiment
        analyze_beer_experiment_results_for_all_reps(N_train, N_test, base_dir)
        print('Finished analyzing beer experiment results \n')
        print('~' * 100)
    
    return None

if __name__ == '__main__':
    main()