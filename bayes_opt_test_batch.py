import sys
import field_test_fish
import pandas as pd
from bayes_opt import BayesianOptimization  # NEED TO INSTALL THIS

method, scaling, batch_name, fish_label = sys.argv
n_iterations = 500  # will really be 2x this + initial 30

test_fish = field_test_fish.FieldTestFish(fish_label)

if scaling == 'linear':
    def f(delta_0, alpha_0, beta, Z_0, c_1, discriminability, sigma_t, tau_0):
        test_fish.cforager.modify_parameters(delta_0, alpha_0, beta, Z_0, c_1, discriminability, sigma_t, tau_0)
        test_fish.optimize(500, 15, True, False, False, False, False, False, True)
        return -test_fish.evaluate_fit(verbose=True)

    param_limits = {
        'delta_0': (0.00001, 2.0),          # delta_0           -- Scales effect of angular size on tau; bigger delta_0 = harder detection.
        'alpha_0': (0.00001, 2.0),          # alpha_0           -- Scales effect of feature-based attention on tau; bigger alpha_0 = harder detection.
        'beta': (0.00001, 2.0),             # Z_0               -- Scales effect of search rate / spatial attention on tau; smaller Z_0 = harder detection.
        'Z_0': (0.00001, 2.0),              # c_1               -- Scales effect of FBA and saccade time on discrimination; bigger values incentivize longer saccades
        'c_1': (0.00001, 2.0),              # beta              -- Scales effect of set size on tau; larger beta = harder detection, more incentive to drop debris-laden prey
        'discriminability': (0.01, 2.0),   # discriminability  -- Difference in mean preyishness between prey and debris.
        'sigma_t': (0.01, 2.0),            # sigma_t           -- Variation in actual preyishness of both prey and debris (all types combined, for now) before perceptual effects are applied
        'tau_0': (1e-5, 1.0)               # tau_0             -- Base aptitude of the fish, i.e mean time-until-detection with no other effects present
    }
elif scaling == 'log':
    def f(delta_0, alpha_0, beta, Z_0, c_1, discriminability, sigma_t, tau_0):
        test_fish.cforager.modify_parameters(10**delta_0, 10**alpha_0, 10**beta, 10**Z_0, 10**c_1, discriminability, sigma_t, 10**tau_0)
        test_fish.optimize(500, 26, True, False, False, False, False, False, True)
        return -test_fish.evaluate_fit(verbose=False)

    param_limits = {
        'delta_0': (-5, 1),          # delta_0           -- Scales effect of angular size on tau; bigger delta_0 = harder detection.
        'alpha_0': (-5, 1),          # alpha_0           -- Scales effect of feature-based attention on tau; bigger alpha_0 = harder detection.
        'beta': (-5, 1),             # Z_0               -- Scales effect of search rate / spatial attention on tau; smaller Z_0 = harder detection.
        'Z_0': (-5, 1),              # c_1               -- Scales effect of FBA and saccade time on discrimination; bigger values incentivize longer saccades
        'c_1': (-5, 1),              # beta              -- Scales effect of set size on tau; larger beta = harder detection, more incentive to drop debris-laden prey
        'discriminability': (0.02, 2.0),   # discriminability  -- Difference in mean preyishness between prey and debris.
        'sigma_t': (0.02, 2.0),            # sigma_t           -- Variation in actual preyishness of both prey and debris (all types combined, for now) before perceptual effects are applied
        'tau_0': (-5, 1)               # tau_0             -- Base aptitude of the fish, i.e mean time-until-detection with no other effects present
    }

bo = BayesianOptimization(f, param_limits)

if method == 'ei':
    method1 = 'ei'
    method2 = 'ei'
elif method == 'ucb':
    method1 = 'ucb'
    method2 = 'ucb'
elif method == 'mixed':
    method1 = 'ei'
    method2 = 'ucb'

maxes = []


def record_max_entry():
    entry = {'max_val': bo.res['max']['max_val']}
    for key, value in bo.res['max']['max_params'].items():
        entry[key] = value
    entry['neval'] = bo.space._length
    maxes.append(entry)


bo.maximize(init_points=30, n_iter=0, acq=method1, kappa=5)
record_max_entry()

for i in range(n_iterations):
    bo.maximize(init_points=0, n_iter=1, acq=method1, kappa=5)
    record_max_entry()
    bo.maximize(init_points=0, n_iter=1, acq=method2, kappa=5)
    record_max_entry()


maxes_df = pd.DataFrame(maxes)
maxes_df.to_csv('/home/alaskajn/results/bayes_opt_test/' + batch_name + '.csv')