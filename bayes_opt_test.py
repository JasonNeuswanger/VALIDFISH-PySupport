import pandas as pd
import inspectable_fish
from bayes_opt import BayesianOptimization

test_fish = inspectable_fish.InspectableFish('2015-07-11-1 Chena - Chinook Salmon (id #4)')
#print(inspectable_fish.InspectableFish('2015-06-10-1 Chena - Chinook Salmon (id #1)').cforager.get_angular_resolution())
#print(inspectable_fish.InspectableFish('2015-08-13-1 Clearwater - Arctic Grayling (id #3)').cforager.get_angular_resolution())

# Linear space search
# def f(delta_0, alpha_0, beta, Z_0, c_1, discriminability, sigma_t, tau_0):
#     test_fish.cforager.modify_parameters(delta_0, alpha_0, beta, Z_0, c_1, discriminability, sigma_t, tau_0)
#     test_fish.optimize(500, 15, True, False, False, False, False, False, True)
#     return -test_fish.evaluate_fit(verbose=True)
#
# param_limits = {
#     'delta_0': (0.00001, 2.0),          # delta_0           -- Scales effect of angular size on tau; bigger delta_0 = harder detection.
#     'alpha_0': (0.00001, 2.0),          # alpha_0           -- Scales effect of feature-based attention on tau; bigger alpha_0 = harder detection.
#     'beta': (0.00001, 2.0),             # Z_0               -- Scales effect of search rate / spatial attention on tau; smaller Z_0 = harder detection.
#     'Z_0': (0.00001, 2.0),              # c_1               -- Scales effect of FBA and saccade time on discrimination; bigger values incentivize longer saccades
#     'c_1': (0.00001, 2.0),              # beta              -- Scales effect of set size on tau; larger beta = harder detection, more incentive to drop debris-laden prey
#     'discriminability': (0.01, 2.0),   # discriminability  -- Difference in mean preyishness between prey and debris.
#     'sigma_t': (0.01, 2.0),            # sigma_t           -- Variation in actual preyishness of both prey and debris (all types combined, for now) before perceptual effects are applied
#     'tau_0': (1e-5, 1.0)               # tau_0             -- Base aptitude of the fish, i.e mean time-until-detection with no other effects present
# }

# Log space search
def f(delta_0, alpha_0, beta, Z_0, c_1, discriminability, sigma_t, tau_0, t_V):
    test_fish.cforager.modify_parameters(10**delta_0, 10**alpha_0, 10**beta, 10**Z_0, 10**c_1, discriminability, sigma_t, 10**tau_0, 10**t_V)
    test_fish.optimize(500, 15, True, False, False, False, False, False, True)
    return -test_fish.evaluate_fit(verbose=True)

param_limits = {
    'delta_0': (-5, 1),          # delta_0           -- Scales effect of angular size on tau; bigger delta_0 = harder detection.
    'alpha_0': (-5, 1),          # alpha_0           -- Scales effect of feature-based attention on tau; bigger alpha_0 = harder detection.
    'beta': (-5, 1),             # Z_0               -- Scales effect of search rate / spatial attention on tau; smaller Z_0 = harder detection.
    'Z_0': (-5, 1),              # c_1               -- Scales effect of FBA and saccade time on discrimination; bigger values incentivize longer saccades
    'c_1': (-5, 1),              # beta              -- Scales effect of set size on tau; larger beta = harder detection, more incentive to drop debris-laden prey
    'discriminability': (0.02, 2.0),   # discriminability  -- Difference in mean preyishness between prey and debris.
    'sigma_t': (0.02, 2.0),            # sigma_t           -- Variation in actual preyishness of both prey and debris (all types combined, for now) before perceptual effects are applied
    'tau_0': (-5, 1),                    # tau_0             -- Base aptitude of the fish, i.e mean time-until-detection with no other effects present
    't_V':(-2,1)
}

bo = BayesianOptimization(f, param_limits)

maxes = []
def record_max_entry():
    entry = {'max_val': bo.res['max']['max_val']}
    for key, value in bo.res['max']['max_params'].items():
        entry[key] = value
    entry['neval'] = bo.space._length
    maxes.append(entry)

maxes_df = pd.DataFrame(maxes)
maxes_df.to_csv("filename")


bo.maximize(init_points=10, n_iter=0, acq='ei')
print(bo.res['max'])
record_max_entry()

for i in range(250):
    bo.maximize(init_points=0, n_iter=1, acq='ucb', kappa=5)
    print(bo.res['max'])
    record_max_entry()
    bo.maximize(init_points=0, n_iter=1, acq='ei')
    print(bo.res['max'])
    record_max_entry()




test550_maxes = { # from a log-scaled test
    'max_val': -0.34411109082806396,
    'max_params': {'beta': 1.0, 'c_1': -1.0902349739464894, 'Z_0': -5.0, 'delta_0': -1.2078338475676513, 'sigma_t': 2.0, 'alpha_0': -5.0, 'discriminability': 0.02, 'tau_0': -5.0, 't_V': 0.0}}
mp = test550_maxes['max_params']


# Need to start looking at how it's exploring the parameter space.
# mp = bo.res['max']['max_params']
#test_fish.cforager.modify_parameters(mp['delta_0'], mp['alpha_0'], mp['beta'], mp['Z_0'], mp['c_1'], mp['discriminability'], mp['sigma_t'], mp['tau_0'], mp['t_V']) # for linear search space
test_fish.cforager.modify_parameters(10**mp['delta_0'], 10**mp['alpha_0'], 10**mp['beta'], 10**mp['Z_0'], 10**mp['c_1'], mp['discriminability'], mp['sigma_t'], 10**mp['tau_0'], 10**mp['t_V']) # for log search space
test_fish.optimize(500, 15, True, False, False, False, False, False, True)
test_fish.evaluate_fit(verbose=True)
test_fish.plot_predicted_detection_field()  # radius and theta are way too large, not a good match to the real shape at all

# somehow this gives an objective function value of 2.03, not 0.34




# think about more forces to restrict search volume
# try exploring the parameters in log space... not a lot of small ones being tried

# For my next big overnight batch, I should run bayes_opt_test out to lots of iterations, several times for each fish, for several fish,
#
# Save pickled results so I can do 3-D plots on my Mac with them.
# Also make sure to track/plot convergence rates of both parameter estimates and fitness value.

