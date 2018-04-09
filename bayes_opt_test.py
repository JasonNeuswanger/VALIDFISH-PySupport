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
def f(delta_0, alpha_0, beta, Z_0, c_1, discriminability, sigma_t, tau_0):
    test_fish.cforager.modify_parameters(10**delta_0, 10**alpha_0, 10**beta, 10**Z_0, 10**c_1, discriminability, sigma_t, 10**tau_0)
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
    'tau_0': (-5, 1)               # tau_0             -- Base aptitude of the fish, i.e mean time-until-detection with no other effects present
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


# Linear-scaled parameter search, after 50 points:
# {'max_val': -0.85112326150202988, 'max_params': {'delta_0': 0.62911196349096932, 'sigma_t': 1.6551423790291766, 'beta': 0.12517227772042963, 'alpha_0': 1.9280868139013942, 'Z_0': 1.0173111250046725, 'tau_0': 0.030370979269513477, 'discriminability': 0.1326085814899563, 'c_1': 1.1621502702060809}}
# Foraging attempt rate is predicted 0.225, observed 0.233 attempts/s.
# Focal velocity is predicted 0.159, observed 0.161 m/s.
# Proportion of attempts ingested is predicted 0.037, observed 0.069.
# For distance bin 0.000 to 0.024 m, predicted proportion 0.000, observed proportion 0.020.
# For distance bin 0.024 to 0.048 m, predicted proportion 0.002, observed proportion 0.184.
# For distance bin 0.048 to 0.096 m, predicted proportion 0.027, observed proportion 0.408.
# For distance bin 0.096 to 0.192 m, predicted proportion 0.270, observed proportion 0.327.
# For distance bin 0.192 to 0.384 m, predicted proportion 0.702, observed proportion 0.061.
# For angle bin 0.000 to 0.785 radians, predicted proportion 0.060, observed proportion 0.388.
# For angle bin 0.785 to 1.571 radians, predicted proportion 0.179, observed proportion 0.531.
# For angle bin 1.571 to 2.356 radians, predicted proportion 0.234, observed proportion 0.061.
# For angle bin 2.356 to 3.142 radians, predicted proportion 0.203, observed proportion 0.020.
# For angle bin 3.142 to 3.927 radians, predicted proportion 0.159, observed proportion 0.000.
# For angle bin 3.927 to 4.712 radians, predicted proportion 0.113, observed proportion 0.000.
# For diet category '1 mm size class', predicted proportion 0.000, observed proportion 0.091.
# For diet category '2 mm size class', predicted proportion 0.002, observed proportion 0.209.
# For diet category '3 mm size class', predicted proportion 0.000, observed proportion 0.427.
# For diet category '4-5 mm size class', predicted proportion 0.957, observed proportion 0.218.
# For diet category '6-8 mm size class', predicted proportion 0.073, observed proportion 0.027.
# NREI: predicted 0.01630 J/s, observed estimate 0.05306 J/s.
# Objective function value is 0.85441. (Attempt rate 0.001, velocity 0.000, ingestion 0.001, distance 0.474, angle 0.221, diet 0.156)


# Solution from the log scale worth picking apart... it shows it's at least possible to get the distance bins right and
# angle bins not too far off. However, the parameter values tend to be pretty extreme in this case.
# Evaluating fit to field data for one solution.
# Foraging attempt rate is predicted 0.352, observed 0.233 attempts/s.
# Focal velocity is predicted 0.037, observed 0.161 m/s.
# Proportion of attempts ingested is predicted 0.007, observed 0.069.
# For distance bin 0.000 to 0.024 m, predicted proportion 0.017, observed proportion 0.020.
# For distance bin 0.024 to 0.048 m, predicted proportion 0.147, observed proportion 0.184.
# For distance bin 0.048 to 0.096 m, predicted proportion 0.445, observed proportion 0.408.
# For distance bin 0.096 to 0.192 m, predicted proportion 0.312, observed proportion 0.327.
# For distance bin 0.192 to 0.384 m, predicted proportion 0.042, observed proportion 0.061.
# For angle bin 0.000 to 0.785 radians, predicted proportion 0.175, observed proportion 0.388.
# For angle bin 0.785 to 1.571 radians, predicted proportion 0.492, observed proportion 0.531.
# For angle bin 1.571 to 2.356 radians, predicted proportion 0.333, observed proportion 0.061.
# For angle bin 2.356 to 3.142 radians, predicted proportion 0.000, observed proportion 0.020.
# For angle bin 3.142 to 3.927 radians, predicted proportion 0.000, observed proportion 0.000.
# For angle bin 3.927 to 4.712 radians, predicted proportion 0.000, observed proportion 0.000.
# For diet category '1 mm size class', predicted proportion 0.226, observed proportion 0.091.
# For diet category '2 mm size class', predicted proportion 0.509, observed proportion 0.209.
# For diet category '3 mm size class', predicted proportion 0.260, observed proportion 0.427.
# For diet category '4-5 mm size class', predicted proportion 0.195, observed proportion 0.218.
# For diet category '6-8 mm size class', predicted proportion 0.024, observed proportion 0.027.
# NREI: predicted 0.00102 J/s, observed estimate 0.05306 J/s.
# Objective function value is 1.83655. (Contributions: attempt rate 0.164, velocity 1.559, ingestion 0.004, distance 0.003, angle 0.080, diet 0.027)
#    45 | 01m32s |   -1.83655 |   -5.0000 |   -5.0000 |   -5.0000 |    1.0000 |    1.0000 |             0.0200 |    2.0000 |   -5.0000 |




# Having trouble matching number of smaller prey in the diet, and distance/angle proportions

# Big question is why the previous one didn't change even a little bit in the outer digits.

# Need to start looking at how it's exploring the parameter space.
mp = bo.res['max']['max_params']
#test_fish.cforager.modify_parameters(mp['delta_0'], mp['alpha_0'], mp['beta'], mp['Z_0'], mp['c_1'], mp['discriminability'], mp['sigma_t'], mp['tau_0']) # for linear search space
test_fish.cforager.modify_parameters(10**mp['delta_0'], 10**mp['alpha_0'], 10**mp['beta'], 10**mp['Z_0'], 10**mp['c_1'], mp['discriminability'], mp['sigma_t'], 10**mp['tau_0']) # for log search space
test_fish.optimize(500, 15, True, False, False, False, False, False, True)
test_fish.evaluate_fit(verbose=True)
test_fish.plot_predicted_detection_field()  # radius and theta are way too large, not a good match to the real shape at all



# think about more forces to restrict search volume
# try exploring the parameters in log space... not a lot of small ones being tried

# For my next big overnight batch, I should run bayes_opt_test out to lots of iterations, several times for each fish, for several fish,
#
# Save pickled results so I can do 3-D plots on my Mac with them.
# Also make sure to track/plot convergence rates of both parameter estimates and fitness value.

