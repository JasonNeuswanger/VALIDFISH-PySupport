#from mayavi import mlab
#import Fish3D

import math
import numpy as np
import importlib.util
import json
import sys
import field_test_fish as ftf

test_fish = ftf.FieldTestFish('2015-07-11-1 Chena - Chinook Salmon (id #4)')
# test_fish.evaluate_fit()
# test_fish.optimize(200, 7)
# test_fish.evaluate_fit()

# next step: do this optimize / evaluate for a bunch of parameter sets
# https://github.com/befelix/SafeOpt/blob/master/examples/2d_example.ipynb

import safeopt
import random
import GPy


# also probably need to optimize these three
# 1.0,  # lambda_c
# 0.5,  # sigma_t
# 0.03,  # base crypticity
# but for now, starting with just the main 5

bounds = [
    (0.00001, 2.0),  # delta_0  -- Scales effect of angular size on tau; bigger delta_0 = harder detection.
    (0.00001, 2.0),  # alpha_0  -- Scales effect of feature-based attention on tau; bigger alpha_0 = harder detection.
    (0.00001, 2.0),  # Z_0      -- Scales effect of search rate / spatial attention on tau; smaller Z_0 = harder detection.
    (0.00001, 2.0),  # c_1      -- Scales effect of FBA and saccade time on discrimination; bigger values incentivize longer saccades
    (0.00001, 2.0),  # beta     -- Scales effect of set size on tau; larger beta = harder detection, more incentive to drop debris-laden prey
]

parameter_sets = safeopt.linearly_spaced_combinations(bounds, 10)
n_initial_sets = 10
x = []
for i in range(n_initial_sets):
    x.append(random.choice(parameter_sets))

y = []
for params in x:
    delta_0, alpha_0, beta, Z_0, c_1 = params
    test_fish.cforager.modify_parameters(delta_0, alpha_0, beta, Z_0, c_1)
    test_fish.optimize(500, 7)
    y.append(test_fish.evaluate_fit(verbose=True))

x = np.array([np.array([ 0.22223111,  1.33333667,  0.88889444,  1.77777889,  2.        ]), np.array([  2.22231111e-01,   1.00000000e-05,   4.44452222e-01,
         6.66673333e-01,   1.55555778e+00]), np.array([  1.55555778e+00,   1.33333667e+00,   1.00000000e-05,
         1.11111556e+00,   6.66673333e-01]), np.array([ 1.77777889,  1.77777889,  2.        ,  0.88889444,  0.88889444]), np.array([ 1.55555778,  0.88889444,  0.88889444,  1.33333667,  1.77777889]), np.array([ 0.44445222,  0.22223111,  1.11111556,  1.33333667,  1.77777889]), np.array([  1.00000000e-05,   4.44452222e-01,   1.11111556e+00,
         6.66673333e-01,   1.00000000e-05]), np.array([  1.00000000e-05,   1.77777889e+00,   2.00000000e+00,
         8.88894444e-01,   1.11111556e+00]), np.array([ 0.44445222,  1.33333667,  0.88889444,  1.77777889,  0.22223111]), np.array([  2.22231111e-01,   1.33333667e+00,   1.00000000e-05,
         8.88894444e-01,   1.33333667e+00])]);
y = 100-np.array([[0.45505950721330446, 1.2415875290285674, 0.43263568881885056, 1.3353898097872654, 0.5928920054753741, 0.4228535837642009, 0.5776951450566175, 1.1010143390194187, 0.38342095277659494, 0.6617810466337234]]).T;

# it tries to maximize fitness, so need objective function to be subtracting fits... but can't subtract from 0
# because safeopt complains about negative y, so i subtract from 100

gp = GPy.models.GPRegression(x, y, noise_var=1e-4)
opt = safeopt.SafeOptSwarm(gp, 0., bounds=bounds, threshold=0.2)
for i in range(20):
    x_next = opt.optimize()
    delta_0, alpha_0, beta, Z_0, c_1 = x_next
    test_fish.cforager.modify_parameters(delta_0, alpha_0, beta, Z_0, c_1)
    test_fish.optimize(500, 7)
    y_meas = 100-test_fish.evaluate_fit(verbose=True)
    opt.add_new_data_point(x_next, y_meas)
    print("Added new point with objective fn = {0:.6f}, params delta_0={1:.5f}, alpha_0=={2:.5f}, beta={3:.5f}, Z_0={4:.5f}, c_1={5:.5f}.".format(y_meas, delta_0, alpha_0, beta, Z_0, c_1))
    best_params, best_objective = list(opt.get_maximum())
    delta_0, alpha_0, beta, Z_0, c_1 = best_params
    print("Best value with objective fn = {0:.6f} has params delta_0={1:.5f}, alpha_0=={2:.5f}, beta={3:.5f}, Z_0={4:.5f}, c_1={5:.5f}.".format(best_objective[0], delta_0, alpha_0, beta, Z_0, c_1))
test_fish.cforager.modify_parameters(delta_0, alpha_0, beta, Z_0, c_1)
test_fish.optimize(500, 7)
test_fish.cforager.print_strategy()

# start: Best value with objective fn = 99.616579 has params delta_0=0.44445, alpha_0==1.33334, beta=0.88889, Z_0=1.77778, c_1=0.22223.
# end: Best value with objective fn = 99.676775 has params delta_0=0.76730, alpha_0==0.62820, beta=0.41424, Z_0=1.57476, c_1=1.85856.
