from sys import platform, argv
#import pandas as pd
import numpy as np
import GPyOpt as gpo # note, need to pip install both this and sobol_seq
from GPyOpt.experiment_design import initial_design

IS_MAC = (platform == 'darwin')
if IS_MAC:
    import inspectable_fish
    acquisition_type, scaling, batch_name = 'EI', 'linear', 'mytest5'
    # fish_labels = ['2015-07-11-1 Chena - Chinook Salmon (id #4)',
    #                 '2015-08-05-1 Chena - Chinook Salmon (id #4)',
    #                 '2015-07-10-1 Chena - Chinook Salmon (id #4)',
    #                 '2015-08-05-2 Chena - Chinook Salmon (id #4)',
    #                 '2015-06-12-1 Chena - Chinook Salmon (id #1)'
    #                 ]
    # fish_labels = [#'2015-07-11-1 Chena - Chinook Salmon (id #4)',
    #                '2015-08-05-2 Chena - Chinook Salmon (id #4)',
    #                '2015-07-17-3 Panguingue - Dolly Varden (id #4)',
    #                #'2015-06-17-1 Panguingue - Dolly Varden (id #4)',
    #                '2016-06-10-2 Clearwater - Arctic Grayling (id #1)',
    #                #'2016-08-02-2 Clearwater - Arctic Grayling (id #1)'
    #                ]
    # fish_labels = ['2015-07-11-1 Chena - Chinook Salmon (id #4)']
    # fish_labels = ['2015-07-11-1 Chena - Chinook Salmon (id #4)',
    #                '2015-07-10-1 Chena - Chinook Salmon (id #4)',
    #                '2015-08-05-2 Chena - Chinook Salmon (id #4)',
    #                '2015-06-16-2 Panguingue - Dolly Varden (id #2)',
    #                '2016-06-17-1 Panguingue - Dolly Varden (id #3)',
    #                '2015-07-17-3 Panguingue - Dolly Varden (id #4)',
    #                '2016-06-10-2 Clearwater - Arctic Grayling (id #1)',
    #                '2015-06-23-2 Clearwater - Arctic Grayling (id #1)',
    #                '2016-08-02-2 Clearwater - Arctic Grayling (id #1)'
    #                ]
    #fish_labels = ['2016-08-02-2 Clearwater - Arctic Grayling (id #1)']
    fish_labels = ['2016-08-12-1 Chena - Chinook Salmon (id #1)']
    fishes = [inspectable_fish.InspectableFish(fish_label) for fish_label in fish_labels]
    RESULTS_FOLDER = '/Users/Jason/Dropbox/Drift Model Project/Calculations/cluster_pretest_results/'
    n_iterations = 50            # number of times new values are requested to calculate fitnesses
    n_evals_per_iteration = 1   # number of jobs per iteration above, for parallelizing across nodes eventually
    opt_cores = 7     # grey wolf algorithm pack size
    opt_iters = 100    # grey wolf algorithm iterations
else:
    import field_test_fish
    acquisition_type, scaling, batch_name = argv[1:]
    fish_labels = ['2015-07-11-1 Chena - Chinook Salmon (id #4)',
                    '2015-08-05-1 Chena - Chinook Salmon (id #4)',
                    '2015-07-10-1 Chena - Chinook Salmon (id #4)'
                    ]
    fishes = [field_test_fish.FieldTestFish(fish_label) for fish_label in fish_labels]
    RESULTS_FOLDER = '/home/alaskajn/results/bayes_opt_test/'
    n_iterations = 1000  # number of times new values are requested to calculate fitnesses
    n_evals_per_iteration = 1  # number of jobs per iteration above, for parallelizing across nodes eventually
    opt_cores = 26   # grey wolf algorithm pack size
    opt_iters = 100  # grey wolf algorithm iterations

def objective_function(*args):
    invalid_objective_function_value = 1000000  # used to replace inf, nan, or extreme values with something slightly less bad
    argnames = [item['name'] for item in domain]
    argvalues = args
    def scale(argname, argvalue):
        return 10**argvalue if argname in log_scaled_params else argvalue
    scaled_values = [scale(name, value) for name, value in zip(argnames, argvalues)]
    for key, value in fixed_parameters.items():
        argnames.append(key)
        scaled_values.append(value)
    d = dict(zip(argnames, scaled_values))
    ordered_params = [d[key] for key in [row['name'] for row in full_domain]]
    objective = 0
    for fish in fishes:
        fish.cforager.set_parameters(*ordered_params)
        print("Optimizing strategy for fish ", fish.label)
        fish.optimize(opt_iters, opt_cores, True, False, False, False, False, False, True)
        fit_value = fish.evaluate_fit(verbose=True)
        if fit_value > invalid_objective_function_value or not np.isfinite(fit_value):
            return np.nan
        else:
            objective += fit_value
    return objective

def optimize_forager_with_parameters(forager, *args):
    argnames = [item['name'] for item in domain]
    argvalues = args
    def scale(argname, argvalue):
        return 10 ** argvalue if argname in log_scaled_params else argvalue

    scaled_values = [scale(name, value) for name, value in zip(argnames, argvalues)]
    for key, value in fixed_parameters.items():
        argnames.append(key)
        scaled_values.append(value)
    d = dict(zip(argnames, scaled_values))
    ordered_params = [d[key] for key in [row['name'] for row in full_domain]]
    forager.cforager.set_parameters(*ordered_params)
    forager.optimize(opt_iters, opt_cores, True, False, False, False, False, False, True)

def X_as_string(X):
    pieces=[]
    for i, value in enumerate(X):
        name = domain[i]['name']
        printed_value = 10**value if name in log_scaled_params else value
        if value < 0.001 or value > 1000:
            pieces.append("{0}={1:.3e}".format(name, printed_value))
        else:
            pieces.append("{0}={1:.3f}".format(name, printed_value))
    return ', '.join(pieces)

log_scaled_params = ['delta_0', 'A_0', 'alpha_tau', 'alpha_d', 'beta', 't_s_0', 'tau_0', 'nu']
scaled_parameter_bounds = {key: ((np.log10(value[0]), np.log10(value[1])) if key in log_scaled_params else value) for key, value in fishes[0].parameter_bounds.items()}

# Online guidelines suggest the algorithm performs well with a number of function evaluations
# equal to about 10-20 times the dimensionality. That would be 800-2000 for my problem.

full_domain =[  # must contain all inputs and in order the're given to cforager.set_parameters()
        {'name': 'delta_0', 'type': 'continuous', 'domain': scaled_parameter_bounds['delta_0']},
        {'name': 'alpha_tau', 'type': 'continuous', 'domain': scaled_parameter_bounds['alpha_tau']},
        {'name': 'alpha_d', 'type': 'continuous', 'domain': scaled_parameter_bounds['alpha_d']},
        {'name': 'beta', 'type': 'continuous', 'domain': scaled_parameter_bounds['beta']},
        {'name': 'A_0', 'type': 'continuous', 'domain': scaled_parameter_bounds['A_0']},
        {'name': 't_s_0', 'type': 'continuous', 'domain': scaled_parameter_bounds['t_s_0']},
        {'name': 'discriminability', 'type': 'continuous', 'domain': scaled_parameter_bounds['discriminability']},
        {'name': 'flicker_frequency', 'type': 'continuous', 'domain': scaled_parameter_bounds['flicker_frequency']},
        {'name': 'tau_0', 'type': 'continuous', 'domain': scaled_parameter_bounds['tau_0']},
        {'name': 'nu', 'type': 'continuous', 'domain': scaled_parameter_bounds['nu']},
]
fixed_parameters = {    # Fix a parameter's value here to exclude it from optimization analysis, especially alpha_tau and alpha_d if not allowing search images
    'alpha_tau': 1,
    'alpha_d': 1
}
domain = [item for item in full_domain if item['name'] not in fixed_parameters.keys()]

space = gpo.Design_space(domain)

n_initial_points = 100
X_init = initial_design('sobol', space, n_initial_points)
Y_init = []
for i, x in enumerate(X_init[:1]):
    value = objective_function(*x)
    Y_init.append(value)
    print("For initial point {0} of {1}, appending objective function value {2:.3f} (best so far = {4:.3f}) for x={3}.\n".format(i+1, n_initial_points, value, X_as_string(x), min(Y_init)))
Y_init = np.array(Y_init).reshape(n_initial_points, 1)

X_all = X_init
Y_all = Y_init
X_best = X_all[Y_all.argmin()]
Y_best = Y_all[Y_all.argmin()]

# things to try:
# acquisition_jitter > 0
# normalize_y = True
# noise_var = correct value
# model type up 'GP_MCMC' and acquisition_type = 'EI_MCMC' or 'MPI_MCMC'
# try model type 'RF' for random forest
# try model_type 'warpedGP' or 'input_warped_GP'

acquisition_type = 'EI'  # should be EI (expected improvement) or MPI (maximum probability of improvement) or LCB
n_iterations = 200
n_evals_per_iteration = 1

for i in range(n_iterations):
    bo = gpo.methods.BayesianOptimization(f=None,
                                          domain=domain,
                                          X=X_all, Y=Y_all,
                                          model_type='GP',
                                          acquisition_type=acquisition_type,
                                          normalize_Y=False,
                                          evaluator_type='local_penalization',  # needs to be local_penalization to keep next-value suggestions from overlapping too much within batches
                                          batch_size=n_evals_per_iteration,
                                          num_cores=1,
                                          noise_var=0.15*len(fishes),
                                          acquisition_jitter=0)
    X_next = bo.suggest_next_locations()
    #X_next1 = best1 * np.random.uniform(low=0.97, high=1.03, size=len(best1))
    #X_next2 = best2 * np.random.uniform(low=0.97, high=1.03, size=len(best2))
    #X_next = np.vstack((X_next1, X_next2))
    # When I tweak this for the cluster, use pending_x arguments above to track the queue
    for j, x in enumerate(X_next):
        value = objective_function(*x)
        if np.isfinite(value):
            X_all = np.vstack((X_all, x))
            Y_all = np.vstack((Y_all, value))
            X_best = np.vstack((X_best, X_all[Y_all.argmin()]))
            Y_best = np.vstack((Y_best, Y_all[Y_all.argmin()]))
            np.savetxt(RESULTS_FOLDER + batch_name + "_X_all.csv", X_all, delimiter=",")
            np.savetxt(RESULTS_FOLDER + batch_name + "_Y_all.csv", Y_all, delimiter=",")
            np.savetxt(RESULTS_FOLDER + batch_name + "_X_best.csv", X_best, delimiter=",")
            np.savetxt(RESULTS_FOLDER + batch_name + "_Y_best.csv", Y_best, delimiter=",")
            print("\nFor iteration {0} of {1}, point {2} of {3}, appending objective function value {4:.3f} (vs best {5:.3f}) for x={6}.\n\n".format(i+1, n_iterations, j+1, len(X_next), value, Y_best[-1][0], X_as_string(x)))
        else:
            print("\nFor iteration {0} of {1}, point {2} of {3}, got NaN objective function value at x = {4}.\n\n".format(i+1, n_iterations, j+1, len(X_next), x))

print("COMPLETED. After {2} evaluations, best Y = {0:.4f} at X = {1}.".format(Y_best[-1][0], X_as_string(X_best[-1]), len(Y_best)))


# xy = np.hstack((X_all, Y_all))
# xyf = np.array([row for row in xy if np.isfinite(row[-1]) and row[-1] < 5000])
# X_all = xyf[:,:-1]
# Y_all = np.array(xyf[:,-1]).reshape(len(xyf), 1)


# For iteration 5 of 40, point 1 of 1, appending objective function value 9.376 (vs best 9.376) for x=delta_0=2.667e-06, beta=1.654, A_0=2.625e-03, t_s_0=3.373e-01, discriminability=3.297, flicker_frequency=76.911, tau_0=5.089, nu=4.205e-04.
# For iteration 2 of 30, point 1 of 3, appending objective function value 9.087 (vs best 9.087) for x=delta_0=1.737e-06, beta=2.339, A_0=2.589e-03, t_s_0=2.578e-01, discriminability=3.369, flicker_frequency=76.927, tau_0=10.000, nu=2.869e-04.


# -------- To find the noise variance:
# Y_for_same_X = []
# for i in range(10):
#     Y_for_same_X.append(objective_function(*X_best[-1]))
# print("Noise variance is {0:.6f}.".format(np.var(Y_for_same_X)))
# For 3 fish, we had [4.652192077439163, 2.3255931894201534, 3.3090159239949166, 3.1344221772592387, 3.1537014988485845, 3.130648305382892, 2.2491513112702504, 3.139061537148923, 2.8922810807656774, 3.200043343080637]
# The noise variance was 0.38.
# This would presumably be divided by 3 for just one fish.

X_unique = np.unique(X_best, axis=0)
for fish in fishes:
    optimize_forager_with_parameters(fish, *X_best[-1])
    fish.save_state()

test_fish = fishes[0]
optimize_forager_with_parameters(test_fish, *X_best[-1])
test_fish.save_state()
test_fish.plot_roc_curves()
test_fish.cforager.print_strategy()
test_fish.cforager.print_status()


# test_fish.cforager.get_strategy(vf.Forager.Strategy.sigma_A)
# test_fish.cforager.set_strategy(vf.Forager.Strategy.sigma_A, 1.15*np.pi)

for fish in fishes:
    fish.load_state()

dolly = inspectable_fish.InspectableFish('2015-07-17-3 Panguingue - Dolly Varden (id #4)')
grayling = inspectable_fish.InspectableFish('2016-08-02-2 Clearwater - Arctic Grayling (id #1)')


optimize_forager_with_parameters(fish, *X_unique[-1])

from mayavi import mlab
mlab.options.backend = 'simple'

test_fish = fishes[7]
fig3d = test_fish.plot_predicted_detection_field(gridsize=50j, colorMax=None, bgcolor=(1,1,1), pointcolor=(1,0,0))
test_fish.evaluate_fit()


test_fish.foraging_point_distribution_distance(verbose=False, plot=True)


test_fish.plot_angle_response_to_velocity()

# maybe the problem now is fish are going into faster water to boost loom?

#todo why does fish '2016-08-02-2 Clearwater - Arctic Grayling (id #1)' only ahve 2 prey classes?

test_x, test_z, test_pt = 0.03, 0.03, test_fish.cforager.get_prey_type('3 mm size class')
df = test_fish.plot_tau_components(test_x, test_z, test_pt)
test_fish.cforager.print_strategy()
test_fish.cforager.print_parameters()

import matplotlib.pyplot as plt
plt.figure()
df.plot()
plt.show()


# Old run for different 6 fish
# COMPLETED. After 101 evaluations, best Y = 1.5226 at X = delta_0=1.428e-05, beta=1.872e-01, A_0=2.394e-03, t_s_0=1.431, discriminability=5.632, flicker_frequency=10.006, tau_0=5.462, nu=2.412e-03.
# X = np.array([ -4.84531681,  -0.72769643,  -2.62089764,   0.15565454, 5.63170009,  10.0057271 ,   0.73733045,  -2.61764453])
# Test fish 1 looks great under these parameter sets... good one for figures.
# COMPLETED. After 401 evaluations, best Y = 1.2788 at X = delta_0=8.975e-04, beta=1.000e-01, A_0=1.710e-02, t_s_0=8.531e-01, discriminability=3.994, flicker_frequency=58.042, tau_0=5.582e-02, nu=1.000e-02.
# X = np.array([ -3.04694903,  -1.        ,  -1.76695891,  -0.06899052, 3.99396835,  58.04223186,  -1.25317802,  -2.        ])
# New run for new fish.

# COMPLETED. After 201 evaluations, best Y = 5.1211 at X = delta_0=5.757e-05, beta=4.814, A_0=3.492e-02, t_s_0=8.554e-01, discriminability=3.237, flicker_frequency=32.107, tau_0=2.495e-02, nu=2.529e-04.
# X = np.array([ -4.23980078,   0.6825426 ,  -1.45689422,  -0.06781831, 3.23741818,  32.10711545,  -1.60295352,  -3.59700022])

# New cost function
# COMPLETED. After 31 evaluations, best Y = 8.7423 at X = delta_0=2.371e-06, beta=8.660e-01, A_0=3.398e-01, t_s_0=3.377e-01, discriminability=2.406, flicker_frequency=33.031, tau_0=3.924e-01, nu=3.398e-03.



test_fish.evaluate_fit(verbose=True)

test_fish.load_state()
test_fish.plot_variable_reports()


gridsize=30j
def func(x, y, z):
    return test_fish.cforager.relative_pursuits_by_position(0.01 * x, 0.01 * y, 0.01 * z)
vfunc = np.vectorize(func)
r = 1.05 * 100 * test_fish.cforager.get_max_radius()
x, y, z = np.mgrid[-r:r:gridsize, -r:r:gridsize, -r:r:gridsize]
s = vfunc(x, y, z)
sf = s.flatten()
nzsf = sf[sf>0]

fieldvals = []
for xf, yf, zf in test_fish.fielddata['detection_positions']:
    fieldvals.append(test_fish.cforager.relative_pursuits_by_position(xf, yf, zf))


import numpy as np
import statsmodels.api as sm # recommended import according to the docs
import matplotlib.pyplot as plt
import scipy

scipy.stats.wasserstein_distance(nzsf, fieldvals)

ecdf = sm.distributions.ECDF(nzsf)
x = np.linspace(min(nzsf), max(nzsf))
y = ecdf(x)
plt.step(x, y)

fcdf = sm.distributions.ECDF(fieldvals)
yfp = fcdf(x)
plt.step(x, yfp)
plt.show()

# for goodness of fit...



# Strange relationships with detection probability, but working as intended:
# Mean column velocity: detection probability drops to 0 beyond a certain point because pursuit isn't profitable.
# Theta: detection probability starts at 0 for low theta, until the search volume includes (test_x, test_z).

# Make a graphical summary of fit-to-data (paired histograms for proportions)

# maybe do plots like this with detection probability as well

# After this, start trying to plot variable relationships for the fish based on deviation from the optimal scenario.



# todo look into this result:
# Optimizing strategy for fish  2016-08-12-1 Chena - Chinook Salmon (id #1)
# Focal velocity is predicted -0.072, observed 0.092 m/s.
# How can focal velocity EVER be negative in the model?????
# Is the fish below the bottom or something? Objective function was 200+ with parameters that worked well for others. Something wrong.

# Also, the surface is still messed up for vid_2016_06_02_2 resulting in data unusable for the model because
# distance-below-surface and distance-above-bottom is all messed up (therefore so is coord system for foraging attempts)
# It might have one or two badly-corresponded points throwing everything off -- try plotting them as actual points.


# import time
# tstart = time.clock()
# fishes[0].optimize(70, 7, True, False, False, False, False, False, True)
# tend = time.clock()
# print("Took ", tend - tstart)


X_all = np.loadtxt(RESULTS_FOLDER + batch_name + "_X_all.csv", delimiter=",")
Y_all = np.loadtxt(RESULTS_FOLDER + batch_name + "_Y_all.csv", delimiter=",").reshape(len(X_all), 1)
X_best = np.loadtxt(RESULTS_FOLDER + batch_name + "_X_best.csv", delimiter=",")
Y_best = np.loadtxt(RESULTS_FOLDER + batch_name + "_Y_best.csv", delimiter=",").reshape(len(X_best), 1)
