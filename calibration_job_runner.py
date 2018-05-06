from platform import uname
from sys import argv
import json
import pickle
import pymysql
import numpy as np
import GPyOpt as gpo  # note, need to pip install both this and sobol_seq

MAX_ITERATIONS = 5000

IS_MAC = (uname()[0] == 'Darwin')
NODE_NAME = uname()[1]

with open('Fish_Groups.json', 'r') as fgf:
    fish_groups = json.load(fgf)
with open('db_settings.pickle', 'rb') as handle:
    db_settings = pickle.load(handle)

if IS_MAC:
    import inspectable_fish
    job_name, fish_group = 'SecondFiveOfEach', 'calibration_five_of_each'
    fish_labels, fish_species = fish_groups[fish_group]
    fishes = [inspectable_fish.InspectableFish(fish_label) for fish_label in fish_labels]
    opt_cores = 12     # grey wolf algorithm pack size
    opt_iters = 200    # grey wolf algorithm iterations
else:
    import field_test_fish
    job_name, fish_group = argv[1:]
    fish_labels, fish_species = fish_groups[fish_group]
    fishes = [field_test_fish.FieldTestFish(fish_label) for fish_label in fish_labels]
    opt_cores = 12   # grey wolf algorithm pack size
    opt_iters = 200  # grey wolf algorithm iterations

search_images_allowed = (fish_species == 'all' or fish_species == 'grayling')
if search_images_allowed:  # DISABLING THESE FOR NOW
    fixed_parameters = {    # fix search image parameters to have no effect if not doing grayling
        'alpha_tau': 1,
        'alpha_d': 1,
        'flicker_frequency': 50
    }
else:
    fixed_parameters = {    # fix search image parameters to have no effect if not doing grayling
        'alpha_tau': 1,
        'alpha_d': 1,
        'flicker_frequency': 50
    }
log_scaled_params = ['delta_0', 'A_0', 'alpha_tau', 'alpha_d', 'beta', 't_s_0', 'tau_0', 'nu']

def objective_function(*args):
    invalid_objective_function_value = 1000000  # used to replace inf, nan, or extreme values with something slightly less bad
    job_id = args[0]
    argnames = [item['name'] for item in domain]
    argvalues = args[1:]
    if len(argnames) != len(argvalues):
        print(argnames)
        print(argvalues)
    assert(len(argnames) == len(argvalues))  # make sure function was called with exact # of arguments to map onto current domain
    def scale(argname, argvalue):
        return 10**argvalue if argname in log_scaled_params else argvalue
    scaled_values = [scale(name, value) for name, value in zip(argnames, argvalues)]
    for key, value in fixed_parameters.items():
        argnames.append(key)
        scaled_values.append(value)
    d = dict(zip(argnames, scaled_values))
    ordered_params = [d[key] for key in [row['name'] for row in full_domain]]
    objective = 0
    for i, fish in enumerate(fishes):
        fish.cforager.set_parameters(*ordered_params)
        print("Optimizing strategy for fish {0} of {1}: {2}.".format(i+1, len(fishes), fish.label))
        fish.optimize(opt_iters, opt_cores, True, False, False, False, False, False, True)
        fit_value = fish.evaluate_fit(verbose=True)
        if fit_value > invalid_objective_function_value or not np.isfinite(fit_value):
            return np.nan
        else:
            objective += fit_value
        cursor.execute("UPDATE jobs SET progress={1:.2f} WHERE id={0}".format(job_id, (i+1)/len(fishes)))
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

def create_new_jobs():
    select_query = "SELECT * FROM jobs WHERE objective_function IS NOT NULL AND job_name='{0}'".format(job_name)
    cursor.execute(select_query)
    completed_job_data = cursor.fetchall()
    if len(completed_job_data) >= MAX_ITERATIONS:
        print("There are {0} completed iterations out of a maximum {1} requested -- ceasing new job creation.".format(len(completed_job_data), MAX_ITERATIONS))
        return
    Y_all = []
    X_all = []
    for row in completed_job_data:
        job_id, is_initial, read_job_name, read_fish_group, machine_assigned, start_time, progress, completed_time, objective_val_read, delta_0, alpha_tau, alpha_d, beta, A_0, t_s_0, discriminability, flicker_frequency, tau_0, nu = row
        Y_all.append(objective_val_read)
        if search_images_allowed:
            X_all.append([delta_0, alpha_tau, alpha_d, beta, A_0, t_s_0, discriminability, flicker_frequency, tau_0, nu])
        else:
            X_all.append([delta_0, beta, A_0, t_s_0, discriminability, flicker_frequency, tau_0, nu])
    bo = gpo.methods.BayesianOptimization(f=None,
                                          domain=domain,
                                          X=np.array(X_all), Y=np.array(Y_all).reshape(len(Y_all), 1),
                                          model_type='GP',
                                          acquisition_type='EI',
                                          normalize_Y=False,
                                          evaluator_type='local_penalization',  # needs to be local_penalization to keep next-value suggestions from overlapping too much within batches
                                          batch_size=20,
                                          num_cores=1,
                                          noise_var=0.15*len(fishes),
                                          acquisition_jitter=0)
    X_next = bo.suggest_next_locations()
    def value_from_X(X, param_name):
        names = [item['name'] for item in domain]
        if param_name in names:
            return X[names.index(param_name)]
        else:
            return fixed_parameters[param_name]
    queries = []
    for X in X_next:
        query = "INSERT INTO jobs (job_name, is_initial, fish_group, machine_assigned, start_time, completed_time, objective_function,\
                   delta_0, alpha_tau, alpha_d, beta, A_0, t_s_0, discriminability, flicker_frequency, tau_0, nu) VALUES \
                   (\"{job_name}\", FALSE, \"{fish_group}\", NULL, NULL, NULL, NULL, \
                   {delta_0}, {alpha_tau}, {alpha_d}, {beta}, {A_0}, {t_s_0}, {discriminability}, {flicker_frequency}, {tau_0}, {nu})" \
            .format(job_name=job_name,
                    fish_group=fish_group,
                    delta_0=value_from_X(X, 'delta_0'),
                    alpha_tau=value_from_X(X, 'alpha_tau'),
                    alpha_d=value_from_X(X, 'alpha_d'),
                    beta=value_from_X(X, 'beta'),
                    A_0=value_from_X(X, 'A_0'),
                    t_s_0=value_from_X(X, 't_s_0'),
                    discriminability=value_from_X(X, 'discriminability'),
                    flicker_frequency=value_from_X(X, 'flicker_frequency'),
                    tau_0=value_from_X(X, 'tau_0'),
                    nu=value_from_X(X, 'nu'))
        queries.append(query)
    for query in queries:
        cursor.execute(query)


actual_parameter_bounds = fishes[0].parameter_bounds.items()
scaled_parameter_bounds = {key: ((np.log10(value[0]), np.log10(value[1])) if key in log_scaled_params else value) for key, value in actual_parameter_bounds}
full_domain = [  # must contain all inputs and in order the're given to cforager.set_parameters()
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
domain = [item for item in full_domain if item['name'] not in fixed_parameters.keys()]

db = pymysql.connect(host=db_settings['host'], port=db_settings['port'], user=db_settings['user'], passwd=db_settings['passwd'], db=db_settings['db'], autocommit=True)
cursor = db.cursor()
select_query = "SELECT * FROM jobs WHERE start_time IS NULL AND job_name='{0}' LIMIT 1".format(job_name)
cursor.execute(select_query)
job_data = cursor.fetchone()
if job_data is None:
    create_new_jobs()
    cursor.execute(select_query)
    job_data = cursor.fetchone()
while job_data is not None:
    job_id, is_initial, read_job_name, read_fish_group, machine_assigned, start_time, progress, completed_time, objective_val_read, delta_0, alpha_tau, alpha_d, beta, A_0, t_s_0, discriminability, flicker_frequency, tau_0, nu = job_data
    cursor.execute("UPDATE jobs SET start_time=NOW(), machine_assigned='{1}' WHERE id={0}".format(job_id, NODE_NAME))
    #if search_images_allowed:
    #    obj_value = objective_function(job_id, delta_0, alpha_tau, alpha_d, beta, A_0, t_s_0, discriminability, flicker_frequency, tau_0, nu)
    #else:
    obj_value = objective_function(job_id, delta_0, beta, A_0, t_s_0, discriminability, tau_0, nu)
    cursor.execute("UPDATE jobs SET completed_time=NOW(), objective_function={1}, progress=1.0 WHERE id={0}".format(job_id, obj_value))
    cursor.execute(select_query)
    job_data = cursor.fetchone()
    if job_data is None:
        create_new_jobs()
        cursor.execute(select_query)
        job_data = cursor.fetchone()
db.close()