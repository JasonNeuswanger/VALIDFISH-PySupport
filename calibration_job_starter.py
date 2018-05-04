import numpy as np
import pymysql
import pickle
import field_test_fish
import GPyOpt as gpo # note, need to pip install both this and sobol_seq
from GPyOpt.experiment_design import initial_design

# This script is meant to be run on my local machine just to kick-start a calibration job by specifying the initial
# points to evaluate and storing them in the tracking database. Then calibration_job_runner.py loops through empty
# entries in that database, adding new lines as needed.

# SETTINGS THAT MIGHT VARY BY JOB

JOB_NAME = "FirstFiveGrayling"
FISH_GROUP = "calibration_five_grayling"
n_initial_points = 100
fixed_parameters = {  # Fix a parameter's value here to exclude it from optimization analysis, especially alpha_tau and alpha_d if not allowing search images
    'alpha_tau': 1,
    'alpha_d': 1
}
log_scaled_params = ['delta_0', 'A_0', 'alpha_tau', 'alpha_d', 'beta', 't_s_0', 'tau_0', 'nu']

# CODE THAT SHOULD REMAIN THE SAME FOR ALL JOBS

actual_parameter_bounds = field_test_fish.FieldTestFish('2015-06-10-1 Chena - Chinook Salmon (id #1)').parameter_bounds.items()
scaled_parameter_bounds = {key: ((np.log10(value[0]), np.log10(value[1])) if key in log_scaled_params else value) for key, value in actual_parameter_bounds}
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
domain = [item for item in full_domain if item['name'] not in fixed_parameters.keys()]
space = gpo.Design_space(domain)
X_init = initial_design('sobol', space, n_initial_points)

def value_from_X(X, param_name):
    names = [item['name'] for item in domain]
    if param_name in names:
        return X[names.index(param_name)]
    else:
        return fixed_parameters[param_name]

queries = []
for X in X_init:
    query = "INSERT INTO jobs (job_name, is_initial, fish_group, machine_assigned, start_time, completed_time, objective_function,\
               delta_0, alpha_tau, alpha_d, beta, A_0, t_s_0, discriminability, flicker_frequency, tau_0, nu) VALUES \
               (\"{job_name}\", TRUE, \"{fish_group}\", NULL, NULL, NULL, NULL, \
               {delta_0}, {alpha_tau}, {alpha_d}, {beta}, {A_0}, {t_s_0}, {discriminability}, {flicker_frequency}, {tau_0}, {nu})"\
        .format(job_name            = JOB_NAME,
                fish_group          = FISH_GROUP,
                delta_0             = value_from_X(X, 'delta_0'),
                alpha_tau           = value_from_X(X, 'alpha_tau'),
                alpha_d             = value_from_X(X, 'alpha_d'),
                beta                = value_from_X(X, 'beta'),
                A_0                 = value_from_X(X, 'A_0'),
                t_s_0               = value_from_X(X, 't_s_0'),
                discriminability    = value_from_X(X, 'discriminability'),
                flicker_frequency   = value_from_X(X, 'flicker_frequency'),
                tau_0               = value_from_X(X, 'tau_0'),
                nu                  = value_from_X(X, 'nu'))
    queries.append(query)

with open('db_settings.pickle', 'rb') as handle:
    db_settings = pickle.load(handle)

db = pymysql.connect(host=db_settings['host'], port=db_settings['port'], user=db_settings['user'], passwd=db_settings['passwd'], db=db_settings['db'], autocommit=True)
cursor = db.cursor()
for query in queries:
    cursor.execute(query)
db.close()