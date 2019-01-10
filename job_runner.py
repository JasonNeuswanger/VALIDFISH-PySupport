from platform import uname
import sys
import json
from time import sleep
import pickle
import pymysql
import numpy as np
import GPyOpt as gpo  # note, need to pip install both this and sobol_seq
from GPyOpt.experiment_design import initial_design

IS_MAC = (uname()[0] == 'Darwin')
if IS_MAC:
    import inspectable_fish
    FishConstructor = inspectable_fish.InspectableFish
else:
    import field_test_fish
    FishConstructor = field_test_fish.FieldTestFish

def quit(message):
    print(message)
    sys.exit(0)

class JobRunner:

    def __init__(self, initial_job_name, cores_per_node=14, readonly=False): # set readonly to True for analysis of job results
        self.machine_name = uname()[1]
        self.job_name = initial_job_name
        self.previous_job_properties = None
        self.readonly = readonly
        self.cores_per_node = cores_per_node
        with open('Fish_Groups.json', 'r') as fgf:
            self.fish_groups = json.load(fgf)
        with open('db_settings.pickle', 'rb') as handle:
            db_settings = pickle.load(handle)
            self.db = pymysql.connect(host=db_settings['host'], port=db_settings['port'], user=db_settings['user'],
                                 passwd=db_settings['passwd'], db=db_settings['db'], autocommit=True,
                                 cursorclass=pymysql.cursors.DictCursor)
            self.cursor = self.db.cursor()

        if not readonly:
            self.safe_query("INSERT INTO job_runners (current_job_name, machine_name, current_task, cores_per_node) VALUES (\"{0}\", \"{1}\", \"{2}\", {3})".format(initial_job_name, self.machine_name, "Awaiting Job Assignment", self.cores_per_node))
            self.safe_query("SELECT LAST_INSERT_ID()")
            self.runner_id = self.cursor.fetchone()['LAST_INSERT_ID()']
        else:
            self.runner_id = -1
        self.load_job_properties()

    def __del__(self):
        self.db.close()

    def job_property_changed(self, job_property=None):  # check on a specific property or pass None to check all
        if self.previous_job_properties is None:
            return True
        else:
            if job_property is None:
                return self.job_properties != self.previous_job_properties
            else:
                return self.job_properties[job_property] != self.previous_job_properties[job_property]

    def safe_query(self, query):
        retry = True  # just to start the loop, even though the first try isn't a retry
        while retry:
            try:
                self.cursor.execute(query)
                retry = False
            except Exception as e:
                print("Exception in MySQL during query:\n\n{0}\n\nException was:\n\n{1}\n\nSleeping 30 seconds and retrying.".format(query, repr(e)))
                retry = True
                sleep(30)

    def load_job_properties(self):
        if self.readonly:
            self.safe_query("SELECT * FROM job_descriptions WHERE job_name=\"{0}\"".format(self.job_name))
            self.job_properties = self.cursor.fetchone()
            self.fish_labels, self.fish_species = self.fish_groups[self.job_properties['fish_group']]
            self.fishes = [FishConstructor(fish_label) for fish_label in self.fish_labels]
            self.build_domain()
            self.opt_cores = self.cores_per_node - 2  # grey wolf algorithm pack size
            self.opt_iters = self.job_properties['grey_wolf_iterations']
        else:
            self.safe_query("SELECT * FROM job_runners WHERE id={0}".format(self.runner_id))
            self.runner_properties = self.cursor.fetchone()
            self.job_name = self.runner_properties['current_job_name']
            if self.runner_properties['stopped']:
                quit("Job runner {0} was manually stopped on machine {1}.".format(self.runner_id, self.machine_name))
            self.safe_query("SELECT * FROM job_descriptions WHERE job_name=\"{0}\"".format(self.job_name))
            self.job_properties = self.cursor.fetchone()
            if self.job_properties is None:
                quit("ERROR: No job description found with name {0}.".format(self.job_name))
            if self.job_properties['iterations_completed'] >= self.job_properties['max_iterations']:
                self.safe_query("UPDATE job_runners SET current_task=\"Completed (Max Iterations Reached)\", stopped=1 WHERE id={0}".format(self.runner_id))
                quit("quiting script : reached max number of iterations.")
            self.opt_cores = self.cores_per_node - 2  # grey wolf algorithm pack size
            self.opt_iters = self.job_properties['grey_wolf_iterations']
            if self.job_property_changed('fish_group'):
                if self.job_properties['fish_group'] not in self.fish_groups.keys():
                    quit("ERROR: Fish group {0} not defined.".format(self.job_properties['fish_group']))
                self.fish_labels, self.fish_species = self.fish_groups[self.job_properties['fish_group']]
                self.fishes = [FishConstructor(fish_label) for fish_label in self.fish_labels]
                self.build_domain()
            self.previous_job_properties = self.job_properties

    def build_domain(self):
        self.actual_parameter_bounds = self.fishes[0].parameter_bounds
        self.scaled_parameter_bounds = self.fishes[0].scaled_parameter_bounds
        self.all_parameters = self.fishes[0].parameter_names
        full_domain = [{'name': param, 'type': 'continuous', 'domain': self.scaled_parameter_bounds[param]} for param in self.all_parameters]
        self.fixed_parameters = {}
        for param in self.all_parameters:
            if self.job_properties[param] is not None:
                self.fixed_parameters[param] = self.job_properties[param]
        self.domain = [item for item in full_domain if item['name'] not in self.fixed_parameters.keys()]
        self.parameters_to_optimize = [item['name'] for item in self.domain]

    def objective_function(self, job_id, *args):
        invalid_objective_function_value = 1000000  # used to replace inf, nan, or extreme values with something slightly less bad
        if self.readonly: return invalid_objective_function_value
        argnames = [item['name'] for item in self.domain]
        if len(argnames) != len(args):
            print(argnames)
            print(args)
        assert(len(argnames) == len(args))  # make sure function was called with exact # of arguments to map onto current domain
        def scale(argname, argvalue):
            p = self.fishes[0].cforager.get_parameter_named(argname)
            scaled_value = self.fishes[0].cforager.reverse_transform_parameter_value(p, argvalue)
            print("Scaled argument {0} from {1} to {2}.".format(argname, argvalue, scaled_value))
            return scaled_value
        scaled_values = [scale(name, value) for name, value in zip(argnames, args)]
        for key, value in self.fixed_parameters.items():
            argnames.append(key)
            scaled_values.append(scale(key, value))
        d = dict(zip(argnames, scaled_values))
        ordered_params = [d[key] for key in self.all_parameters]
        objective = 0
        for i, fish in enumerate(self.fishes):
            fish.cforager.set_parameters(*ordered_params)
            print("Optimizing strategy for fish {0} of {1}: {2}.".format(i+1, len(self.fishes), fish.label))
            fish.optimize(self.opt_iters, self.opt_cores, True, False, False, False, False, False, True)
            fit_value = fish.evaluate_fit(verbose=True)
            if fit_value > invalid_objective_function_value or not np.isfinite(fit_value):
                return np.nan
            else:
                objective += fit_value
            self.safe_query("UPDATE job_results SET progress={1:.2f} WHERE id={0}".format(job_id, (i+1)/len(self.fishes)))
        return objective

    def optimize_forager_with_parameters(self, forager, *args):
        argnames = [item['name'] for item in self.domain]
        def scale(argname, argvalue):
            p = self.fishes[0].cforager.get_parameter_named(argname)
            return self.fishes[0].cforager.reverse_transform_parameter_value(p, argvalue)
        scaled_values = [scale(name, value) for name, value in zip(argnames, args)]
        for key, value in self.fixed_parameters.items():
            argnames.append(key)
            scaled_values.append(value)
        d = dict(zip(argnames, scaled_values))
        ordered_params = [d[key] for key in self.all_parameters]
        forager.cforager.set_parameters(*ordered_params)
        forager.optimize(self.opt_iters, self.opt_cores, True, False, False, False, False, False, True)

    def X_as_string(self, X):
        pieces=[]
        for i, value in enumerate(X):
            name = self.domain[i]['name']
            p = self.fishes[0].cforager.get_parameter_named(name)
            printed_value = self.fishes[0].cforager.reverse_transform_parameter_value(p, value)
            if value < 0.001 or value > 1000:
                pieces.append("{0}={1:.3e}".format(name, printed_value))
            else:
                pieces.append("{0}={1:.3f}".format(name, printed_value))
        return ', '.join(pieces)

    def value_from_X(self, X, param_name):
        names = [item['name'] for item in self.domain]
        if param_name in names:
            return X[names.index(param_name)]
        else:
            return self.fixed_parameters[param_name]

    def insert_query_from_X(self, X, is_initial):
        query = "INSERT INTO job_results (job_name, is_initial, runner_id, start_time, completed_time, objective_value,\
                   delta_0, alpha_tau, alpha_d, beta, A_0, flicker_frequency, tau_0, nu_0, discriminability, delta_p, omega_p, ti_p, sigma_p_0) VALUES \
                   (\"{job_name}\", {is_initial}, {runner_id}, NULL, NULL, NULL, \
                   {delta_0}, {alpha_tau}, {alpha_d}, {beta}, {A_0}, {flicker_frequency}, {tau_0}, {nu_0}, {discriminability}, {delta_p}, {omega_p}, {ti_p}, {sigma_p_0})" \
            .format(job_name=self.job_name,
                    is_initial=is_initial,
                    runner_id=self.runner_id,
                    delta_0=self.value_from_X(X, 'delta_0'),
                    alpha_tau=self.value_from_X(X, 'alpha_tau'),
                    alpha_d=self.value_from_X(X, 'alpha_d'),
                    beta=self.value_from_X(X, 'beta'),
                    A_0=self.value_from_X(X, 'A_0'),
                    flicker_frequency=self.value_from_X(X, 'flicker_frequency'),
                    tau_0=self.value_from_X(X, 'tau_0'),
                    nu_0=self.value_from_X(X, 'nu_0'),
                    discriminability=self.value_from_X(X, 'discriminability'),
                    delta_p=self.value_from_X(X, 'delta_p'),
                    omega_p=self.value_from_X(X, 'omega_p'),
                    ti_p=self.value_from_X(X, 'ti_p'),
                    sigma_p_0=self.value_from_X(X, 'sigma_p_0')
                    )
        print(query)
        return query

    def create_initial_jobs(self):
        print("\nCreating initial jobs.\n")
        self.safe_query("UPDATE job_runners SET current_task=\"Creating Initial Jobs\" WHERE id={0}".format(self.runner_id))
        space = gpo.Design_space(self.domain)
        X_init = initial_design('sobol', space, self.job_properties['initial_iterations'])
        for X in X_init:
            self.safe_query(self.insert_query_from_X(X, True))

    def create_iterated_jobs(self):
        print("\nCreating {0} new jobs.\n".format(self.job_properties['batch_size']))
        self.safe_query("UPDATE job_runners SET current_task=\"Creating New Jobs\" WHERE id={0}".format(self.runner_id))
        self.safe_query("SELECT * FROM job_results WHERE objective_value IS NOT NULL AND job_name=\"{0}\"".format(self.job_name))
        completed_job_data = self.cursor.fetchall()
        if len(completed_job_data) >= self.job_properties['max_iterations']:
            print("There are {0} completed iterations out of a maximum {1} requested -- ceasing new job creation.".format(len(completed_job_data), self.job_properties['max_iterations']))
            return
        Y_all = []
        X_all = []
        for row in completed_job_data:
            Y_all.append(row['objective_value'])
            X_all.append([row[key] for key in self.parameters_to_optimize])
        bo = gpo.methods.BayesianOptimization(f=None,
                                              domain=self.domain,
                                              X=np.array(X_all), Y=np.array(Y_all).reshape(len(Y_all), 1),
                                              model_type=self.job_properties['model_type'],
                                              acquisition_type=self.job_properties['acquisition_type'],
                                              normalize_Y=False,
                                              evaluator_type='local_penalization', # needs to be local_penalization to keep next-value suggestions from overlapping too much within batches
                                              batch_size=self.job_properties['batch_size'],
                                              num_cores=1,
                                              noise_var=0.15 * len(self.fishes),
                                              acquisition_jitter=self.job_properties['acquisition_jitter'])
        X_next = bo.suggest_next_locations()
        for X in X_next:
            self.safe_query(self.insert_query_from_X(X, False))

    def run_jobs(self):
        while True:
            self.load_job_properties()
            self.safe_query("SELECT id FROM job_results WHERE job_name=\"{0}\" LIMIT 1".format(self.job_name))
            if self.cursor.fetchone() is None:
                self.create_initial_jobs()
            job_data = None
            while job_data is None:
                self.safe_query("SELECT * FROM job_results WHERE start_time IS NULL AND job_name=\"{0}\" LIMIT 1".format(self.job_name))
                job_data = self.cursor.fetchone()
                if job_data is None:
                    self.safe_query("SELECT * FROM job_runners WHERE current_task=\"Creating New Jobs\"")
                    if self.cursor.fetchone() is None:
                        self.create_iterated_jobs()
                    else:
                        sleep(15)
            self.safe_query("UPDATE job_results SET start_time=NOW() WHERE id={0}".format(job_data['id']))
            self.safe_query("UPDATE job_runners SET current_task=\"Running Job {1}\" WHERE id={0}".format(self.runner_id, job_data['id']))
            param_values = [job_data[param] for param in self.parameters_to_optimize]
            obj_value = self.objective_function(job_data['id'], *param_values)
            fquery = "UPDATE job_results SET completed_time=NOW(), objective_value={1}, progress=1.0 WHERE id={0}".format(job_data['id'], obj_value)
            print(fquery)
            self.safe_query(fquery)
            self.safe_query("UPDATE job_descriptions SET iterations_completed=(SELECT COUNT(*) FROM job_results WHERE job_name=\"{0}\" AND objective_value IS NOT NULL) WHERE job_name=\"{0}\"".format(self.job_name))
