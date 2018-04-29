import math
import numpy as np
import importlib.util
import json
import sys
import pickle
import statsmodels.api as sm
import scipy


IS_MAC = (sys.platform == 'darwin')

if IS_MAC:
    PYVALIDFISH_FILE = "/Users/Jason/Dropbox/Drift Model Project/Calculations/VALIDFISH/VALIDFISH/cmake-build-release/pyvalidfish.cpython-35m-darwin.so"
    #PYVALIDFISH_FILE = "/Users/Jason/Dropbox/Drift Model Project/Calculations/VALIDFISH/VALIDFISH/cmake-build-debug/pyvalidfish.cpython-36m-darwin.so"
    INTERPOLATION_ROOT = "/Users/Jason/Dropbox/Drift Model Project/Calculations/driftmodeldev/maneuver-model-tables/"
    FIELD_DATA_FILE = "/Users/Jason/Dropbox/Drift Model Project/Calculations/VALIDFISH/PySupport/Model_Testing_Data.json"
    TEMP_PICKLE_FOLDER = "/Users/Jason/.temp_fish_cache/"
else:
    PYVALIDFISH_FILE = "/home/alaskajn/VALIDFISH/VALIDFISH/build/pyvalidfish.cpython-35m-x86_64-linux-gnu.so"
    INTERPOLATION_ROOT = "/home/alaskajn/maneuver-model-tables/"
    FIELD_DATA_FILE = "/home/alaskajn/VALIDFISH/PySupport/Model_Testing_Data.json"
    TEMP_PICKLE_FOLDER = "~/.temp_fish_cache/"

spec = importlib.util.spec_from_file_location("pyvalidfish",  PYVALIDFISH_FILE)
vf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vf)
with open(FIELD_DATA_FILE, 'r') as fdf:
    json_data = json.load(fdf)

# fish_data = json_data['2015-07-17-3 Panguingue - Dolly Varden (id #4)']
# fish_data['prey_categories_characteristics']
# fish_data['diet_by_category']

# for item in sorted(list(json_data.keys())):
#     print(item)

objective_weights = {
    'foraging_attempt_rate': 1,
    'focal_velocity': 1,
    'proportion_ingested': 1,
    # 'detection_distances_combined': 5,
    # 'detection_angles_combined': 5,
    'spatial': 5,
    'diet_proportions_combined': 2
}

class FieldTestFish:

    def __init__(self, fish_label):
        data = json_data[fish_label]
        assert(data['bottom_z_m'] < 0 < data['surface_z_m'])
        self.fielddata = data
        self.color = tuple(data['color'])
        self.species = data['species']
        self.label = data['label']
        self.fork_length_cm = data['fork_length_cm']
        self.cforager = vf.Forager(
            data['fork_length_cm'],                 # fork length (cm)
            data['mass_g'],                         # mass (g)
            0.02,                                   # delta_min (min angular size attended)
            1.0,                                    # sigma_A
            1.2 * data['focal_velocity_m_per_s'],   # mean_column_velocity
            0.3,                                    # saccade_time
            0.8,                                    # discrimination_threshold
            -1,                                     # search image
            1e-4,                                   # delta_0
            10,                                     # alpha_tau
            10,                                     # alpha_d
            0.05,                                   # A_0
            0.1,                                    # t_s_0
            0.3,                                    # beta
            data['bottom_z_m'],                     # bottom_z
            data['surface_z_m'],                    # surface_z
            int(data['temperature_C']),             # temperature (integer)
            0.05,                                   # bed_roughness
            1.0,                                    # discriminability
            0.1,                                    # tau_0
            30,                                     # flicker frequency
            1e-3,                                   # nu
            INTERPOLATION_ROOT                      # base directory for maneuver interpolation files
        )
        for pt in data['prey_categories_characteristics'].values():
            self.cforager.add_prey_type(
                pt['number'],
                pt['name'],
                pt['mean_prey_length'] * 0.001,  # convert prey length from data units (mm) to model units (m)
                pt['mean_prey_energy'],
                pt['crypticity'],
                pt['prey_drift_concentration'],
                pt['debris_drift_concentration'],
                pt['search_image_eligible']
            )
        self.cforager.process_prey_type_changes()

        self.parameter_bounds = {
            'delta_0': self.cforager.get_parameter_bounds(vf.Forager.Parameter.delta_0),
            'alpha_tau': self.cforager.get_parameter_bounds(vf.Forager.Parameter.alpha_tau),
            'alpha_d': self.cforager.get_parameter_bounds(vf.Forager.Parameter.alpha_d),
            'beta': self.cforager.get_parameter_bounds(vf.Forager.Parameter.beta),
            'A_0': self.cforager.get_parameter_bounds(vf.Forager.Parameter.A_0),
            't_s_0': self.cforager.get_parameter_bounds(vf.Forager.Parameter.t_s_0),
            'discriminability': self.cforager.get_parameter_bounds(vf.Forager.Parameter.discriminability),
            'flicker_frequency': self.cforager.get_parameter_bounds(vf.Forager.Parameter.flicker_frequency),
            'tau_0': self.cforager.get_parameter_bounds(vf.Forager.Parameter.tau_0),
            'nu': self.cforager.get_parameter_bounds(vf.Forager.Parameter.nu)
        }

    def foraging_point_distribution_distance(self, verbose=True, plot=False):
        # First, we split the a cubic foraging region encompassing the max radius into many (gridsize^3)
        # cells and compute the relative concentration of prey pursuits at each position. Areas predicted
        # to have not just very few but zero pursuits (generally outside the search volume) are then
        # excluded from the analysis. All of these values are flattened into a list of predicted relative
        # pursuit densities throughout the foraging volume.
        gridsize = 50j
        def func(x, y, z):
            return self.cforager.relative_pursuits_by_position(x, y, z)
        vfunc = np.vectorize(func)
        r = self.cforager.get_max_radius()
        x, y, z = np.mgrid[-r:r:gridsize, -r:r:gridsize, -r:r:gridsize]
        raw_predictions = vfunc(x, y, z).flatten()
        predictions = np.array(sorted(raw_predictions[raw_predictions > 0]))
        # Now we load the field observations of detection positions and calculate the relative concentration
        # of pursued items predicted to be detected at each of these points.
        observations = np.array([self.cforager.relative_pursuits_by_position(*coords) for coords in
                                 self.fielddata['detection_positions']])

        overall_max = max(predictions.max(), observations.max())
        predictions /= overall_max
        observations /= overall_max
        observation_cdf = sm.distributions.ECDF(observations)
        # Now calculate the Wasserstein Distance, the area between the empirical CDFs of the predictions
        # and observations, as our measure of how far apart the distributions were.
        # wd = scipy.stats.wasserstein_distance(predictions, observations)
        sums = [0]
        i = 0
        for i in range(len(predictions)):
            sums.append(sums[i] + predictions[i])
        sums = np.array(sums[1:])
        sums /= sums[-1]
        sums_interp = scipy.interpolate.interp1d(predictions, sums)
        def integrand(x):
            return abs(observation_cdf(x) - sums_interp(x))
        newmetric = scipy.integrate.quad(integrand, min(predictions), max(predictions))[0]
        if verbose:
            print("Foraging position wasserstein-like distance is {0:.4f}.".format(newmetric))
        if plot:
            import matplotlib.pyplot as plt
            plt.clf()
            # prediction_cdf = sm.distributions.ECDF(predictions)
            plot_x = np.linspace(min(predictions), max(predictions), num=200)
            # plot_y_predictions = prediction_cdf(plot_x)
            plot_y_observations = observation_cdf(plot_x)
            # plt.step(plot_x, plot_y_predictions, label="Predictions ECDF")
            plt.step(plot_x, plot_y_observations, label="Observed")
            # plt.step(predictions, sums, label="Sums")
            plot_y_sums = sums_interp(plot_x)
            plt.step(plot_x, plot_y_sums, label="Predicted")
            plt.legend(loc=4)
            plt.title('Fish {0}'.format(self.label))
            plt.figtext(0.8, 0.3, "dist={0:.4f}".format(newmetric))
            plt.xlabel("Normalized 'relative pursuits by position'")
            plt.ylabel("Proportion of activity at that value or below")
            plt.show()
        return newmetric

    def evaluate_fit(self, verbose=True):
        """ Values based on proportions (most values) are contribute to the objective function as sums of squares of the
        differences between observed and predicted proportions. Values not based on proportions (focal velocity, foraging
        rate), contribute as the square of the proportional difference between observed and predicted values, which is the
        difference between them divided by the average of the two numbers. Each metric can be weighted."""
        def vprint(text):
            if verbose:
                print(text)
        vprint("Evaluating fit to field data for one solution.")
        objective_value = 0  # objective function value to be minimized... built up in pieces through this function
        self.cforager.analyze_results()
        # Individually important fields
        predicted_fa_rate = self.cforager.get_foraging_attempt_rate()
        observed_fa_rate = self.fielddata['foraging_attempt_rate']
        fa_rate_part = (((observed_fa_rate - predicted_fa_rate) / (0.5 * (observed_fa_rate + predicted_fa_rate))) ** 2) * objective_weights['foraging_attempt_rate']
        vprint("Foraging attempt rate is predicted {0:.3f}, observed {1:.3f} attempts/s.".format(predicted_fa_rate, observed_fa_rate))
        predicted_focal_velocity = self.cforager.get_focal_velocity()
        observed_focal_velocity = self.fielddata['focal_velocity_m_per_s']
        velocity_part = (((observed_focal_velocity - predicted_focal_velocity) / (
                    0.5 * (observed_focal_velocity + predicted_focal_velocity))) ** 2) * objective_weights['focal_velocity']
        vprint("Focal velocity is predicted {0:.3f}, observed {1:.3f} m/s.".format(predicted_focal_velocity, observed_focal_velocity))
        predicted_proportion_ingested = self.cforager.get_proportion_of_attempts_ingested()
        observed_proportion_ingested = self.fielddata['proportion_of_attempts_ingested']
        proportion_ingested_part = (((observed_proportion_ingested - predicted_proportion_ingested) / (
                    0.5 * (observed_proportion_ingested + predicted_proportion_ingested))) ** 2) * objective_weights['proportion_ingested']
        vprint("Proportion of attempts ingested is predicted {0:.3f}, observed {1:.3f}.".format(predicted_proportion_ingested, observed_proportion_ingested))
        # Foraging point data
        spatial_part = self.foraging_point_distribution_distance(verbose=verbose, plot=False) * objective_weights['spatial']
        # spatial_detection_proportions = self.cforager.spatial_detection_proportions(None, "All", verbose) # use None for 'All' prey types
        # Detection distance data
        # distance_obj_total = 0
        # for i, bin in enumerate(spatial_detection_proportions['distance']):
        #     binmin, binmax = bin['min_distance'], bin['max_distance']
        #     predicted = bin['proportion']
        #     observed = self.fielddata['detection_distance_proportions'][i]
        #     distance_obj_total += (predicted - observed) ** 2
        #     vprint("For distance bin {0:.3f} to {1:.3f} m,    predicted proportion {2:.3f}, observed proportion {3:.3f}.".format(binmin, binmax, predicted, observed))
        # distance_part = (distance_obj_total / len(spatial_detection_proportions['distance'])) * objective_weights['detection_distances_combined']
        # # Detection angle data
        # angle_obj_total = 0
        # for i, bin in enumerate(spatial_detection_proportions['angle']):
        #     binmin, binmax = bin['min_angle'], bin['max_angle']
        #     predicted = bin['proportion']
        #     observed = self.fielddata['detection_angle_proportions'][i]
        #     angle_obj_total += (predicted - observed) ** 2
        #     vprint("For angle bin {0:.3f} to {1:.3f} radians, predicted proportion {2:.3f}, observed proportion {3:.3f}.".format(binmin, binmax, predicted, observed))
        # angle_part = (angle_obj_total / len(spatial_detection_proportions['angle'])) * objective_weights['detection_angles_combined']
        # Diet data
        dietdata = [item for item in self.fielddata['diet_by_category'].values() if item['number'] is not None]
        diet_obj_total = 0
        diet_obj_count = 0
        for dd in sorted(dietdata, key=lambda x: x['number']):
            pt = self.cforager.get_prey_type(dd['name'])
            observed = dd['diet_proportion']
            predicted = self.cforager.get_diet_proportion_for_prey_type(pt)
            if predicted > 0 or observed > 0:
                diet_obj_count += 1
                diet_obj_total += (predicted - observed) ** 2
                vprint("For diet category '{0}', predicted proportion {1:.3f}, observed proportion {2:.3f}.".format(pt.get_name(), predicted, observed))
        diet_part = (diet_obj_total / diet_obj_count) * objective_weights['diet_proportions_combined']
        # NREI -- not used in objective function, just for curiosity/printing.
        predicted_NREI = self.cforager.NREI()
        observed_NREI = self.fielddata['empirical_NREI_J_per_s']
        #objective_value = fa_rate_part + velocity_part + proportion_ingested_part + distance_part + angle_part + diet_part
        objective_value = fa_rate_part + velocity_part + proportion_ingested_part + spatial_part + diet_part
        vprint("NREI: predicted {0:.5f} J/s, observed estimate {1:.5f} J/s.".format(predicted_NREI, observed_NREI))
        # vprint("Objective function value is {0:.5f}. (Contributions: attempt rate {1:.3f}, velocity {2:.3f}, ingestion {3:.3f}, distance {4:.3f}, angle {5:.3f}, diet {6:.3f}).\n".format(
        #     objective_value, fa_rate_part, velocity_part, proportion_ingested_part, distance_part, angle_part, diet_part))
        vprint(
            "Objective function value is {0:.5f}. (Contributions: attempt rate {1:.4f}, velocity {2:.4f}, ingestion {3:.4f}, spatial {4:.4f}, diet {5:.4f}).\n".format(
                objective_value, fa_rate_part, velocity_part, proportion_ingested_part, spatial_part,
                diet_part))
        return objective_value

    def optimize(self, iterations, pack_size, verbose=True, use_chaos=False, use_dynamic_C=False, use_exponential_decay=False, use_levy=False, use_only_alpha=False, use_weighted_alpha=True):
        opt = vf.Optimizer(self.cforager, iterations, pack_size, verbose)
        opt.set_algorithm_options(use_chaos, use_dynamic_C, use_exponential_decay, use_levy, use_only_alpha, use_weighted_alpha)
        return opt.optimize_forager()  # returns array of fitnesses by step

    def save_state(self):
        delta_0 = self.cforager.get_parameter(vf.Forager.Parameter.delta_0)
        alpha_tau = self.cforager.get_parameter(vf.Forager.Parameter.alpha_tau)
        alpha_d = self.cforager.get_parameter(vf.Forager.Parameter.alpha_d)
        beta = self.cforager.get_parameter(vf.Forager.Parameter.beta)
        A_0 = self.cforager.get_parameter(vf.Forager.Parameter.A_0)
        t_s_0 = self.cforager.get_parameter(vf.Forager.Parameter.t_s_0)
        discriminability = self.cforager.get_parameter(vf.Forager.Parameter.discriminability)
        flicker_frequency = self.cforager.get_parameter(vf.Forager.Parameter.flicker_frequency)
        tau_0 = self.cforager.get_parameter(vf.Forager.Parameter.tau_0)
        nu = self.cforager.get_parameter(vf.Forager.Parameter.nu)

        delta_min = self.cforager.get_strategy(vf.Forager.Strategy.delta_min)
        sigma_A = self.cforager.get_strategy(vf.Forager.Strategy.sigma_A)
        mean_column_velocity = self.cforager.get_strategy(vf.Forager.Strategy.mean_column_velocity)
        saccade_time = self.cforager.get_strategy(vf.Forager.Strategy.saccade_time)
        discrimination_threshold = self.cforager.get_strategy(vf.Forager.Strategy.discrimination_threshold)
        search_image = self.cforager.get_strategy(vf.Forager.Strategy.search_image)

        save_dict = {
            'parameters': [delta_0, alpha_tau, alpha_d, beta, A_0, t_s_0, discriminability, flicker_frequency, tau_0, nu],
            'strategies': [delta_min, sigma_A, mean_column_velocity, saccade_time, discrimination_threshold, search_image]
        }
        with open(TEMP_PICKLE_FOLDER + self.label + ' temp save.pickle', 'wb') as handle:
            pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_state(self):
        with open(TEMP_PICKLE_FOLDER + self.label + ' temp save.pickle', 'rb') as handle:
            retrieved_dict = pickle.load(handle)
        self.cforager.set_parameters(*retrieved_dict['parameters'])
        self.cforager.set_strategies(*retrieved_dict['strategies'])