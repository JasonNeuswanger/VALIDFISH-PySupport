import math
import numpy as np
import importlib.util
import json
import sys
import os
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

fish_data = json_data['2015-07-17-3 Panguingue - Dolly Varden (id #4)']
# fish_data['prey_categories_characteristics']
# fish_data['diet_by_category']

# for item in sorted(list(json_data.keys())):
#     print(item)

objective_weights = {
    'foraging_attempt_rate': 1,
    'focal_velocity': 1,
    'proportion_ingested': 1,
    'spatial': 5,
    'diet_proportions_combined': 3
}

# CALCULATE OVERALL AVERAGE PREY TYPE ATTRIBUTES for filling in mass/energy in cases where it shows up as 0 in the fish
# drift data. Mostly this is for debris length purposes, because anytime there's prey in the drift, there are length
# and energy values for that particular site and the modeling is based on that. But we need some data here for debris
# length and therefore debris visibility/distraction in classes with no prey counted at the current site.
all_prey_lengths = {}
all_prey_energies = {}
for fish_key, fish_data in json_data.items():
    for prey_type_key, prey_type_data in fish_data['prey_categories_characteristics'].items():
        if prey_type_key not in all_prey_lengths.keys(): all_prey_lengths[prey_type_key] = []
        if prey_type_key not in all_prey_energies.keys(): all_prey_energies[prey_type_key] = []
        if prey_type_data['mean_prey_energy'] > 0:
            all_prey_energies[prey_type_key].append(prey_type_data['mean_prey_energy'])
        if prey_type_data['mean_prey_length'] > 0:
            all_prey_lengths[prey_type_key].append(prey_type_data['mean_prey_length'])
prey_type_mean_lengths = {prey_type_number : np.mean(np.array(all_prey_lengths[prey_type_number])) for prey_type_number in all_prey_lengths.keys()}
prey_type_mean_energies = {prey_type_number : np.mean(np.array(all_prey_energies[prey_type_number])) for prey_type_number in all_prey_lengths.keys()}


class FieldTestFish:

    def __init__(self, fish_label):
        data = json_data[fish_label]
        assert(data['bottom_z_m'] < 0 < data['surface_z_m'])
        self.fielddata = data
        self.color = tuple(data['color'])
        self.species = data['species']
        self.label = data['label']
        self.fork_length_cm = data['fork_length_cm']

        # In the original field data coords, +x is downstream, +y is cross-stream to the right, and +z is vertical
        # In the model coords, +x is cross-stream to the right, +y is upstream, and +z is vertical
        # The line below, we convert detection positions from field data coords into model coords
        self.field_detection_positions = [[y, -x, z] for (x, y, z) in data['detection_positions']]

        self.cforager = vf.Forager(
            data['fork_length_cm'],                 # fork length (cm)
            data['mass_g'],                         # mass (g)
            1.0,                                    # sigma_A
            1.2 * data['focal_velocity_m_per_s'],   # mean_column_velocity
            0.5,                                    # inspection_time
            2.0,                                    # discrimination_threshold
            -1,                                     # search image
            1e-4,                                   # delta_0
            10,                                     # alpha_tau
            10,                                     # alpha_d
            0.5,                                    # A_0
            0.5,                                    # beta
            data['bottom_z_m'],                     # bottom_z
            data['surface_z_m'],                    # surface_z
            int(data['temperature_C']),             # temperature (integer)
            0.5,                                    # tau_0
            50,                                     # flicker frequency
            1e-3,                                   # nu_0
            2.0,                                    # discriminability
            1e-2,                                   # delta_p
            1.0,                                    # omega_p
            0.5,                                    # ti_p
            1.0,                                    # sigma_p_0
            INTERPOLATION_ROOT                      # base directory for maneuver interpolation files
        )
        for pt in data['prey_categories_characteristics'].values():
            mean_prey_length = pt['mean_prey_length'] if pt['mean_prey_length'] > 0 else prey_type_mean_lengths[str(pt['number'])]
            mean_prey_energy = pt['mean_prey_energy'] if pt['mean_prey_energy'] > 0 else prey_type_mean_energies[str(pt['number'])]
            self.cforager.add_prey_type(
                pt['number'],
                pt['name'],
                mean_prey_length * 0.001,  # convert prey length from data units (mm) to model units (m)
                mean_prey_energy,
                pt['crypticity'],
                pt['prey_drift_concentration'],
                pt['debris_drift_concentration'],
                pt['search_image_eligible']
            )
        self.cforager.process_prey_type_changes()

        self.parameter_names = ['delta_0', 'alpha_tau', 'alpha_d', 'beta',
                                'A_0', 'flicker_frequency', 'tau_0', 'nu_0',
                                'discriminability', 'delta_p', 'omega_p', 'ti_p', 'sigma_p_0']

        self.parameter_bounds = {
            name: self.cforager.get_parameter_bounds(self.cforager.get_parameter_named(name))
            for name in self.parameter_names
        }

        self.scaled_parameter_bounds = {name: ((np.log10(bounds[0]), np.log10(bounds[1]))
                                              if self.cforager.is_parameter_log_scaled(self.cforager.get_parameter_named(name))
                                              else bounds)
                                        for name, bounds in self.parameter_bounds.items()}

    def foraging_point_distribution_distance(self, **kwargs):
        # First, we split the a cubic foraging region encompassing the max radius into many (gridsize^3)
        # cells and compute the relative concentration of prey pursuits at each position. Areas predicted
        # to have not just very few but zero pursuits (generally outside the search volume) are then
        # excluded from the analysis. All of these values are flattened into a list of predicted relative
        # pursuit densities throughout the foraging volume.
        # todo Try comparing these only against captures, or non-surface captures
        gridsize = 50j
        def func(x, y, z):
            return self.cforager.relative_pursuits_by_position(x, y, z)
        vfunc = np.vectorize(func)
        r = self.cforager.get_max_radius()
        x, y, z = np.mgrid[-r:r:gridsize, -r:r:gridsize, -r:r:gridsize]
        raw_predictions = vfunc(x, y, z).flatten()
        predictions = np.array(sorted(raw_predictions[raw_predictions > 0]))
        if len(predictions) == 0:
            print("WARNING! Predicted no pursuits at any spatial position with relative_pursuits_by_position. Something is wrong with the parameters. Returning maxium Wasserstein-like distance of 1.")
            return 1
        # Now we load the field observations of detection positions and calculate the relative concentration
        # of pursued items predicted to be detected at each of these points.
        observations = np.array([self.cforager.relative_pursuits_by_position(*coords) for coords in self.field_detection_positions])
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
        if kwargs.get('verbose', True):
            print("Foraging position wasserstein-like distance is {0:.4f}.".format(newmetric))
        if kwargs.get('plot', False):
            import matplotlib.pyplot as plt
            plt.clf()
            plot_x = np.linspace(min(predictions), max(predictions), num=200)
            plot_y_observations = observation_cdf(plot_x)
            plt.step(plot_x, plot_y_observations, label="Observed")
            plot_y_sums = sums_interp(plot_x)
            plt.step(plot_x, plot_y_sums, label="Predicted")
            plt.legend(loc=4)
            plt.title('Fish {0}'.format(self.label))
            plt.figtext(0.8, 0.3, "dist={0:.4f}".format(newmetric))
            plt.xlabel("Normalized 'relative pursuits by position'")
            plt.ylabel("Proportion of activity at that value or below")
            if 'figure_folder' in kwargs: plt.savefig(os.path.join(kwargs['figure_folder'], "Foraging Point Distribution Distance.pdf"))
            if kwargs.get('show', True): plt.show()
        return newmetric

    def evaluate_fit(self, verbose=True):
        """ Values based on proportions (most values) are contribute to the objective function as sums of squares of the
        differences between observed and predicted proportions. Values not based on proportions (focal velocity, foraging
        rate), contribute as the square of the proportional difference between observed and predicted values, which is the
        difference between them divided by the average of the two numbers. Each metric can be weighted."""
        def vprint(text):
            if verbose:
                print(text)
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
                vprint("For diet category '{0}', predicted proportion {1:.6f}, observed proportion {2:.6f}.".format(pt.get_name(), predicted, observed))
        if (np.isnan(diet_obj_total)):
            vprint("WARNING: Diet proportions were NaN, perhaps because fish didn't eat anything. Maximizing this part of the objective function.")
            diet_part = diet_obj_count * objective_weights['diet_proportions_combined']
        else:
            diet_part = np.sqrt(diet_obj_total / diet_obj_count) * objective_weights['diet_proportions_combined'] # RMS error, weighted
        # NREI -- not used in objective function, just for curiosity/printing.
        predicted_NREI = self.cforager.NREI()
        predicted_GREI = self.cforager.GREI()
        predicted_focal_swimming_cost = self.cforager.get_focal_swimming_cost()
        predicted_maneuver_cost_rate = self.cforager.maneuver_cost_rate()
        observed_NREI = self.fielddata['empirical_NREI_J_per_s']
        objective_value = fa_rate_part + velocity_part + proportion_ingested_part + spatial_part + diet_part
        vprint("NREI: predicted {0:.5f} J/s, observed estimate {1:.5f} J/s. (Prediced gross: {2:.5f}, focal cost: {3:.5f}, maneuver cost: {4:.5f})".format(predicted_NREI, observed_NREI, predicted_GREI, predicted_focal_swimming_cost, predicted_maneuver_cost_rate))
        # vprint("Objective function value is {0:.5f}. (Contributions: attempt rate {1:.3f}, velocity {2:.3f}, ingestion {3:.3f}, distance {4:.3f}, angle {5:.3f}, diet {6:.3f}).\n".format(
        #     objective_value, fa_rate_part, velocity_part, proportion_ingested_part, distance_part, angle_part, diet_part))
        vprint(
            "Objective function value is {0:.5f}. (Contributions: attempt rate {1:.4f}, velocity {2:.4f}, ingestion {3:.4f}, spatial {4:.4f}, diet {5:.4f}).\n".format(
                objective_value, fa_rate_part, velocity_part, proportion_ingested_part, spatial_part,
                diet_part))
        return objective_value

    def optimize(self, iterations, pack_size, verbose=True, use_chaos=False, use_dynamic_C=False, use_exponential_decay=False, use_levy=False, use_only_alpha=False, use_weighted_alpha=True):
        opt = vf.Optimizer(self.cforager, iterations, pack_size, verbose)
        return opt.optimize_forager()  # returns array of fitnesses by step

    def save_state(self):
        # These parameter need to be saved in the dict in the order they're called by set_strategies() and set_parameters()
        save_dict = {
            'parameters': [self.cforager.get_parameter(vf.Forager.Parameter.delta_0),
                           self.cforager.get_parameter(vf.Forager.Parameter.alpha_tau),
                           self.cforager.get_parameter(vf.Forager.Parameter.alpha_d),
                           self.cforager.get_parameter(vf.Forager.Parameter.beta),
                           self.cforager.get_parameter(vf.Forager.Parameter.A_0),
                           self.cforager.get_parameter(vf.Forager.Parameter.flicker_frequency),
                           self.cforager.get_parameter(vf.Forager.Parameter.tau_0),
                           self.cforager.get_parameter(vf.Forager.Parameter.nu_0),
                           self.cforager.get_parameter(vf.Forager.Parameter.discriminability),
                           self.cforager.get_parameter(vf.Forager.Parameter.delta_p),
                           self.cforager.get_parameter(vf.Forager.Parameter.omega_p),
                           self.cforager.get_parameter(vf.Forager.Parameter.ti_p),
                           self.cforager.get_parameter(vf.Forager.Parameter.sigma_p_0)
                           ],
            'strategies': [self.cforager.get_strategy(vf.Forager.Strategy.sigma_A),
                           self.cforager.get_strategy(vf.Forager.Strategy.mean_column_velocity),
                           self.cforager.get_strategy(vf.Forager.Strategy.inspection_time),
                           self.cforager.get_strategy(vf.Forager.Strategy.discrimination_threshold),
                           self.cforager.get_strategy(vf.Forager.Strategy.search_image)
                           ]
        }
        with open(TEMP_PICKLE_FOLDER + self.label + ' temp save.pickle', 'wb') as handle:
            pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_state(self):
        with open(TEMP_PICKLE_FOLDER + self.label + ' temp save.pickle', 'rb') as handle:
            retrieved_dict = pickle.load(handle)
        self.cforager.set_parameters(*retrieved_dict['parameters'])
        self.cforager.set_strategies(*retrieved_dict['strategies'])