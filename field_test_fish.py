import math
import numpy as np
import importlib.util
import json
import sys

IS_MAC = (sys.platform == 'darwin')

if IS_MAC:
    PYVALIDFISH_FILE = "/Users/Jason/Dropbox/Drift Model Project/Calculations/VALIDFISH/VALIDFISH/cmake-build-release/pyvalidfish.cpython-35m-darwin.so"
    INTERPOLATION_ROOT = "/Users/Jason/Dropbox/Drift Model Project/Calculations/driftmodeldev/maneuver-model-tables/"
    FIELD_DATA_FILE = "/Users/Jason/Dropbox/Drift Model Project/Data/Model_Testing_Data.json"
else:
    PYVALIDFISH_FILE = "/home/alaskajn/VALIDFISH/VALIDFISH/build/pyvalidfish.cpython-35m-x86_64-linux-gnu.so"
    INTERPOLATION_ROOT = "/home/alaskajn/maneuver-model-tables/"
    FIELD_DATA_FILE = "/home/alaskajn/VALIDFISH/PySupport/Model_Testing_Data.json"

spec = importlib.util.spec_from_file_location("pyvalidfish",  PYVALIDFISH_FILE)
vf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vf)
json_data = json.load(open(FIELD_DATA_FILE, 'r'))

objective_weights = {
    'foraging_attempt_rate': 1,
    'focal_velocity': 1,
    'proportion_ingested': 1,
    'detection_distances_combined': 4,
    'detection_angles_combined': 4,
    'diet_proportions_combined': 1
}

class FieldTestFish:

    def __init__(self, fish_label):
        data = json_data[fish_label]
        initial_radius = data['fork_length_cm'] * 0.01 * 4.  # 3 fork lengths, converted to m
        self.fielddata = data
        self.color = tuple(data['color'])
        self.species = data['species']
        self.label = data['label']
        self.fork_length_cm = data['fork_length_cm']
        self.cforager = vf.Forager(
            data['fork_length_cm'],                 # fork length (cm)
            data['mass_g'],                         # mass (g)
            initial_radius,                         # search radius (m)
            2.0,                                    # theta
            1.2 * data['focal_velocity_m_per_s'],   # mean_column_velocity
            0.3,                                    # saccade_time
            0.8,                                    # discrimination_threshold
            0.1,                                    # delta_0  -- Scales effect of angular size on tau; bigger delta_0 = harder detection.
            0.1,                                    # alpha_0  -- Scales effect of feature-based attention on tau; bigger alpha_0 = harder detection.
            0.03,                                   # Z_0      -- Scales effect of search rate / spatial attention on tau; smaller Z_0 = harder detection.
            1.0,                                    # c_1      -- Scales effect of FBA and saccade time on discrimination; bigger values incentivize longer saccades
            0.3,                                    # beta     -- Scales effect of set size on tau; larger beta = harder detection, more incentive to drop debris-laden prey
            data['bottom_z_m'],                     # bottom_z
            data['surface_z_m'],                    # surface_z
            int(data['temperature_C']),             # temperature (integer)
            0.05,                                   # bed_roughness
            1.0,                                    # lambda_c
            0.5,                                    # sigma_t
            0.03,                                   # base crypticity
            0.5,                                    # t_V
            INTERPOLATION_ROOT                      # base directory for maneuver interpolation files
        )

        for pc in data['prey_categories_characteristics'].values():
            if pc['prey_drift_density'] > 0:  # ignore prey categories that aren't found in the drift for a fish
                self.cforager.add_prey_category(
                    pc['number'],
                    pc['name'],
                    pc['mean_prey_length'] * 0.001,  # convert prey length from data units (mm) to model units (m)
                    pc['mean_prey_energy'],
                    pc['crypticity'],
                    pc['prey_drift_density'],
                    pc['debris_drift_density'],
                    pc['feature_size']
                )
        self.cforager.process_prey_category_changes()

    def calculate_proportion_bins(self):
        bodylength_binmins = np.array([0.0, 0.5, 1.0, 2.0, 4.0])
        bodylength_binmaxes = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
        distance_binmins = bodylength_binmins * 0.01 * self.fork_length_cm  # 0.01 converts cm to m
        distance_binmaxes = bodylength_binmaxes * 0.01 * self.fork_length_cm
        angle_binmins = np.array([0.0, 0.25*math.pi, 0.5*math.pi, 0.75*math.pi, math.pi, 1.25*math.pi])
        angle_binmaxes = np.array([0.25*math.pi, 0.5*math.pi, 0.75*math.pi, math.pi, 1.25*math.pi, 1.5*math.pi])
        distance_limit = self.cforager.get_radius()
        self.full_distance_bins = np.array([distance_binmins, distance_binmaxes]).T
        self.truncated_distance_bins = np.array([[min(x, distance_limit) for x in distance_binmins], [min(x, distance_limit) for x in distance_binmaxes]]).T
        angle_limit = self.cforager.get_theta()
        self.full_angle_bins = np.array([angle_binmins, angle_binmaxes]).T
        self.truncated_angle_bins = np.array([[min(x, angle_limit) for x in angle_binmins], [min(x, angle_limit) for x in angle_binmaxes]]).T

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
        self.calculate_proportion_bins()
        # Detection distance data
        distance_obj_total = 0
        for i, dbin in enumerate(self.truncated_distance_bins):
            binmin, binmax = list(dbin)
            labelmin, labelmax = list(self.full_distance_bins[i])
            predicted = self.cforager.proportion_of_detections_within(binmin, binmax, np.nan, np.nan, "All", "All")
            observed = self.fielddata['detection_distance_proportions'][i]
            distance_obj_total += (predicted - observed) ** 2
            vprint("For distance bin {0:.3f} to {1:.3f} m, predicted proportion {2:.3f}, observed proportion {3:.3f}.".format(labelmin, labelmax, predicted, observed))
        distance_part = (distance_obj_total / len(self.truncated_distance_bins)) * objective_weights['detection_distances_combined']
        # Detection angle data
        angle_obj_total = 0
        for i, dbin in enumerate(self.truncated_angle_bins):
            binmin, binmax = list(dbin)
            labelmin, labelmax = list(self.full_angle_bins[i])
            predicted = self.cforager.proportion_of_detections_within(np.nan, np.nan, binmin, binmax, "All", "All")
            observed = self.fielddata['detection_angle_proportions'][i]
            angle_obj_total += (predicted - observed) ** 2
            vprint("For angle bin {0:.3f} to {1:.3f} radians, predicted proportion {2:.3f}, observed proportion {3:.3f}.".format(labelmin, labelmax, predicted, observed))
        angle_part = (angle_obj_total / len(self.truncated_angle_bins)) * objective_weights['detection_angles_combined']
        # Diet data
        dietdata = [item for item in self.fielddata['diet_by_category'].values() if item['number'] is not None]
        diet_obj_total = 0
        diet_obj_count = 0
        for dd in sorted(dietdata, key=lambda x: x['number']):
            category_name = dd['name']
            observed = dd['diet_proportion']
            predicted = self.cforager.get_diet_proportion_for_prey_category(category_name)
            if predicted > 0 or observed > 0:
                diet_obj_count += 1
                diet_obj_total += (predicted - observed) ** 2
                vprint("For diet category '{0}', predicted proportion {1:.3f}, observed proportion {2:.3f}.".format(category_name, predicted, observed))
        diet_part = (diet_obj_total / diet_obj_count) * objective_weights['diet_proportions_combined']
        # NREI -- not used in objective function, just for curiosity/printing.
        predicted_NREI = self.cforager.NREI()
        observed_NREI = self.fielddata['empirical_NREI_J_per_s']
        objective_value = fa_rate_part + velocity_part + proportion_ingested_part + distance_part + angle_part + diet_part
        vprint("NREI: predicted {0:.5f} J/s, observed estimate {1:.5f} J/s.".format(predicted_NREI, observed_NREI))
        vprint("Objective function value is {0:.5f}. (Contributions: attempt rate {1:.3f}, velocity {2:.3f}, ingestion {3:.3f}, distance {4:.3f}, angle {5:.3f}, diet {6:.3f})".format(
            objective_value, fa_rate_part, velocity_part, proportion_ingested_part, distance_part, angle_part, diet_part))
        return objective_value

    def optimize(self, iterations, pack_size, verbose=True, use_chaos=False, use_dynamic_C=False, use_exponential_decay=False, use_levy=False, use_only_alpha=False, use_weighted_alpha=True):
        opt = vf.Optimizer(self.cforager, iterations, pack_size, verbose)
        opt.set_algorithm_options(use_chaos, use_dynamic_C, use_exponential_decay, use_levy, use_only_alpha, use_weighted_alpha)
        return opt.optimize_forager()  # returns array of fitnesses by step