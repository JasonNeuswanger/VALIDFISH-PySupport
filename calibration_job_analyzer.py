import numpy as np
import pandas as pd
from seaborn import jointplot, lmplot, hls_palette
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pickle
import pymysql
import os
import sys
from matplotlib import cm
import json
import inspectable_fish

# job_name = 'First Cluster Test'
# fish_group = 'calibration_five_of_each'

job_name = 'FirstFiveGrayling'
fish_group = 'calibration_five_grayling'

# job_name = 'FirstFiveDollies'
# fish_group = 'calibration_five_dollies'

# job_name = 'FirstFiveChinook'
# fish_group = 'calibration_five_chinook'


export_plots = True
show_plots = False

with open('Fish_Groups.json', 'r') as fgf:
    fish_groups = json.load(fgf)
with open('db_settings.pickle', 'rb') as handle:
    db_settings = pickle.load(handle)

fish_labels, fish_species = fish_groups[fish_group]
search_images_allowed = (fish_species == 'all' or fish_species == 'grayling')
fishes = [inspectable_fish.InspectableFish(fish_label) for fish_label in fish_labels]

figure_folder = os.path.join(os.path.sep, 'Users', 'Jason', 'Desktop', 'TempFig', job_name)
if not os.path.exists(figure_folder):
    os.makedirs(figure_folder)

with open('db_settings.pickle', 'rb') as handle:
    db_settings = pickle.load(handle)

db = pymysql.connect(host=db_settings['host'], port=db_settings['port'], user=db_settings['user'], passwd=db_settings['passwd'], db=db_settings['db'], autocommit=True)
cursor = db.cursor()
select_query = "SELECT * FROM jobs WHERE objective_function IS NOT NULL AND job_name='{0}'".format(job_name)
cursor.execute(select_query)
completed_job_data = cursor.fetchall()
Y_all = []
X_all = []
for row in completed_job_data:
    job_id, is_initial, read_job_name, read_fish_group, machine_assigned, start_time, progress, completed_time, objective_val_read, delta_0, alpha_tau, alpha_d, beta, A_0, t_s_0, discriminability, flicker_frequency, tau_0, nu = row
    Y_all.append(objective_val_read)
    if search_images_allowed:
        X_all.append([delta_0, alpha_tau, alpha_d, beta, A_0, t_s_0, discriminability, flicker_frequency, tau_0, nu])
    else:
        X_all.append([delta_0, beta, A_0, t_s_0, discriminability, flicker_frequency, tau_0, nu])
db.close()
X_all = np.asarray(X_all)
Y_all = np.asarray(Y_all).reshape(len(Y_all), 1)

X_best = X_all[:1]              # The X/Y_best arrays store the best value so far at every iteration of the calculation.
Y_best = Y_all[:1]
X_only_best = X_all[:1]         # The X/Y_only_best store values only where a new best was found.
Y_only_best = Y_all[:1]
i_only_best = np.array([0])

for i in range(len(Y_all)):
    if Y_all[i] < Y_best[-1]:
        X_best = np.vstack((X_best, X_all[i]))
        Y_best = np.vstack((Y_best, Y_all[i]))
        X_only_best = np.vstack((X_only_best, X_all[i]))
        Y_only_best = np.vstack((Y_only_best, Y_all[i]))
        i_only_best = np.vstack((i_only_best, i))
    else:
        X_best = np.vstack((X_best, X_best[-1]))
        Y_best = np.vstack((Y_best, Y_best[-1]))

# --------------------------------------------------------------------------------------------------------
# Load information on parameter bounds
# --------------------------------------------------------------------------------------------------------

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

# --------------------------------------------------------------------------------------------------------
# ACTUALLY EXPORT PLOTS
# --------------------------------------------------------------------------------------------------------

# Prep data structures for both plots

param_names = [item['name'] for item in domain]
df = pd.DataFrame(data=X_all, index=np.arange(0, len(X_all)), columns=param_names)  # 1st row as the column names
df['value'] = Y_all
df = df[abs(df['value']) < 1000000]

def color_for_value(value):
    def scale(x):
        return x  # identity for now, or use np.log
    vmin = scale(min(df['value']))
    vmax = scale(max(df['value']))
    proportion = scale((value - vmin) / (vmax - vmin))
    return cm.viridis_r(proportion)
colors = [color_for_value(value) for value in df['value']]

# Plot all explored values of each two-parameter combination together, explored points
# color-coded by the objective function value (yellow=best, blue=worst) at that point.
for i in range(len(param_names)):
    x_param = param_names[i]
    for j in range(i):
        y_param = param_names[j]
        print("Doing plot for {0} vs {1}.".format(x_param, y_param))
        g = jointplot(x=x_param, y=y_param, data=df, kind="kde", color="k", alpha=0.4)
        best_value = Y_all.min()
        marker_sizes = [(600 if item == best_value else 30) for item in df['value']]
        g.plot_joint(plt.scatter, c=colors, s=marker_sizes, linewidth=1, marker='+')
        g.ax_joint.collections[0].set_alpha(0)
        g.set_axis_labels(x_param, y_param)
        plt.xlim(*scaled_parameter_bounds[x_param])
        plt.ylim(*scaled_parameter_bounds[y_param])
        if export_plots: plt.savefig(os.path.join(figure_folder, "Joint_{0}_{1}.pdf".format(x_param, y_param)))
        if show_plots: plt.show()

# Plot distributions of individual parameter values tested
for param in param_names:
    print("Doing single plot for {0}.".format(param))
    g = lmplot(x=param, y='value', data=df, lowess=True, ci=None, scatter_kws={"s": 80})
    if export_plots: plt.savefig(os.path.join(figure_folder, "Single_{0}.pdf".format(param)))
    if show_plots: plt.show()

# Plot parameter and objective function values over time

print("Plotting values vs evaluations.")
#fig, ((ax1, ax4), (ax2, ax3)) = plt.subplots(2, 2, figsize=(9, 9))
fig = plt.figure(figsize=(9, 9))
gs1 = gridspec.GridSpec(2, 2)
ax1 = fig.add_subplot(gs1[0])
ax2 = fig.add_subplot(gs1[1])
ax3 = fig.add_subplot(gs1[2])
ax4 = fig.add_subplot(gs1[3])
# Subplot for objective function
ax1.plot(df['value'])
ax1.plot(i_only_best, Y_only_best, marker='o', color='r')
ax1.set_ylabel('Objective function')
ax1.set_xlabel('Evaluations')
# Subplot for all parameters
legend_handles = []
palette = hls_palette(len(X_all[0]), l=.4, s=.8)
for i, param_data in enumerate(X_all.T):
    handle = ax2.plot(param_data, color=palette[i], label=param_names[i])
    legend_handles.append(handle[0])
ax2.legend(handles=legend_handles, loc=2)
ax2.set_ylabel('Parameter value (or log10)')
ax2.set_xlabel('Evaluations')
# Subplot for best parameters
legend_handles = []
palette = hls_palette(len(X_best[0]), l=.4, s=.8)
for i, param_data in enumerate(X_best.T):
    handle = ax3.plot(param_data, color=palette[i], label=param_names[i])
    legend_handles.append(handle[0])
ax3.legend(handles=legend_handles, loc=2)
ax3.set_ylabel('Parameter value (or log10)')
ax3.set_xlabel('Evaluations')
# Subplot for distance between consecutive X
xdiffs = np.linalg.norm(X_all[1:]-X_all[:-1], axis=1)
ax4.plot(xdiffs)
ax4.set_ylabel('Distance between consecutive X')
ax4.set_xlabel('Evaluations')
fig.suptitle("Convergence for {0} with fish group {1}".format(job_name, fish_group))
gs1.tight_layout(fig, rect=[0, 0, 1, 0.97])
# Save
if export_plots: plt.savefig(os.path.join(figure_folder, "Convergence.pdf".format(param)))
plt.show()

# --------------------------------------------------------------------------------------------------------
# SAVE FISH STATES
# --------------------------------------------------------------------------------------------------------

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
    forager.optimize(200, 14, True, False, False, False, False, False, True)

for fish in fishes:
    optimize_forager_with_parameters(fish, *X_best[-1])
    fish.save_state()

# --------------------------------------------------------------------------------------------------------
# LOAD FISH STATES
# --------------------------------------------------------------------------------------------------------

for fish in fishes:
    fish.load_state()

# --------------------------------------------------------------------------------------------------------
# PLOT/ANALYZE INDIVIDUAL FISH FITS
# --------------------------------------------------------------------------------------------------------

# Chinook plots:
# 0, 1, 3 -- excellent
# 2 -- fish forages pretty far forward of predictions in an unpredicted cone pattern
# 4 -- fish ranges a bit farther than predicted in all fwd directions

# Dolly plots
# Mostly too arc-shaped, not getting a good cone when appropriate.

# Grayling
# All detecting stuff on the outer surface, must be some other factor in the cost
# function overriding spatial variables.

# For future optimizations, reduce the range of discriminability from 1.8 to 4
# And fix flicker_frequency at a good value based on this set of calculations.
# That should provide a bit more resolving power for the other parameters.
# Double check that we really are getting bad-maneuver cutoffs and other suitable math in the
# spatial data we're checking against plots.

test_fish = fishes[12]
fig3d = test_fish.plot_predicted_detection_field(gridsize=50j, colorMax=None, bgcolor=(0, 0, 0))

#test_fish.foraging_point_distribution_distance(verbose=False, plot=True)

# --------------------------------------------------------------------------------------------------------
# Save individual fish fits to file
# --------------------------------------------------------------------------------------------------------

fit_file_path = os.path.join(figure_folder, "FitStatistics - {0}.txt".format(job_name))
sys.stdout = open(fit_file_path, "w")
for fish in fishes:
    print("{1}: Fit statistics for {0}:\n".format(fish.label, job_name))
    fish.evaluate_fit(verbose=True)
sys.stdout = sys.__stdout__
print("Saved fit statistics to ", fit_file_path)

# Issue: Chinook are consistently predicted to prefer faster water than they actually use,
# even when the model is calibrated only for Chinook.