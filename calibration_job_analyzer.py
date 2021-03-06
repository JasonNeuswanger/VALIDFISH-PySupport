import numpy as np
import pandas as pd
from seaborn import jointplot, lmplot, hls_palette
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
import sys
from matplotlib import cm

from job_runner import JobRunner

export_plots = True
show_plots = False

job_name = 'Fifteen6'
runner = JobRunner(job_name, readonly=True)

figure_folder = os.path.join(os.path.sep, 'Users', 'Jason', 'Desktop', 'TempFig', job_name)
if not os.path.exists(figure_folder):
    os.makedirs(figure_folder)

runner.cursor.execute("SELECT * FROM job_results WHERE objective_value IS NOT NULL AND job_name=\"{0}\"".format(job_name))
completed_job_data = runner.cursor.fetchall()
Y_all = []
X_all = []
for row in completed_job_data:
    Y_all.append(row['objective_value'])
    X_all.append([row[key] for key in runner.parameters_to_optimize])
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
# ACTUALLY EXPORT PLOTS
# --------------------------------------------------------------------------------------------------------

# Prep data structures for both plots

param_names = [item['name'] for item in runner.domain]
df = pd.DataFrame(data=X_all, index=np.arange(0, len(X_all)), columns=param_names)  # 1st row as the column names
df['value'] = Y_all
df = df[abs(df['value']) < 1000000]
cf = runner.fishes[0].cforager

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
    parameter_name = param_names[i]
    p = cf.get_parameter_named(parameter_name)
    def scaled_value(value):
        plotval = 10**value if cf.is_parameter_log_scaled(p) else value
        return cf.transform_parameter_value_as_proportion(p, plotval)
    vscale = np.vectorize(scaled_value)
    handle = ax2.plot(vscale(param_data), color=palette[i], label=param_names[i])
    legend_handles.append(handle[0])
ax2.legend(handles=legend_handles, loc=2)
ax2.set_ylabel('Parameter or log10 (scaled)')
ax2.set_xlabel('Evaluations')
# Subplot for best parameters
legend_handles = []
palette = hls_palette(len(X_best[0]), l=.4, s=.8)
for i, param_data in enumerate(X_best.T):
    parameter_name = param_names[i]
    p = cf.get_parameter_named(parameter_name)
    def scaled_value(value):
        plotval = 10 ** value if cf.is_parameter_log_scaled(p) else value
        return cf.transform_parameter_value_as_proportion(p, plotval)
    vscale = np.vectorize(scaled_value)
    handle = ax3.plot(vscale(param_data), color=palette[i], label=param_names[i])
    legend_handles.append(handle[0])
ax3.legend(handles=legend_handles, loc=2)
ax3.set_ylabel('Parameter or log10 (scaled)')
ax3.set_xlabel('Evaluations')
# Subplot for distance between consecutive X
xdiffs = np.linalg.norm(X_all[1:]-X_all[:-1], axis=1)
ax4.plot(xdiffs)
ax4.set_ylabel('Distance between consecutive X')
ax4.set_xlabel('Evaluations')
fig.suptitle("Convergence for {0}".format(job_name))
gs1.tight_layout(fig, rect=[0, 0, 1, 0.97])
# Save
if export_plots: plt.savefig(os.path.join(figure_folder, "Convergence.pdf"))
plt.show()

def color_for_value(value):
    def scale(x):
        return x  # identity for now, or use np.log
    vmin = min(df['value'])
    vmax = max(df['value'])
    proportion = scale((value - vmin) / (vmax - vmin))
    return cm.viridis_r(proportion)
colors = [color_for_value(value) for value in df['value']]

# Plot all explored values of each two-parameter combination together, explored points
# color-coded by the objective function value (yellow=best, blue=worst) at that point.

show_plots = False
for i in range(len(param_names)):
    x_param = param_names[i]
    for j in range(i):
        y_param = param_names[j]
        print("Doing plot for {0} vs {1}.".format(x_param, y_param))
        g = jointplot(x=x_param, y=y_param, data=df, kind="kde", color="k", alpha=0.4)
        best_value_index = Y_all.argmin()
        best_x_param_value = df[x_param][best_value_index]
        best_y_param_value = df[y_param][best_value_index]
        x_param_enum = cf.get_parameter_named(x_param)
        x_label = "log10({0})".format(x_param) if cf.is_parameter_log_scaled(x_param_enum) else x_param
        y_param_enum = cf.get_parameter_named(y_param)
        y_label = "log10({0})".format(y_param) if cf.is_parameter_log_scaled(y_param_enum) else y_param
        g.plot_joint(plt.scatter, c=colors, s=30, linewidth=1, marker='+')
        g.ax_joint.collections[0].set_alpha(0)
        g.set_axis_labels(x_label, y_label)
        g.ax_joint.plot([best_x_param_value], [best_y_param_value], c=(1, 0, 0), markersize=10, marker='o')
        plt.xlim(*runner.scaled_parameter_bounds[x_param])
        plt.ylim(*runner.scaled_parameter_bounds[y_param])
        if export_plots: plt.savefig(os.path.join(figure_folder, "Joint_{0}_{1}.pdf".format(x_param, y_param)))
        if show_plots:
            plt.show()
        else:
            plt.close()

# Plot distributions of individual parameter values tested

for param in param_names:
    print("Doing single plot for {0}.".format(param))
    g = lmplot(x=param, y='value', data=df, lowess=True, ci=None, scatter_kws={"s": 80})
    best_value_index = Y_all.argmin()
    best_param_value = df[param][best_value_index]
    x_param_enum = cf.get_parameter_named(param)
    g.axes[0,0].axvline(best_param_value, color=(1, 0, 0))
    x_label = "log10({0})".format(param) if cf.is_parameter_log_scaled(x_param_enum) else param
    g.set_axis_labels(x_label, 'objective function value')
    if export_plots: plt.savefig(os.path.join(figure_folder, "Single_{0}.pdf".format(param)))
    if show_plots:
        plt.show()
    else:
        plt.close()

# todo I should try to set parameters so that values on the low end of the range correspond
# todo to small/no effect, and values on the high end of the range a large effect. Just
# todo use 1/const others.

# --------------------------------------------------------------------------------------------------------
# SAVE FISH STATES
# --------------------------------------------------------------------------------------------------------

for fish in runner.fishes:
    print("Optimizing strategy for fish {0}.".format(fish.label))
    runner.optimize_forager_with_parameters(fish, *X_best[-1])
    fish.save_state()

# --------------------------------------------------------------------------------------------------------
# EXPORT EVERYTHING TO FILE
# --------------------------------------------------------------------------------------------------------

for fish in runner.fishes:
    fish.export_full_analysis(base_folder="/Users/Jason/Desktop/TempFig/", job_name=job_name)

# --------------------------------------------------------------------------------------------------------
# LOAD FISH STATES
# --------------------------------------------------------------------------------------------------------

for fish in runner.fishes:
    fish.load_state()

# --------------------------------------------------------------------------------------------------------
# PRINT STRATEGIES AND PARAMETERS
# --------------------------------------------------------------------------------------------------------

runner.fishes[0].cforager.print_parameters()

for fish in runner.fishes:
    print("\nStrategy for fish {0}".format(fish.label))
    fish.cforager.print_strategy()

# --------------------------------------------------------------------------------------------------------
# PLOT DISCRIMINATION AND DETECTION MODELS FOR ALL FISH
# --------------------------------------------------------------------------------------------------------

for fish in runner.fishes:
    fish.plot_discrimination_model(x=0.03, z=0.03)
    fish.plot_detection_model(x=0.03, z=0.03)

# --------------------------------------------------------------------------------------------------------
# PLOT/ANALYZE INDIVIDUAL FISH FITS
# --------------------------------------------------------------------------------------------------------

# i wonder if it makes more sense to calculate parameters with velocity fixed to field values
# and then see if predicted optimal velocities match observations after fitting the model based
# solely on the other considerations?

# fish 2 is insufficiently forward-directed
# fish 3 is too forward-directed
# dollies generally are slightly insufficiently forward-directed

# for grayling, i need to check if discrimination is unrealistically hard, and if making it easier would
# give better bounds of profitability an lead to realistic foraging

# are fish still generally eating smaller stuff than predicted, especially chinook?
# consider Bayesian estimation of drift AND item energy contents, maybe try it out predicting subsamples?
# I'm getting occasional weird results on profitability of larger size classes because they're worth less energy due to
# the specific taxonomic composition of a fairly small net sample that probably doesn't reflect the fish's thinking.

# I want some kind of plotting function to look at the fit quality at a glance.

# Why are the foraging distribution distances fairly low even for fairly poor fits?

# Can I estimate velocity better?

# Loom effect seems closed to maxed out despite having a pretty small effect... allow it to be larger?

# TRY NEW OPTIMIZATION ALGORITHM FROM WANG ET AL 2018, ENSEMBLE BAYESIAN OPTIMIZATION
# https://github.com/zi-w/Ensemble-Bayesian-Optimization


# NEW PLOT TO DO: Strategy vs NREI -- how do I not have this already?


test_fish = runner.fishes[0]
test_fish.plot_predicted_detection_field_2D(show=False, figure_folder="/Users/Jason/Desktop/")
test_fish.plot_predicted_depletion_field_2D(show=False, figure_folder="/Users/Jason/Desktop/")


test_fish.map_tau_effects()
test_fish.map_discrimination_model()


test_fish.export_full_analysis(base_folder="/Users/Jason/Desktop/TempFig/", job_name=job_name)


fig3d = test_fish.plot_predicted_detection_field_3D(colorMax=None, gridsize=80j, bgcolor=(0, 0, 0), surfaces=False, velocities=False)



fig3d = test_fish.plot_predicted_depletion_field_3D(colorMax=None, gridsize=80j, bgcolor=(0, 0, 0), surfaces=False)



print(test_fish.cforager.format_prey_to_print())

# Large prey with a concentration of 0 are unprofitable because profitability takes into account the probability
# that an item is prey vs debris, which is based on concentrations and discrimination probabilities, and with
# concentration of 0 that probability is 0, so all the items in that class are assumed to be unprofitable debris.

test_fish.plot_diet_proportions()

test_fish.cforager.relative_pursuits_by_position(-0.24,0.2,0.0)
test_fish.cforager.relative_pursuits_by_position(0.24,0.2,0.0)

test_fish.cforager.relative_pursuits_by_position(0.21,0.2,0.0)
test_fish.cforager.relative_pursuits_by_position(0.21,0.2,0.0)


# the results from relative_pursuits_by_position match up well the first time and second time
# when detection probability cache is disabled, but badly mismatch the second time when it is
# enabled. given that the results when it's disabled are closer to the first of the mismatched
# values, ie when everything is directly calculated rather than pulled from the cache, it seems
# there's a problem with the values the cache is returning
# I added some prints of cache vales, and the problem went away
# Removed them, and it stayed away
# I have no idea what is going on.



test_fish = runner.fishes[9]
test_fish.cforager.relative_pursuits_by_position(-0.2,0.2,0.0)

# some problem with caching of relative_pursuits_by_position, first call on negative side gives negative value
# this happens on the positive side with the first call, too
# how does this affect things?

test_fish.plot_predicted_detection_field_2D()

test_fish.plot_predicted_depletion_field_2D()



test_fish.cforager.print_parameters()
test_fish.evaluate_fit()


test_fish.plot_detection_model()
test_fish.plot_discrimination_model()
test_fish.cforager.print_parameters()

test_fish.foraging_point_distribution_distance(verbose=False, plot=True)
test_fish.plot_variable_reports()


# TOTAL ENERGY PLOT

# import seaborn as sns
# numpts = 30
# r = test_fish.cforager.get_max_radius()
# x = np.linspace(-r, r, numpts)
# y = np.linspace(-r, r, numpts)
# xg, yg = np.meshgrid(x, y)
# zg = np.empty(np.shape(xg))
# for i in range(len(x)):
#     for j in range(len(y)):
#         zg[j, i] = test_fish.cforager.depleted_prey_concentration_total_energy(x[i], y[j], 0)
# xgd = np.vstack((xg, xg))
# ygd = np.vstack((-np.flipud(yg), yg))
# zgd = np.vstack((np.flipud(zg), zg))
# sns.set_style('white')
# efig, (ax) = plt.subplots(1, 1, facecolor='w', figsize=(3.25, 2.6), dpi=300)
# cf = ax.contourf(xgd, ygd, zgd, 10, cmap='viridis_r')
# efig.colorbar(cf, ax=ax, shrink=0.9)
# efig.show()
#
# test_fish.cforager.depleted_prey_concentration_total_energy(0.03, -0.4, 0)

# make probability response plots scale to (0,1)



#runner.optimize_forager_with_parameters(test_fish, *X_best[-1])
#test_fish.cforager.print_strategy()

# now need a way to show attempt rate, focal velocity, proportion ingested, NREI

# just make each its own graph
#



test_fish.evaluate_fit()

test_fish.cforager.print_strategy()

# --------------------------------------------------------------------------------------------------------
# Save individual fish fits to file
# --------------------------------------------------------------------------------------------------------

fit_file_path = os.path.join(figure_folder, "FitStatistics - {0}.txt".format(job_name))
sys.stdout = open(fit_file_path, "w")
for fish in runner.fishes:
    print("{1}: Fit statistics for {0}:\n".format(fish.label, job_name))
    fish.evaluate_fit(verbose=True)
sys.stdout = sys.__stdout__
print("Saved fit statistics to ", fit_file_path)

# Issue: Chinook are consistently predicted to prefer faster water than they actually use,
# even when the model is calibrated only for Chinook.

# --------------------------------------------------------------------------------------------------------
# Manipulate a fish fit
# --------------------------------------------------------------------------------------------------------

import importlib
PYVALIDFISH_FILE = "/Users/Jason/Dropbox/Drift Model Project/Calculations/VALIDFISH/VALIDFISH/cmake-build-release/pyvalidfish.cpython-35m-darwin.so"
spec = importlib.util.spec_from_file_location("pyvalidfish",  PYVALIDFISH_FILE)
vf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vf)

test_fish.cforager.print_strategy()
test_fish.cforager.print_parameters()
test_fish.evaluate_fit()

test_fish.load_state()
test_fish.cforager.set_strategy(vf.Forager.Strategy.mean_column_velocity, 0.2)
test_fish.cforager.set_parameter(vf.Forager.Parameter.tau_0, 0.05)
test_fish.cforager.set_parameter(vf.Forager.Parameter.delta_0, 1e-3)

test_fish.cforager.set_parameter(vf.Forager.Parameter.tau_0, 0.1)
test_fish.cforager.set_parameter(vf.Forager.Parameter.nu, 0.00001)
test_fish.cforager.set_parameter(vf.Forager.Parameter.beta, 0.5)


fig3d = test_fish.plot_predicted_detection_field(gridsize=50j, colorMax=None, bgcolor=(1, 1, 1))

test_fish.foraging_point_distribution_distance(verbose=False, plot=True)

test_fish.plot_tau_components(x=0.05, y=0.05)

opts=test_fish.optimize(200, 14, True, False, False, False, False, False, True)



