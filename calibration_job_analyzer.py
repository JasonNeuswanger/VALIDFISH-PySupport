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

job_name = 'SecondFiveOfEach'
#job_name = 'SecondFiveGrayling'

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
        plt.xlim(*runner.scaled_parameter_bounds[x_param])
        plt.ylim(*runner.scaled_parameter_bounds[y_param])
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
fig.suptitle("Convergence for {0}".format(job_name))
gs1.tight_layout(fig, rect=[0, 0, 1, 0.97])
# Save
if export_plots: plt.savefig(os.path.join(figure_folder, "Convergence.pdf".format(param)))
plt.show()

# --------------------------------------------------------------------------------------------------------
# SAVE FISH STATES
# --------------------------------------------------------------------------------------------------------

for fish in runner.fishes:
    print("Optimizing strategy for fish {0}.".format(fish.label))
    runner.optimize_forager_with_parameters(fish, *X_best[-1])
    fish.save_state()

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
# PLOT TAU COMPONENTS FOR ALL FISH
# --------------------------------------------------------------------------------------------------------

for fish in runner.fishes:
    fish.plot_tau_components(x=0.03, y=0.03)

# --------------------------------------------------------------------------------------------------------
# PLOT/ANALYZE INDIVIDUAL FISH FITS
# --------------------------------------------------------------------------------------------------------

# Loom seems unimportant -- exclude from analyses?
#


test_fish = runner.fishes[14]
runner.optimize_forager_with_parameters(test_fish, *X_best[-1])
test_fish.cforager.print_strategy()

fig3d = test_fish.plot_predicted_detection_field(gridsize=50j, colorMax=None, bgcolor=(0, 0, 0))

test_fish.foraging_point_distribution_distance(verbose=False, plot=True)

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



