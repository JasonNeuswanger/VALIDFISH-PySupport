import numpy as np
import pandas as pd
from seaborn import jointplot, lmplot, hls_palette
import matplotlib.pyplot as plt
from GPyOpt import util
from matplotlib import cm

# --------------------------------------------------------------------------------------------------------
# CODE TO BE EXECUTED IN CONSOLE AFTER RUNNING bayes_opt_test.py IN CONSOLE AS WELL.
# It depends on the variables 'bo' and 'param_limits' from there.
# --------------------------------------------------------------------------------------------------------

FIGURE_FOLDER = "/Users/Jason/Desktop/TempFig/"

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
        g = jointplot(x=x_param, y=y_param, data=df, kind="kde", color="k", alpha=0.4)
        best_value = Y_all.min()
        marker_sizes = [(600 if item == best_value else 30) for item in df['value']]
        g.plot_joint(plt.scatter, c=colors, s=marker_sizes, linewidth=1, marker='+')
        g.ax_joint.collections[0].set_alpha(0)
        g.set_axis_labels(x_param, y_param)
        plt.xlim(*scaled_parameter_bounds[x_param])
        plt.ylim(*scaled_parameter_bounds[y_param])
        plt.savefig("{0}/Joint_{1}_{2}.pdf".format(FIGURE_FOLDER, x_param, y_param))
        plt.show()

# Plot distributions of individual parameter values tested
for param in param_names:
    g = lmplot(x=param, y='value', data=df, lowess=True, ci=None, scatter_kws={"s": 80})
    plt.savefig("{0}/Single_{1}.pdf".format(FIGURE_FOLDER, param))
    plt.show()

# Plot parameter and objective function values over time

fig, ((ax1, ax4), (ax2, ax3)) = plt.subplots(2, 2, figsize=(9, 9))
# Subplot for objective function
ax1.plot(df['value'])
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
ax3.set_xlabel('Evaluations beyond initial')
# Subplot for distance between consecutive X
xdiffs = np.linalg.norm(X_all[1:]-X_all[:-1], axis=1)
ax4.plot(xdiffs)
ax4.set_ylabel('Distance between consecutive X')
ax4.set_xlabel('Evaluations')
# Save
plt.savefig("{0}/Convergence.pdf".format(FIGURE_FOLDER, param))
plt.show()
