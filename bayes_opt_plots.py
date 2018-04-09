import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm

# --------------------------------------------------------------------------------------------------------
# CODE TO BE EXECUTED IN CONSOLE AFTER RUNNING bayes_opt_test.py IN CONSOLE AS WELL.
# It depends on the variables 'bo' and 'param_limits' from there.
# --------------------------------------------------------------------------------------------------------

FIGURE_FOLDER = "/Users/Jason/Desktop/TempFig/"

# Prep data structures for both plots
df = pd.DataFrame(bo.res['all']['params'])
df['value'] = bo.res['all']['values']
df = df[abs(df['value']) < 1e10]
def color_for_value(value):
    vmin = min(df['value'])
    vmax = max(df['value'])
    proportion = (value - vmin) / (vmax - vmin)
    return cm.viridis(proportion)
colors = [color_for_value(value) for value in bo.res['all']['values']]
param_names = ('delta_0', 'alpha_0', 'beta', 'Z_0', 'c_1', 'discriminability', 'sigma_t', 'tau_0')

# Plot all explored values of each two-parameter combination together, explored points
# color-coded by the objective function value (yellow=best, blue=worst) at that point.
for i in range(len(param_names)):
    x_param = param_names[i]
    for j in range(i):
        y_param = param_names[j]
        g = sns.jointplot(x=x_param, y=y_param, data=df, kind="kde", color="k", alpha=0.4)
        max_value = bo.res['max']['max_val']
        marker_sizes = [(600 if item == max_value else 30) for item in df['value']]
        g.plot_joint(plt.scatter, c=colors, s=marker_sizes, linewidth=1, marker='+')
        g.ax_joint.collections[0].set_alpha(0)
        g.set_axis_labels(x_param, y_param)
        plt.xlim(*param_limits[x_param])
        plt.ylim(*param_limits[y_param])
        plt.savefig("{0}/Joint_{1}_{2}.pdf".format(FIGURE_FOLDER, x_param, y_param))
        plt.show()

# Plot distributions of individual parameter values tested
for param in param_names:
    g = sns.lmplot(x=param, y='value', data=df, lowess=True, ci=None, scatter_kws={"s": 80})
    plt.savefig("{0}/Single_{1}.pdf".format(FIGURE_FOLDER, param))
    plt.show()
