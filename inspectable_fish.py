from mayavi import mlab
import Fish3D
import numpy as np
import scipy
import math
import os
from pandas import DataFrame
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
from matplotlib import gridspec
from seaborn import hls_palette, set_style
from tvtk.util.ctf import PiecewiseFunction
import importlib.util
PYVALIDFISH_FILE = "/Users/Jason/Dropbox/Drift Model Project/Calculations/VALIDFISH/VALIDFISH/cmake-build-release/pyvalidfish.cpython-35m-darwin.so"
spec = importlib.util.spec_from_file_location("pyvalidfish",  PYVALIDFISH_FILE)
vf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vf)

import field_test_fish as ftf

class InspectableFish(ftf.FieldTestFish):

    def __init__(self, fish_label):
        super(InspectableFish, self).__init__(fish_label)
        self.response_labels = {
            self.cforager.NREI: "NREI (J/s)",
            self.cforager.tau: "Tau (s)",
            self.cforager.detection_probability: "Detection probability",
            self.cforager.get_foraging_attempt_rate: "Foraging attempt rate (/s)",
            self.cforager.passage_time: "Time available to detect (s)"
        }

    def print_description(self):
        print("             Species: ", self.species)
        print("               Label: ", self.label)
        print("         Fork Length: {0:.2f} cm".format(self.fork_length_cm))
        print("                Mass: {0:.2f} g".format(self.cforager.get_mass_g()))
        print("         Temperature: {0:.2f} C".format(self.fielddata['temperature_C']))
        print("Mean Column Velocity: {0:.2f} m/s".format(self.fielddata['focal_velocity_m_per_s']))

    def export_full_analysis(self, **kwargs):
        base_folder = kwargs.get('base_folder', "~/Desktop")
        job_name = kwargs.get('job_name', "No Job")
        fish_folder = os.path.join(base_folder, job_name, self.species, self.label)
        print("Exporting results for {0} to {1}.".format(self.label, fish_folder))
        if not os.path.exists(fish_folder): os.makedirs(fish_folder)
        print("Making text file for strategy, parameters, goodness-of-fit, and analytics.")
        with open(os.path.join(fish_folder, "{0}.txt".format(self.label)), 'w') as f:
            with redirect_stdout(f):
                print("Results for ", self.label)
                print("\n--------------------------------DESCRIPTION---------------------------------\n")
                self.print_description()
                print("\n---------------------------------PREY TYPES---------------------------------\n")
                print(self.cforager.format_prey_to_print())
                print("\n------------------------------GOODNESS OF FIT-------------------------------\n")
                self.evaluate_fit()
                print("\n---------------------------------STRATEGY-----------------------------------\n")
                print(self.cforager.format_strategy_to_print())
                print("\n--------------------------------PARAMETERS----------------------------------\n")
                print(self.cforager.format_parameters_to_print())
                print("\n---------------------------------ANALYTICS----------------------------------\n")
                print(self.cforager.format_analytics_to_print())
        print("Plotting foraging point predicted vs observed distributions.")
        self.foraging_point_distribution_distance(verbose=False, plot=True, show=False, figure_folder=fish_folder)
        print("Making discrimination model plots.")
        self.plot_discrimination_model(show=False, figure_folder=fish_folder)
        print("Making detection model plots.")
        self.plot_detection_model(show=False, figure_folder=fish_folder)
        print("Making rear-view plots of detection probabilities.")
        self.plot_detection_probabilities_rear_view(show=False, figure_folder=fish_folder)
        print("Plotting bounds of profitability.")
        self.plot_bounds_of_profitability(show=False, figure_folder=fish_folder)
        print("Making predicted depletion field top-view plot.")
        self.plot_predicted_depletion_field_2D(show=False, figure_folder=fish_folder)
        print("Making predicted detection field top-view plot.")
        self.plot_predicted_detection_field_2D(show=False, figure_folder=fish_folder)
        print("Plotting diet proportions.")
        self.plot_diet_proportions(show=False, figure_folder=fish_folder)
        print("Plotting water velocity.")
        self.plot_water_velocity(show=False, figure_folder=fish_folder)
        print("Plotting discrimination probabilities map.")
        self.map_discrimination_model(show=False, figure_folder=fish_folder)
        print("Plotting tau effects map.")
        self.map_tau_effects(show=False, figure_folder=fish_folder)
        print("Making detailed variable report plots (takes a while).")
        self.plot_variable_reports(show=False, figure_folder=fish_folder)
        print("Finished plots for fish {0}.\n\n".format(self.label))

    def plot_3D_field(self, func, **kwargs):
        gridsize = kwargs.get('gridsize', 50j)
        stopButton = kwargs.get('stopButton', False)
        colorMax = kwargs.get('colorMax', None)
        vfunc = np.vectorize(func)
        r = 1.05 * 100 * self.cforager.get_max_radius()
        x, y, z = np.mgrid[-r:r:gridsize, -r:r:gridsize, -r:r:gridsize]
        s = vfunc(x, y, z)
        figname = "3D Fish Test Figure"
        myFig = mlab.figure(figure=figname, bgcolor=kwargs.get('bgcolor', (0, 0, 0)), size=(1024, 768))

        mlab.clf(myFig)
        head_position = np.array((0, 0, 0))
        tail_position = np.array((0, -self.fork_length_cm, 0))
        Fish3D.fish3D(head_position, tail_position, self.species, myFig, color=self.color, world_vertical=np.array([0, 0, 1]))
        vmax = np.max(s) if colorMax is None else colorMax
        vol = mlab.pipeline.volume(mlab.pipeline.scalar_field(x, y, z, s), vmin=0.0, vmax=vmax)

        raw_predictions = s.flatten()
        predictions = sorted(raw_predictions[raw_predictions > 0])
        sums = [0]
        i = 0
        for i in range(len(predictions)):
            sums.append(sums[i] + predictions[i])
        sums = np.array(sums[1:])
        sums /= sums[-1]
        sums_interp = scipy.interpolate.interp1d(predictions, sums, fill_value="extrapolate", assume_sorted=True)

        def rel_pursuits_value_percentile(pct):
            def objfn(x):
                return abs(sums_interp(x) - pct)
            return scipy.optimize.minimize(objfn, [1], method='L-BFGS-B', bounds=[(min(predictions), max(predictions))]).x[0]

        def my_otf_value(v, exponent, divisor):
            return (v / divisor) ** exponent

        flat_s = s.flatten()
        otf = PiecewiseFunction()
        otf.add_point(0.0, 0.0)
        otf_exponent = kwargs.get("otf_exponent", 1.8)
        otf_divisor = kwargs.get("otf_divisor", 4)
        for v in np.linspace(0.05, 1.0, 40):
            otf_val = my_otf_value(v, otf_exponent, otf_divisor)
            otf.add_point(rel_pursuits_value_percentile(v), otf_val)
        vol._otf = otf
        vol._volume_property.set_scalar_opacity(otf)

        if kwargs.get('colorbar', False):
            mlab.colorbar(vol)

        if kwargs.get('show_fielddata', True):
            (px, py, pz) = 100 * np.transpose(np.asarray(self.field_detection_positions))
            point_radius = 0.005 * self.fork_length_cm
            d = np.repeat(2 * point_radius, px.size)  # creates an array of point diameters
            mlab.points3d(px, py, pz, d, color=kwargs.get('pointcolor', self.color), scale_factor=8, resolution=12,
                          opacity=1.0, figure=myFig)

        if kwargs.get('surfaces', True):
            sx = []
            sy = []
            sz = []
            bz = []
            rspace = np.linspace(-r, r, 100)
            for pbx in rspace:
                for pby in rspace:
                    if np.sqrt(pbx * pbx + pby * pby) < 100 * self.cforager.get_max_radius():
                        sx.append(pbx)
                        sy.append(pby)
                        bz.append(100 * self.fielddata['bottom_z_m'])
                        sz.append(100 * self.fielddata['surface_z_m'])
            pts = mlab.points3d(sx, sy, bz)
            bottom_mesh = mlab.pipeline.delaunay2d(pts)
            mlab.pipeline.surface(bottom_mesh, opacity=0.4, color=(0.7, 0.4, 0.3))
            pts.remove()
            pts = mlab.points3d(sx, sy, sz)
            surface_mesh = mlab.pipeline.delaunay2d(pts)
            mlab.pipeline.surface(surface_mesh, opacity=0.1, color=(0.0, 1.0, 1.0))
            pts.remove()
        if kwargs.get('velocities', True):
            def velocity_function_unvectorized(vx, vy, vz, ):
                return (0, -100 * self.cforager.water_velocity(vz), 0)

            velocity_function = np.vectorize(velocity_function_unvectorized)
            velocities_z = 0.9 * np.linspace(100 * self.fielddata['bottom_z_m'], 100 * self.fielddata['surface_z_m'], 8)
            velocities_x = np.repeat(0, len(velocities_z))
            velocities_y = np.repeat(-self.fork_length_cm * 1.3, len(velocities_z))
            mlab.quiver3d(velocities_x, velocities_y, velocities_z, velocity_function, color=(1, 1, 0))

        mlab.show(stop=stopButton)
        return myFig

    def plot_predicted_detection_field_3D(self, **kwargs):
        def detection_func(x, y, z):
            return self.cforager.relative_pursuits_by_position(0.01 * x, 0.01 * y, 0.01 * z)
        return self.plot_3D_field(detection_func, **kwargs)

    def plot_predicted_depletion_field_3D(self, **kwargs):
        max_drift_energy = self.cforager.depleted_prey_concentration_total_energy(100, 100, 100)
        def depletion_func(x, y, z):
            return max_drift_energy - self.cforager.depleted_prey_concentration_total_energy(0.01 * x, 0.01 * y, 0.01 * z)
        return self.plot_3D_field(depletion_func, **kwargs)

    def plot_water_velocity(self, **kwargs):
        bottom_z = self.fielddata['bottom_z_m']
        surface_z = self.fielddata['surface_z_m']
        zvalues = np.linspace(bottom_z, surface_z, 100)
        velocities = [self.cforager.water_velocity(z) for z in zvalues]
        plt.figure(figsize=(5, 5))
        ax = plt.axes()
        ax.plot(zvalues, velocities)
        ax.set_xlabel("Z coordinate (m), from bottom to surface")
        ax.set_ylabel("Water velocity (m/s)")
        plt.tight_layout()
        self.finish_figure("Water Velocity", **kwargs)

    # def _roc_curve_data(self, pt):
    #     perceptual_sigma = pt.get_perceptual_sigma()
    #     def normal_cdf(x):
    #         return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    #     def p_false_positive(disc_thresh):
    #         return 1.0 - normal_cdf(disc_thresh / perceptual_sigma)
    #     def p_true_hit(disc_thresh):
    #         return 1.0 - normal_cdf((disc_thresh - self.cforager.get_parameter(vf.Forager.Parameter.discriminability)) / perceptual_sigma)
    #     linsp = np.linspace(-10, 10, 300)
    #     logsp = np.logspace(-1, 10, 50)
    #     thresholds = np.sort(np.concatenate([np.flip(-logsp, axis=0), logsp, linsp], axis=0))
    #     probabilities = [(p_false_positive(thresh), p_true_hit(thresh)) for thresh in thresholds]
    #     pfps, pths = zip(*probabilities)  # unzip the list
    #     return list(pfps), list(pths)
    #
    # def plot_roc_curves(self, **kwargs):
    #     fig = plt.figure(figsize=(5, 5))
    #     ax = plt.axes()
    #     legend_handles = []
    #     pts = self.cforager.get_prey_types()
    #     palette = hls_palette(len(pts), l=.3, s=.8)
    #     for i, pt in enumerate(pts):
    #         prey_x = kwargs.get('x', 0.5 * pt.get_max_visible_distance())
    #         prey_z = kwargs.get('z', 0.5 * pt.get_max_visible_distance())
    #         x, y = self._roc_curve_data(pt)
    #         dot_x = pc.get_false_positive_probability()
    #         dot_y = pc.get_true_hit_probability()
    #         handle = ax.plot(x, y, color=palette[i], label=pt.get_name())
    #         legend_handles.append(handle[0])
    #         ax.plot([dot_x], [dot_y], marker='o', markersize=5, color=palette[i])
    #     ax.legend(handles=legend_handles, loc=4)
    #     ax.set_xbound(lower=-0.02, upper=1.02)
    #     ax.set_ybound(lower=-0.02, upper=1.02)
    #     ax.set_aspect('equal', 'datalim')
    #     ax.set_xlabel("False positive probability")
    #     ax.set_ylabel("True hit probability")
    #     plt.tight_layout()
    #     plt.show()

    def plot_angle_response_to_velocity(self, **kwargs):
        # We want to be able to plot the effects of one strategy variable, or maybe later
        # one parameter, on the optimal strategy (all other strategy variable).
        previous_velocity = self.cforager.get_strategy(vf.Forager.Strategy.mean_column_velocity)
        previous_sigma_A = self.cforager.get_strategy(vf.Forager.Strategy.sigma_A)
        bounds = self.cforager.get_strategy_bounds(vf.Forager.Strategy.mean_column_velocity)
        plot_x = np.linspace(bounds[0], bounds[1], kwargs.get("n_points", 10))
        def f(x):
            opt = vf.Optimizer(self.cforager, kwargs.get("iterations", 50), kwargs.get("pack_size", 7), kwargs.get("verbose", True))
            opt.add_context(vf.Forager.Strategy.mean_column_velocity, x)
            opt.optimize_forager()
            optimal_angle = self.cforager.get_strategy(vf.Forager.Strategy.sigma_A)
            return optimal_angle
        plot_y = np.array([f(x) for x in plot_x])
        fig = plt.figure(figsize=(5, 5))
        ax = plt.axes()
        ax.plot(plot_x, plot_y)
        ax.plot([previous_velocity], [previous_sigma_A], marker='o', markersize=5, color='r')
        ax.set_xlabel("Mean column velocity (m/s)")
        ax.set_ylabel("Optimal sigma_A")
        plt.tight_layout()
        self.finish_figure("Optimal sigma_A vs Velocity", **kwargs)

    def plot_optimal_strategy(self, strategy_variable, **kwargs):
        # We want to be able to plot the effects of one strategy variable, or maybe later
        # one parameter, on the optimal strategy (all other strategy variable).
        # todo delete or finish this
        bounds = self.cforager.get_strategy_bounds(strategy_variable)
        n_points = kwargs.get("n_points", 10)

    def _plot_single_strategy(self, ax, strategy, response_fn, *response_fn_args, **kwargs):
        def y(strategy, x):
            self.cforager.set_strategy(strategy, x)
            return response_fn(*response_fn_args)
        current_strategy = self.cforager.get_strategy(strategy)
        current_response = response_fn(*response_fn_args)
        bounds = self.cforager.get_strategy_bounds(strategy)
        plot_x = np.linspace(bounds[0], bounds[1], 50)
        plot_y = np.array([y(strategy, x) for x in plot_x])
        ax.plot(plot_x, plot_y)
        ax.plot([current_strategy], [current_response], marker='o', markersize=5, color='r')
        ax.set_xlabel(strategy)
        if 'ylabel' in kwargs:
            ax.set_ylabel(kwargs['ylabel'])
        else:
            ax.set_ylabel(self.response_labels[response_fn])
        if response_fn == self.cforager.detection_probability:
            ax.set_ylim(-0.03, 1.03)
        if response_fn == self.cforager.tau:    # to avoid scaling away the detail at asymptotes
            ax.set_ylim(0, 10)
        self.cforager.set_strategy(strategy, current_strategy)  # put it back when done

    def plot_strategies(self, response_fn, *response_fn_args, **kwargs):
        fig = plt.figure(figsize=(9, 9))
        gs = gridspec.GridSpec(2, 2)
        ax1, ax2, ax3, ax4 = [fig.add_subplot(ss) for ss in gs]
        self._plot_single_strategy(ax1, vf.Forager.Strategy.sigma_A, response_fn, *response_fn_args)
        self._plot_single_strategy(ax2, vf.Forager.Strategy.mean_column_velocity, response_fn, *response_fn_args)
        self._plot_single_strategy(ax3, vf.Forager.Strategy.inspection_time, response_fn, *response_fn_args)
        self._plot_single_strategy(ax4, vf.Forager.Strategy.discrimination_threshold, response_fn, *response_fn_args)
        #self._plot_single_strategy(ax5, vf.Forager.Strategy.search_image, response_fn, *response_fn_args)
        if 'title' in kwargs: plt.suptitle(kwargs['title'], fontsize=15, fontweight='bold')
        gs.tight_layout(fig, rect=[0, 0, 1.0, 0.95])
        self.finish_figure("Strategies", **kwargs)

    def _plot_single_parameter(self, ax, parameter, response_fn, *response_fn_args, **kwargs):
        def y(parameter, x):
            self.cforager.set_parameter(parameter, x)
            return response_fn(*response_fn_args)
        current_parameter = self.cforager.get_parameter(parameter)
        current_response = response_fn(*response_fn_args)
        bounds = self.cforager.get_parameter_bounds(parameter)
        plot_x = np.linspace(bounds[0], bounds[1], 50)
        plot_y = np.array([y(parameter, x) for x in plot_x])
        ax.plot(plot_x, plot_y)
        ax.plot([current_parameter], [current_response], marker='o', markersize=5, color='r')
        ax.set_xlabel(parameter)
        if 'ylabel' in kwargs:
            ax.set_ylabel(kwargs['ylabel'])
        else:
            ax.set_ylabel(self.response_labels[response_fn])
        if response_fn == self.cforager.detection_probability:
            ax.set_ylim(0, 1)
        self.cforager.set_parameter(parameter, current_parameter)  # put it back when done

    def plot_parameters(self, response_fn, *response_fn_args, **kwargs):
        fig = plt.figure(figsize=(12, 15))
        gs = gridspec.GridSpec(3, 4)
        ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12 = [fig.add_subplot(ss) for ss in gs]
        self._plot_single_parameter(ax1, vf.Forager.Parameter.flicker_frequency, response_fn, *response_fn_args)
        self._plot_single_parameter(ax2, vf.Forager.Parameter.tau_0, response_fn, *response_fn_args)
        self._plot_single_parameter(ax3, vf.Forager.Parameter.delta_0, response_fn, *response_fn_args)
        self._plot_single_parameter(ax4, vf.Forager.Parameter.beta, response_fn, *response_fn_args)
        self._plot_single_parameter(ax5, vf.Forager.Parameter.A_0, response_fn, *response_fn_args)
        self._plot_single_parameter(ax6, vf.Forager.Parameter.nu_0, response_fn, *response_fn_args)
        self._plot_single_parameter(ax7, vf.Forager.Parameter.discriminability, response_fn, *response_fn_args)
        self._plot_single_parameter(ax8, vf.Forager.Parameter.delta_p, response_fn, *response_fn_args)
        self._plot_single_parameter(ax9, vf.Forager.Parameter.omega_p, response_fn, *response_fn_args)
        self._plot_single_parameter(ax10, vf.Forager.Parameter.ti_p, response_fn, *response_fn_args)
        self._plot_single_parameter(ax11, vf.Forager.Parameter.sigma_p_0, response_fn, *response_fn_args)
        if 'title' in kwargs: plt.suptitle(kwargs['title'], fontsize=15, fontweight='bold')
        gs.tight_layout(fig, rect=[0, 0, 1.0, 0.95])
        self.finish_figure("Parameters", **kwargs)

    def plot_discrimination_model(self, **kwargs):
        # Plot the relative size of each component of perceptual sigma
        pt = kwargs.get('pt', self.cforager.get_favorite_prey_type())
        x = kwargs.get('x', 0.5 * pt.get_max_visible_distance())
        z = kwargs.get('z', 0.5 * pt.get_max_visible_distance())
        T = self.cforager.passage_time(x, z, pt)
        plot_times = np.linspace(0.01, T - 0.01, 30)
        components_list = [self.cforager.perceptual_variance_components(t, x, z, pt) for t in plot_times]
        df = DataFrame(components_list)
        df.set_index('t')
        components_to_plot = [item for item in components_list[0].keys() if
                              item not in ['t', 'y'] and not (df[item] == 1).all()]
        fig = plt.figure(figsize=(9, 15))
        gs1 = gridspec.GridSpec(3, 1)
        # Create subplot showing multipliers on perceptual sigma, on a log scale.
        ax_components = fig.add_subplot(gs1[2])
        ax_components.axhline(color='k', ls='dotted')
        legend_handles = []
        palette = hls_palette(len(components_to_plot), l=.5, s=0.8)
        position = df['y']
        for i, component in enumerate(components_to_plot):
            response = np.log10(df[component])
            handle = ax_components.plot(position, response, color=palette[i], label=component)
            legend_handles.append(handle[0])
        ax_components.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                             ncol=4, fancybox=True, shadow=True)
        ax_components.set_title("{0}: Perc. var. effects for {1}".format(self.label, pt.get_name()))
        ax_components.set_xlabel("Y-coordinate (in m, from upstream to downstream)")
        ax_components.set_ylabel("Log multiplier on perceptual variance (0 = no effect)")
        ax_components.invert_xaxis()  # upstream to the left, downstream to the right
        gs1.tight_layout(fig, rect=[0, 0.1, 1, 1])
        # Create subplot showing value of perceptual variance itself
        ax_pv = fig.add_subplot(gs1[1])
        response = [self.cforager.perceptual_variance(self.cforager.time_at_y(y_coord, x, z, pt), x, z, pt) for y_coord
                    in df['y']]
        ax_pv.plot(position, response, color=(0, 0, 0))
        ax_pv.grid()
        ax_pv.set_title("{0}: Perc. var. for {1} at (x,z)=({2:.2f}, {3:.2f})".format(self.label, pt.get_name(), x, z))
        ax_pv.set_xlabel("Y-coordinate (in m, from upstream to downstream)")
        ax_pv.set_ylabel("Perceptual variance")
        ax_pv.invert_xaxis()  # upstream to the left, downstream to the right
        gs1.tight_layout(fig, rect=[0, 0.05, 1, 1])
        # Plot the corresponding discrimination probabilities
        ax_prob = fig.add_subplot(gs1[0])
        probabilities = [
            self.cforager.discrimination_probabilities(self.cforager.time_at_y(y_coord, x, z, pt), x, z, pt) for y_coord
            in df['y']]
        response_fp = [item[0] for item in probabilities]
        response_th = [item[1] for item in probabilities]
        prob_fp = ax_prob.plot(position, response_fp, color=(0.6, 0, 0), label='False Positive')
        prob_th = ax_prob.plot(position, response_th, color=(0, 0.6, 0), label='True Hit')
        ax_prob.grid()
        ax_prob.set_title(
            "{0}: Disc. probs. for {1} at (x,z)=({2:.2f}, {3:.2f})".format(self.label, pt.get_name(), x, z))
        ax_prob.set_xlabel("Y-coordinate (in m, from upstream to downstream)")
        ax_prob.set_ylabel("Discrimination probability")
        ax_prob.legend(handles=[prob_fp[0], prob_th[0]], loc='upper center', bbox_to_anchor=(0.5, 0.12),
                       ncol=2, fancybox=True, shadow=True)
        ax_prob.invert_xaxis()  # upstream to the left, downstream to the right
        ax_prob.set_ylim(-0.03, 1.03)
        gs1.tight_layout(fig, rect=[0, 0.05, 1, 1])
        self.finish_figure("Discrimination Model", **kwargs)

    def plot_detection_model(self, **kwargs):
        # Plot the relative size of each component of tau
        pt = kwargs.get('pt', self.cforager.get_favorite_prey_type())
        x = kwargs.get('x', 0.5 * pt.get_max_visible_distance())
        z = kwargs.get('z', 0.5 * pt.get_max_visible_distance())
        T = self.cforager.passage_time(x, z, pt)
        plot_times = np.linspace(0.01, T - 0.01, 30)
        components_list = [self.cforager.tau_components(t, x, z, pt) for t in plot_times]
        df = DataFrame(components_list)
        df.set_index('t')
        components_to_plot = [item for item in components_list[0].keys() if item not in
                              ['t', 'y', 'flicker_frequency'] and not (df[item] == 1).all()]
        fig = plt.figure(figsize=(18, 12))
        gs1 = gridspec.GridSpec(2, 2)
        # Create subplot showing multipliers on tau, on a log scale.
        ax_components = fig.add_subplot(gs1[3])
        ax_components.axhline(color='k', ls='dotted')
        legend_handles = []
        palette = hls_palette(len(components_to_plot), l=.5, s=0.8)
        position = df['y']
        for i, component in enumerate(components_to_plot):
            response = np.log10(df[component])
            handle = ax_components.plot(position, response, color=palette[i], label=component)
            legend_handles.append(handle[0])
        ax_components.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                             ncol=4, fancybox=True, shadow=True)
        ax_components.set_title("{0}: Tau effects for {1}".format(self.label, pt.get_name()))
        ax_components.set_xlabel("Y-coordinate (in m, from upstream to downstream)")
        ax_components.set_ylabel("Log multiplier on tau (0 = no effect)")
        ax_components.invert_xaxis()  # upstream to the left, downstream to the right
        # Create subplot showing value of tau itself
        ax_tau = fig.add_subplot(gs1[1])
        response = [self.cforager.tau(self.cforager.time_at_y(y_coord, x, z, pt), x, z, pt) for y_coord in df['y']]
        ax_tau.semilogy(position, response, color=(0, 0, 0))
        ax_tau.grid()
        ax_tau.set_title("{0}: Tau for {1} at (x,z)=({2:.2f}, {3:.2f})".format(self.label, pt.get_name(), x, z))
        ax_tau.set_xlabel("Y-coordinate (in m, from upstream to downstream)")
        ax_tau.set_ylabel("Tau (log-scaled)")
        ax_tau.invert_xaxis()  # upstream to the left, downstream to the right
        # Create subplot showing detection PDF
        ax_pdf = fig.add_subplot(gs1[2])
        probability = [self.cforager.detection_pdf_at_y(y_coord, x, z, pt) for y_coord in df['y']]
        upstream_profitable_bound_t, downstream_profitable_bound_t = self.cforager.bounds_of_profitability(x, z, pt)
        upstream_profitable_bound_y = self.cforager.y_at_time(upstream_profitable_bound_t, x, z, pt)
        downstream_profitable_bound_y = self.cforager.y_at_time(downstream_profitable_bound_t, x, z, pt)
        ax_pdf.axvline(x=upstream_profitable_bound_y, linestyle='dotted', color='g', alpha=1.0)
        ax_pdf.axvline(x=downstream_profitable_bound_y, linestyle='dotted', color='g', alpha=1.0)
        ax_pdf.plot(position, probability, color=(0, 0, 0.6))
        ax_pdf.grid()
        ax_pdf.set_title("{0}: Det. PDF for {1} at (x,z)=({2:.2f}, {3:.2f})".format(self.label, pt.get_name(), x, z))
        ax_pdf.set_xlabel("Y-coordinate (in m, from upstream to downstream)")
        ax_pdf.set_ylabel("Detection PDF")
        ax_pdf.invert_xaxis()  # upstream to the left, downstream to the right
        # Create subplot showing detection CDF
        ax_cdf = fig.add_subplot(gs1[0])
        probability = [self.cforager.detection_cdf_at_y(y_coord, x, z, pt) for y_coord in df['y']]
        ax_cdf.plot(position, probability, color=(0, 0, 0))
        ax_cdf.grid()
        ax_cdf.set_title("{0}: Det. CDF for {1} at (x,z)=({2:.2f}, {3:.2f})".format(self.label, pt.get_name(), x, z))
        ax_cdf.set_xlabel("Y-coordinate (in m, from upstream to downstream)")
        ax_cdf.set_ylabel("Detection CDF")
        ax_cdf.invert_xaxis()  # upstream to the left, downstream to the right
        ax_cdf.set_ylim(0, 1)
        # Show the plot
        gs1.tight_layout(fig, rect=[0, 0.05, 1, 1])
        self.finish_figure("Detection Model", **kwargs)

    def plot_detection_probabilities_rear_view(self, **kwargs):
        set_style('white')
        resolution = 50
        plot_pts = [pt for pt in self.cforager.get_prey_types() if pt.get_prey_drift_concentration() > 0]
        plot_pts = sorted(plot_pts, key=lambda pt: pt.get_length())
        nrows = int(np.ceil(len(plot_pts) / 3))
        fig = plt.figure(figsize=(9, 3 * nrows), facecolor='w', dpi=300)
        gs = gridspec.GridSpec(nrows, 3)
        r = self.cforager.get_max_radius()
        x = np.linspace(-r, r, resolution)
        z = np.linspace(self.fielddata['bottom_z_m'], self.fielddata['surface_z_m'], resolution)
        xg, zg = np.meshgrid(x, z)
        for i, pt in enumerate(plot_pts):
            if self.cforager.detection_probability(0, 0, pt) > 0:
                ax = fig.add_subplot(gs[i])
                yg = np.empty(np.shape(xg))
                for i in range(len(x)):
                    for j in range(len(z)):
                        if x[i] ** 2 + z[j] ** 2 > r ** 2:
                            yg[j, i] = np.nan
                        else:
                            yg[j, i] = self.cforager.detection_probability(x[i], z[j], pt)
                cf = ax.contourf(xg, zg, yg, 100, cmap='viridis')
                for c in cf.collections: c.set_edgecolor("face")  # prevent annoying contour lines between color shades
                ax.set_aspect('equal', 'datalim')
                fig.colorbar(cf, ax=ax, shrink=0.9)
                plt.title(pt.get_name())
                plt.xlabel('x (m)')
                plt.ylabel('z (m)')
        self.finish_figure("Detection Probabilities Map (Rear View)", **kwargs)

    def plot_bounds_of_profitability(self, **kwargs):
        efig = plt.figure(figsize=(4.65, 2.6), dpi=300)
        ax = plt.axes()
        ax.set_aspect('equal', anchor='W')  # roundabout way to make space for the legend since subplots_adjust is broken
        plot_pts = [pt for pt in self.cforager.get_prey_types() if pt.get_prey_drift_concentration() > 0]
        plot_pts = sorted(plot_pts, key=lambda pt: pt.get_length())
        palette = hls_palette(len(plot_pts), l=.3, s=.8)
        legend_handles = []
        ax.plot([0], [0], 'o', color='k', markersize=2)
        z = 0.0
        for i, pt in enumerate(plot_pts):
            max_visible_distance = pt.get_max_visible_distance()
            circle = plt.Circle((0, 0), max_visible_distance, edgecolor=palette[i], fill=False, linewidth=0.3,
                                alpha=0.3)
            ax.add_artist(circle)
            xes = np.linspace(-max_visible_distance, max_visible_distance, 300)
            bounds = [self.cforager.bounds_of_profitability_y(x, z, pt) for x in xes]
            upper_bounds, lower_bounds = np.array(bounds).T
            handle = ax.plot(xes, upper_bounds, color=palette[i], linewidth=0.5, label=pt.get_name())
            ax.plot(xes, lower_bounds, color=palette[i], linewidth=0.5)
            legend_handles.append(handle[0])
        max_radius = 1.05 * max([pt.get_max_visible_distance() for pt in plot_pts])
        ax.set_xbound(lower=-max_radius, upper=max_radius)
        ax.set_ybound(lower=-max_radius, upper=max_radius)
        ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.title(' ')  # Makes room for suptitle; another stupid workaround for subplots_adjust failing
        plt.suptitle('Bounds of profitability')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.tight_layout()
        self.finish_figure("Bounds of Profitability", **kwargs)

    def plot_effects_of_sigma_A(self, t, x, z, pc, **kwargs):
        fig = plt.figure(figsize=(12, 9))
        gs = gridspec.GridSpec(2, 2)
        ax1, ax2, ax3, ax4 = [fig.add_subplot(ss) for ss in gs]
        def detection_pdf(t, x, z, pc):
            tau = self.cforager.tau(t, x, z, pc)
            return np.exp(-t/tau) / tau
        self._plot_single_strategy(ax1, vf.Forager.Strategy.sigma_A, self.cforager.tau, t, x, z, pc)
        self._plot_single_strategy(ax2, vf.Forager.Strategy.sigma_A, self.cforager.detection_probability, x, z, pc)
        self._plot_single_strategy(ax3, vf.Forager.Strategy.sigma_A, detection_pdf, t, x, z, pc, ylabel='Detection PDF')
        self._plot_single_strategy(ax4, vf.Forager.Strategy.sigma_A, self.cforager.NREI)
        plt.suptitle("Effects of sigma_A at (x,z)=({0:.2f},{1:.2f}) for {2}".format(x, z, pc.get_name()), fontsize=15, fontweight='bold')
        gs.tight_layout(fig, rect=[0, 0, 1.0, 0.95])
        self.finish_figure("Effects of sigma_A", **kwargs)

    def plot_variable_reports(self, **kwargs):
        # When reported quantities here require x, z, t, or a prey type, they either use values provided as kwargs
        # or default to x = half-way to edge of search volume, z same, t = half-way through search volume, and the prey
        # category being the one to which the most attention is devoted.
        fav_pt = self.cforager.get_favorite_prey_type()
        pts = self.cforager.get_prey_types()
        smallest_pt = pts[0]
        for pt in pts:
            if pt.get_length() < smallest_pt.get_length():
                smallest_pt = pt
        fav_radius = fav_pt.get_max_visible_distance()
        theta = self.cforager.get_field_of_view()
        rho = fav_radius * np.sin(theta/2) if theta < np.pi else fav_radius
        test_x = rho/2 if 'x' not in kwargs.keys() else kwargs['x']
        test_z = rho/2 if 'z' not in kwargs.keys() else kwargs['z']
        T = self.cforager.passage_time(test_x, test_z, fav_pt)
        test_t = T / 2 if 't' not in kwargs.keys() else kwargs['t']
        if 'pt' in kwargs.keys():
            test_pt = kwargs['pt']
        else:
            test_pt = fav_pt
        self.plot_strategies(self.cforager.NREI, title="NREI vs strategy variables", **kwargs)
        self.plot_parameters(self.cforager.NREI, title="NREI vs parameters", **kwargs)
        self.plot_strategies(self.cforager.tau, test_t, test_x, test_z, test_pt, title="Tau vs strategy at (x,z)=({0:.2f},{1:.2f}) for {2}".format(test_x, test_z, test_pt.get_name()), **kwargs)
        self.plot_parameters(self.cforager.tau, test_t, test_x, test_z, test_pt, title="Tau vs params at (x,z)=({0:.2f},{1:.2f}) for {2}".format(test_x, test_z, test_pt.get_name()), **kwargs)
        self.plot_strategies(self.cforager.detection_probability, test_x, test_z, test_pt, title="Det prob vs strategy at (x,z)=({0:.2f},{1:.2f}) for {2}".format(test_x, test_z, test_pt.get_name()), **kwargs)
        self.plot_parameters(self.cforager.detection_probability, test_x, test_z, test_pt, title="Det prob vs params at (x,z)=({0:.2f},{1:.2f}) for {2}".format(test_x, test_z, test_pt.get_name()), **kwargs)
        self.plot_effects_of_sigma_A(test_t, test_x, test_z, test_pt, **kwargs)
        self.plot_detection_probabilities_vs_velocity(**kwargs)

    def plot_predicted_depletion_field_2D(self, **kwargs):
        # Do a 2-D plot fixed at z=0 for easier visualization unless specified, of the pattern of
        # drift depletion throughout a fish's volume and behind it, in terms of energy (J) available
        # from all prey classes combined.
        plot_z = 0 if 'z' not in kwargs.keys() else kwargs['z']
        numpts = 50 if 'numpts' not in kwargs.keys() else kwargs['numpts']
        available_pts = [pt for pt in self.cforager.get_prey_types() if pt.get_prey_drift_concentration() > 0]
        r = max([pt.get_max_visible_distance() for pt in available_pts])
        x = np.linspace(-r, r, numpts)
        y = np.linspace(-r, r, numpts)
        xg, yg = np.meshgrid(x, y)
        zg = np.empty(np.shape(xg))
        for i in range(len(x)):
            for j in range(len(y)):
                if x[i] ** 2 + y[j] ** 2 > r ** 2:
                    zg[j, i] = np.nan
                else:
                    zg[j, i] = self.cforager.depleted_prey_concentration_total_energy(x[i], y[j], plot_z)
        set_style('white')
        efig, (ax) = plt.subplots(1, 1, facecolor='w', figsize=(3.25, 2.6), dpi=300)
        cf = ax.contourf(xg, yg, zg, 100, cmap='viridis_r')
        for c in cf.collections: c.set_edgecolor("face")  # prevent annoying contour lines between color shades
        efig.colorbar(cf, ax=ax, shrink=0.9)
        plt.title('Energy in drift after depletion (J/m$^3$)')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.tight_layout()
        self.finish_figure("Depletion Field 2D Top View", **kwargs)

    def plot_predicted_detection_field_2D(self, **kwargs):
        # Do a 2-D plot fixed at z=0 for easier visualization unless specified, of the pattern of
        # drift depletion throughout a fish's volume and behind it, in terms of energy (J) available
        # from all prey classes combined.
        plot_z = 0 if 'z' not in kwargs.keys() else kwargs['z']
        numpts = 200 if 'numpts' not in kwargs.keys() else kwargs['numpts']
        available_pts = [pt for pt in self.cforager.get_prey_types() if pt.get_prey_drift_concentration() > 0]
        r = max([pt.get_max_visible_distance() for pt in available_pts])
        x = np.linspace(-r, r, numpts)
        y = np.linspace(-r, r, numpts)
        xg, yg = np.meshgrid(x, y)
        zg = np.empty(np.shape(xg))
        for i in range(len(x)):
            for j in range(len(y)):
                if x[i] ** 2 + y[j] ** 2 > r ** 2:
                    zg[j, i] = np.nan
                else:
                    zg[j, i] = self.cforager.relative_pursuits_by_position(x[i], y[j], plot_z)
        set_style('white')
        efig, (ax) = plt.subplots(1, 1, facecolor='w', figsize=(3.25, 2.6), dpi=300)
        plt.axhline(color='0.5', linewidth=0.1)
        plt.axvline(color='0.5', linewidth=0.1)
        cf = ax.contourf(xg, yg, zg, 100, cmap='viridis')
        for c in cf.collections: c.set_edgecolor("face")  # prevent annoying contour lines between color shades
        efig.colorbar(cf, ax=ax, shrink=0.9)
        plt.title('Relative pursuits (top view at z={0})'.format(plot_z))
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.tight_layout()
        if kwargs.get('show_fielddata', True):
            (px, py, pz) = np.transpose(np.asarray(self.field_detection_positions))
            plt.scatter(px, py, s=0.1, c='k')
        self.finish_figure("Detection Field 2D Top View", **kwargs)

    def plot_detection_probabilities_vs_velocity(self, **kwargs):
        # We want to be able to plot the effects of one strategy variable, or maybe later
        # one parameter, on the optimal strategy (all other strategy variable).
        previous_velocity = self.cforager.get_strategy(vf.Forager.Strategy.mean_column_velocity)
        bounds = self.cforager.get_strategy_bounds(vf.Forager.Strategy.mean_column_velocity)
        plot_x = np.linspace(bounds[0], bounds[1], kwargs.get("n_points", 30))
        fig = plt.figure(figsize=(5, 5))
        ax = plt.axes()
        plot_pts = [pt for pt in self.cforager.get_prey_types() if pt.get_prey_drift_concentration() > 0]
        plot_pts = sorted(plot_pts, key=lambda pt: pt.get_length())
        legend_handles = []
        for pt in plot_pts:
            if pt.get_prey_drift_concentration() > 0:
                radius = pt.get_max_visible_distance()
                theta = self.cforager.get_field_of_view()
                rho = radius * np.sin(theta / 2) if theta < np.pi else radius
                test_x = rho / 2 if 'x' not in kwargs.keys() else kwargs['x']
                test_z = rho / 2 if 'z' not in kwargs.keys() else kwargs['z']
                previous_det_prob = self.cforager.detection_probability(test_x, test_z, pt)
                def y(x):
                    self.cforager.set_strategy(vf.Forager.Strategy.mean_column_velocity, x)
                    return self.cforager.detection_probability(test_x, test_z, pt)
                plot_y = np.array([y(x) for x in plot_x])
                handle = ax.plot(plot_x, plot_y, label=pt.get_name())
                legend_handles.append(handle[0])
                ax.plot([previous_velocity], [previous_det_prob], marker='o', markersize=5, color='r')
        self.cforager.set_strategy(vf.Forager.Strategy.mean_column_velocity, previous_velocity)
        ax.legend(handles=legend_handles, loc=1)
        ax.set_xlabel("Mean column velocity (m/s)")
        ax.set_ylabel("Detection probability @ 1/2 radius")
        plt.tight_layout()
        self.finish_figure("Detection Probabilities vs Velocity", **kwargs)

    # todo: plot detection probability vs debris

    def plot_diet_proportions(self, **kwargs):
        self.cforager.analyze_results()  # required for calculating diet proportion
        observed_diet = []
        predicted_diet = []
        labels = []
        diet_obj_count = 0
        diet_obj_total = 0
        dietdata = [item for item in self.fielddata['diet_by_category'].values() if item['number'] is not None]
        for dd in sorted(dietdata, key=lambda x: x['number']):
            pt =self.cforager.get_prey_type(dd['name'])
            labels.append(pt.get_name())
            predicted = self.cforager.get_diet_proportion_for_prey_type(pt)
            observed = dd['diet_proportion']
            if predicted > 0 or observed > 0:
                diet_obj_count += 1
                diet_obj_total += (predicted - observed) ** 2
            observed_diet.append(observed)
            predicted_diet.append(predicted)
        diet_rmse = np.sqrt(diet_obj_total / diet_obj_count)
        fig = plt.figure(figsize=(6, 4))
        gs = gridspec.GridSpec(1, 1)
        ax1, = [fig.add_subplot(ss) for ss in gs]
        pred_handle = ax1.barh(np.arange(1, 1 + len(predicted_diet)), predicted_diet, align='center', height=0.8,
                               tick_label=labels, label="Predicted")
        obs_handle = ax1.barh(np.arange(1, 1 + len(predicted_diet)), -np.array(observed_diet), align='center',
                              height=0.8, tick_label=labels, label="Observed")
        ax1.legend(handles=[pred_handle,
                            obs_handle])  # , loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, fancybox=True, shadow=True)
        ax1.set_xlim(-1, 1)
        ax1.axvline(0, color='k', linewidth=1)
        plt.xticks([-1, 0, 1], [1, 0, 1])
        plt.figtext(0.78, 0.72, "rmse={0:.4f}".format(diet_rmse))
        ax1.set_title("Diet proportions")
        gs.tight_layout(fig, rect=[0, 0, 1.0, 1.0])
        self.finish_figure("Diet Proportions", **kwargs)

    def map_discrimination_model(self, **kwargs):
        plot_z = kwargs.get('z', 0)
        numpts = kwargs.get('numpts', 200)
        plot_pts = [pt for pt in self.cforager.get_prey_types() if pt.get_prey_drift_concentration() > 0]
        plot_pts = sorted(plot_pts, key=lambda pt: pt.get_length())
        max_radius = max([pt.get_max_visible_distance() for pt in plot_pts])  # for plotting
        r = max_radius
        ncolumns = 7
        gs = gridspec.GridSpec(len(plot_pts), ncolumns)
        efig = plt.figure(figsize=(4 * ncolumns, 4 * len(plot_pts)), facecolor='w', dpi=300)
        for k, pt in enumerate(plot_pts):
            max_visible_distance = pt.get_max_visible_distance()
            x = np.linspace(-r, r, numpts)
            y = np.linspace(-r, r, numpts)
            xg, yg = np.meshgrid(x, y)
            zg_aa = np.empty(np.shape(xg))  # to store angular area results
            zg_av = np.empty(np.shape(xg))  # to store angular velocity results
            zg_pv = np.empty(np.shape(xg))  # to store perceptual variance results
            zg_fp = np.empty(np.shape(xg))  # to store false positive results
            zg_th = np.empty(np.shape(xg))  # to store true hit results
            zg_en = np.empty(np.shape(xg))  # to store item energetic profitability by position
            for i in range(len(x)):
                for j in range(len(y)):
                    if x[i] ** 2 + y[j] ** 2 + plot_z ** 2 >= max_visible_distance ** 2:
                        zg_aa[j, i] = np.nan
                        zg_av[j, i] = np.nan
                        zg_pv[j, i] = np.nan
                        zg_fp[j, i] = np.nan
                        zg_th[j, i] = np.nan
                        zg_en[j, i] = np.nan
                    else:
                        t_at_y = self.cforager.time_at_y(y[j], x[i], plot_z, pt)
                        pv_components = self.cforager.perceptual_variance_components(t_at_y, x[i], plot_z, pt)
                        zg_aa[j, i] = np.log10(pv_components['angular_area'])
                        zg_av[j, i] = np.log10(pv_components['angular_velocity'])
                        zg_pv[j, i] = self.cforager.perceptual_variance(t_at_y, x[i], plot_z, pt)
                        fp_prob, th_prob = self.cforager.discrimination_probabilities(t_at_y, x[i], plot_z, pt)
                        if fp_prob < 0.6:
                            print("With radius {0}, fp_prob={1:.5f} at R={2} with pv={3:.5f}".format(
                                pt.get_max_visible_distance(), fp_prob, np.sqrt(x[i] ** 2 + y[j] ** 2 + plot_z ** 2),
                                zg_pv[j, i]))
                        zg_fp[j, i] = fp_prob
                        zg_th[j, i] = th_prob
                        maneuver_v = (self.cforager.water_velocity(plot_z) + self.cforager.get_focal_velocity()) / 2;
                        profitability = self.cforager.item_profitability_at_time(t_at_y, x[i], y[j], plot_z, maneuver_v,
                                                                                 pt)
                        zg_en[j, i] = profitability if profitability > -90 else np.nan  # exclude impossible maneuvers
            ax_aa = efig.add_subplot(gs[ncolumns * k])
            ax_av = efig.add_subplot(gs[ncolumns * k + 1])
            ax_pv = efig.add_subplot(gs[ncolumns * k + 2])
            ax_fp = efig.add_subplot(gs[ncolumns * k + 3])
            ax_th = efig.add_subplot(gs[ncolumns * k + 4])
            ax_en = efig.add_subplot(gs[ncolumns * k + 5])
            ax_text = efig.add_subplot(gs[ncolumns * k + 6])
            cf_aa = ax_aa.contourf(xg, yg, zg_aa, 100, cmap='viridis')
            cf_av = ax_av.contourf(xg, yg, zg_av, 100, cmap='viridis')
            cf_pv = ax_pv.contourf(xg, yg, zg_pv, 100, cmap='viridis')
            cf_fp = ax_fp.contourf(xg, yg, zg_fp, 100, cmap='viridis_r')
            cf_th = ax_th.contourf(xg, yg, zg_th, 100, cmap='viridis_r')
            cf_en = ax_en.contourf(xg, yg, zg_en, 100, cmap='viridis')
            for cf in (cf_aa, cf_av, cf_pv, cf_fp, cf_th, cf_en):
                for c in cf.collections:
                    c.set_edgecolor("face")  # prevent annoying contour lines between color shades
            efig.colorbar(cf_aa, ax=ax_aa, shrink=0.8)
            efig.colorbar(cf_av, ax=ax_av, shrink=0.8)
            efig.colorbar(cf_pv, ax=ax_pv, shrink=0.8)
            efig.colorbar(cf_fp, ax=ax_fp, shrink=0.8)
            efig.colorbar(cf_th, ax=ax_th, shrink=0.8)
            efig.colorbar(cf_en, ax=ax_en, shrink=0.8)
            ax_aa.set_title('Log10(Ang. area on P.V.)')
            ax_av.set_title('Log10(Ang. velocity on P.V.)')
            ax_pv.set_title('Perceptual variance')
            ax_fp.set_title('P(false positive)')
            ax_th.set_title('P(true hit)')
            ax_en.set_title('Expected energy/item (J)')
            ax_text.text(0.0, 0.5, "{0}\n{1:.2f} J/item".format(pt.get_name(), pt.get_energy_content()), fontsize=10)
            ax_text.axis('off')
            for ax in (ax_aa, ax_av, ax_pv, ax_fp, ax_th, ax_en):
                ax.axhline(color='0.5', linewidth=0.1)
                ax.axvline(color='0.5', linewidth=0.1)
                ax.plot([0], [0], 'o', color='k', markersize=2)
                ax.set_xlabel('x (m)')
                ax.set_ylabel('y (m)')
                ax.set_aspect('equal')
            # Now add bounds of profitability on the energy plot
            xes = np.linspace(-max_visible_distance, max_visible_distance, 300)
            bounds = [self.cforager.bounds_of_profitability_y(line_x, plot_z, pt) for line_x in xes]
            upper_bounds, lower_bounds = np.array(bounds).T
            ax_en.plot(xes, upper_bounds, color='k', linewidth=0.5)
            ax_en.plot(xes, lower_bounds, color='k', linewidth=0.5)
        plt.suptitle("Discrimination probabilities and consequent expected energy per item at z={0}".format(plot_z))
        self.finish_figure("Discrimination Probabilities Maps", **kwargs)

    def map_tau_effects(self, **kwargs):
        plot_z = kwargs.get('z', 0)
        numpts = kwargs.get('numpts', 200)
        plot_pts = [pt for pt in self.cforager.get_prey_types() if pt.get_prey_drift_concentration() > 0]
        plot_pts = sorted(plot_pts, key=lambda pt: pt.get_length())
        max_radius = max([pt.get_max_visible_distance() for pt in plot_pts])  # for plotting
        r = max_radius
        # not plotted: 'angular_length_too_small_to_see',  'crypticity', 'excluded_by_search_image', 'flicker_frequency'
        # 'search_image', 'set_size' (because it's spatially constant -- maybe print it), 't', 'tau_0', 'y'
        # probably print effects from set_size, tau_0
        gs = gridspec.GridSpec(len(plot_pts), 5)
        efig = plt.figure(figsize=(20, 4 * len(plot_pts)), facecolor='w', dpi=300)
        for k, pt in enumerate(plot_pts):
            x = np.linspace(-r, r, numpts)
            y = np.linspace(-r, r, numpts)
            xg, yg = np.meshgrid(x, y)
            zg_sa = np.empty(np.shape(xg))  # to store spatial attention results
            zg_aa = np.empty(np.shape(xg))  # to store angular area results
            zg_loom = np.empty(np.shape(xg))  # to store loom results
            zg_tau = np.empty(np.shape(xg))  # to store overall tau values
            for i in range(len(x)):
                for j in range(len(y)):
                    if x[i] ** 2 + y[j] ** 2 + plot_z ** 2 >= pt.get_max_visible_distance() ** 2:
                        zg_sa[j, i] = np.nan
                        zg_aa[j, i] = np.nan
                        zg_loom[j, i] = np.nan
                        zg_tau[j, i] = np.nan
                    else:
                        t_at_y = self.cforager.time_at_y(y[j], x[i], plot_z, pt)
                        components = self.cforager.tau_components(t_at_y, x[i], plot_z, pt)
                        zg_sa[j, i] = np.log10(components['spatial_attention'])
                        zg_aa[j, i] = np.log10(components['angular_area'])
                        zg_loom[j, i] = np.log10(components['loom'])
                        zg_tau[j, i] = np.log10(self.cforager.tau(t_at_y, x[i], plot_z, pt))
            ax_sa = efig.add_subplot(gs[5 * k])
            ax_aa = efig.add_subplot(gs[5 * k + 1])
            ax_loom = efig.add_subplot(gs[5 * k + 2])
            ax_tau = efig.add_subplot(gs[5 * k + 3])
            ax_text = efig.add_subplot(gs[5 * k + 4])
            cf_sa = ax_sa.contourf(xg, yg, zg_sa, 100, cmap='viridis_r')
            cf_aa = ax_aa.contourf(xg, yg, zg_aa, 100, cmap='viridis_r')
            cf_loom = ax_loom.contourf(xg, yg, zg_loom, 100, cmap='viridis_r')
            cf_tau = ax_tau.contourf(xg, yg, zg_tau, 100, cmap='viridis_r')
            for cf in (cf_sa, cf_aa, cf_loom, cf_tau):
                for c in cf.collections:
                    c.set_edgecolor("face")  # prevent annoying contour lines between color shades
            efig.colorbar(cf_sa, ax=ax_sa, shrink=0.8)
            efig.colorbar(cf_aa, ax=ax_aa, shrink=0.8)
            efig.colorbar(cf_loom, ax=ax_loom, shrink=0.8)
            efig.colorbar(cf_tau, ax=ax_tau, shrink=0.8)
            ax_text.text(0.0, 0.5, "{0}\n{1:.2f} J/item".format(pt.get_name(), pt.get_energy_content()), fontsize=10)
            ax_text.axis('off')
            ax_sa.set_title('Log10(Spatial attention)')
            ax_aa.set_title('Log10(Angular area)')
            ax_loom.set_title('Log10(Loom)')
            ax_tau.set_title('Log10(Tau)')
            for ax in (ax_sa, ax_aa, ax_loom, ax_tau):
                ax.axhline(color='0.5', linewidth=0.1)
                ax.axvline(color='0.5', linewidth=0.1)
                ax.plot([0], [0], 'o', color='k', markersize=2)
                ax.set_xlabel('x (m)')
                ax.set_ylabel('y (m)')
                ax.set_aspect('equal')
        plt.suptitle("Spatial effects on tau at z={0}".format(plot_z))
        self.finish_figure("Tau Effects Maps", **kwargs)

    def finish_figure(self, name, **kwargs):
        if 'figure_folder' in kwargs:
            plt.savefig(os.path.join(kwargs['figure_folder'], "{0}.pdf".format(name)))
        if kwargs.get('show', True):
            plt.show()
        else:
            plt.close()