from mayavi import mlab
import Fish3D
import numpy as np
import scipy
import math
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib import gridspec
from seaborn import hls_palette
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

    def plot_predicted_detection_field(self, gridsize=50j, stopButton=False, colorMax=None, **kwargs):
        def func(x, y, z):
            return self.cforager.relative_pursuits_by_position(0.01 * x, 0.01 * y, 0.01 * z)

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

        def my_otf_value(v):
            return (v / 4) ** 1.8

        flat_s = s.flatten()
        nonzero_s = flat_s[flat_s > 0]
        otf = PiecewiseFunction()
        otf.add_point(0.0, 0.0)
        for v in np.linspace(0.05, 1.0, 40):
            otf_val = my_otf_value(v)
            otf.add_point(rel_pursuits_value_percentile(v), otf_val)
        # otf.add_point(0.0, 0.0)
        # otf.add_point(rel_pursuits_value_percentile(0.05), 0.006)
        # otf.add_point(rel_pursuits_value_percentile(0.15), 0.008)
        # otf.add_point(rel_pursuits_value_percentile(0.25), 0.013)
        # otf.add_point(rel_pursuits_value_percentile(0.35), 0.022)
        # otf.add_point(rel_pursuits_value_percentile(0.50), 0.03)
        # otf.add_point(rel_pursuits_value_percentile(0.65), 0.04)
        # otf.add_point(rel_pursuits_value_percentile(0.80), 0.05)
        # otf.add_point(rel_pursuits_value_percentile(0.90), 0.10)
        vol._otf = otf
        vol._volume_property.set_scalar_opacity(otf)

        (px, py, pz) = 100 * np.transpose(np.asarray(self.fielddata['detection_positions']))
        point_radius = 0.005 * self.fork_length_cm
        d = np.repeat(2 * point_radius, px.size)  # creates an array of point diameters
        mlab.points3d(py, -px, pz, d, color=kwargs.get('pointcolor', self.color), scale_factor=8, resolution=12,
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

    def plot_water_velocity(self):
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
        plt.show()

    def _roc_curve_data(self, pc):
        perceptual_sigma = pc.get_perceptual_sigma()
        def normal_cdf(x):
            return 0.5 * (1 + math.erf(x / math.sqrt(2)))
        def p_false_positive(disc_thresh):
            return 1.0 - normal_cdf(disc_thresh / perceptual_sigma)
        def p_true_hit(disc_thresh):
            return 1.0 - normal_cdf((disc_thresh - self.cforager.get_parameter(vf.Forager.Parameter.discriminability)) / perceptual_sigma)
        linsp = np.linspace(-10, 10, 300)
        logsp = np.logspace(-1, 10, 50)
        thresholds = np.sort(np.concatenate([np.flip(-logsp, axis=0), logsp, linsp], axis=0))
        probabilities = [(p_false_positive(thresh), p_true_hit(thresh)) for thresh in thresholds]
        pfps, pths = zip(*probabilities)  # unzip the list
        return list(pfps), list(pths)

    def plot_roc_curves(self):
        fig = plt.figure(figsize=(5, 5))
        ax = plt.axes()
        legend_handles = []
        pcs = self.cforager.get_prey_types()
        palette = hls_palette(len(pcs), l=.3, s=.8)
        for i, pc in enumerate(pcs):
            x, y = self._roc_curve_data(pc)
            dot_x = pc.get_false_positive_probability()
            dot_y = pc.get_true_hit_probability()
            handle = ax.plot(x, y, color=palette[i], label=pc.get_name())
            legend_handles.append(handle[0])
            ax.plot([dot_x], [dot_y], marker='o', markersize=5, color=palette[i])
        ax.legend(handles=legend_handles, loc=4)
        ax.set_xbound(lower=-0.02, upper=1.02)
        ax.set_ybound(lower=-0.02, upper=1.02)
        ax.set_aspect('equal', 'datalim')
        ax.set_xlabel("False positive probability")
        ax.set_ylabel("True hit probability")
        plt.tight_layout()
        plt.show()

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
        plt.show()

    def plot_optimal_strategy(self, strategy_variable, **kwargs):
        # We want to be able to plot the effects of one strategy variable, or maybe later
        # one parameter, on the optimal strategy (all other strategy variable).
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
        # if response_fn == self.cforager.detection_probability: ax.set_ylim(0, 1)
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
        plt.show()

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
        # if response_fn == self.cforager.detection_probability: ax.set_ylim(0, 1)
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
        plt.show()

    def plot_discrimination_model(self, **kwargs):
        # Plot the relative size of each component of tau
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
        # Create subplot showing multipliers on tau, on a log scale.
        ax_components = fig.add_subplot(gs1[2])
        ax_components.axhline(color='k', ls='dotted')
        legend_handles = []
        palette = hls_palette(len(components_to_plot), l=.5, s=0.8)
        position = df['y']
        for i, component in enumerate(components_to_plot):
            response = df[component]
            handle = ax_components.plot(position, response, color=palette[i], label=component)
            legend_handles.append(handle[0])
        ax_components.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                             ncol=4, fancybox=True, shadow=True)
        ax_components.set_title("{0}: Perc. var. effects for {1}".format(self.label, pt.get_name()))
        ax_components.set_xlabel("Y-coordinate (in m, from upstream to downstream)")
        ax_components.set_ylabel("Multiplier on perceptual variance (0 = no effect)")
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
        ax_prob.set_ylim(0, 1)
        gs1.tight_layout(fig, rect=[0, 0.05, 1, 1])
        # Show the plot
        plt.show()

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
        ax_components.set_ylabel("Multiplier on tau (1 = no effect)")
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
        plt.show()

    def plot_tau_by_class(self, x, z):
        # using x/z defined by half the max viewing distance for the smallest size class
        # and time bounds defined by the passthrough time for the max sized prey, plot
        # tau for each prey type. but be aware that t=0 for each prey type is different
        # with respect to tau, and maybe offset to account for that?
        pass

    def plot_detection_probability_for_class(self, pc):
        # radial rear-view map of detection probability
        pass

    def plot_effects_of_sigma_A(self, t, x, z, pc):
        fig = plt.figure(figsize=(12, 9))
        gs = gridspec.GridSpec(3, 2)
        ax1, ax2, ax3, ax4, ax5, ax6 = [fig.add_subplot(ss) for ss in gs]
        def detection_pdf(t, x, z, pc):
            tau = self.cforager.tau(t, x, z, pc)
            return np.exp(-t/tau) / tau
        def search_rate_multiplier_on_tau():
            return 1 + self.cforager.get_search_rate() / self.cforager.get_parameter(vf.Forager.Parameter.Z_0)
        self._plot_single_strategy(ax1, vf.Forager.Strategy.sigma_A, self.cforager.tau, t, x, z, pc)
        self._plot_single_strategy(ax2, vf.Forager.Strategy.sigma_A, self.cforager.passage_time, x, z, pc)
        self._plot_single_strategy(ax3, vf.Forager.Strategy.sigma_A, self.cforager.detection_probability, x, z, pc)
        self._plot_single_strategy(ax4, vf.Forager.Strategy.sigma_A, detection_pdf, t, x, z, pc, ylabel='Detection PDF')
        self._plot_single_strategy(ax5, vf.Forager.Strategy.sigma_A, self.cforager.NREI)
        plt.suptitle("Effects of sigma_A at (x,z)=({0:.2f},{1:.2f}) for {2}".format(x, z, pc.get_name()), fontsize=15, fontweight='bold')
        gs.tight_layout(fig, rect=[0, 0, 1.0, 0.95])
        plt.show()

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
        self.plot_strategies(self.cforager.NREI, title="NREI vs strategy variables")
        self.plot_parameters(self.cforager.NREI, title="NREI vs parameters")
        self.plot_strategies(self.cforager.tau, test_t, test_x, test_z, test_pt, title="Tau vs strategy at (x,z)=({0:.2f},{1:.2f}) for {2}".format(test_x, test_z, test_pt.get_name()))
        self.plot_parameters(self.cforager.tau, test_t, test_x, test_z, test_pt, title="Tau vs params at (x,z)=({0:.2f},{1:.2f}) for {2}".format(test_x, test_z, test_pt.get_name()))
        self.plot_strategies(self.cforager.detection_probability, test_x, test_z, test_pt, title="Det prob vs strategy at (x,z)=({0:.2f},{1:.2f}) for {2}".format(test_x, test_z, test_pt.get_name()))
        self.plot_parameters(self.cforager.detection_probability, test_x, test_z, test_pt, title="Det prob vs params at (x,z)=({0:.2f},{1:.2f}) for {2}".format(test_x, test_z, test_pt.get_name()))
        self.plot_effects_of_sigma_A(test_t, test_x, test_z, test_pt)

    def plot_depletion(self, **kwargs):
        # Do a plot, probably 2-D fixed at Z=0 for easier visualization, of the pattern of
        # drift depletion throughout a fish's volume and behind it. This should be the concentration
        # of prey at those (x,z) coordinates times the probability that the fish hasn't
        # captured the item yet at that point.
        pass

    # also todo: plot detection probability vs velocity, detection probability vs debris, by prey type