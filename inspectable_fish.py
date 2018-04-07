from mayavi import mlab
import Fish3D
import numpy as np

import field_test_fish as ftf

class InspectableFish(ftf.FieldTestFish):

    def plot_predicted_detection_field(self):
        def func(x, y, z):
            return self.cforager.relative_pursuits_by_position(0.01 * x, 0.01 * y, 0.01 * z)
        vfunc = np.vectorize(func)
        r = 1.05*100*self.cforager.get_radius()
        gridsize = 30j
        x, y, z = np.mgrid[-r:r:gridsize, -r:r:gridsize, -r:r:gridsize]
        s = vfunc(x, y, z)
        figname = "3D Fish Test Figure"
        myFig = mlab.figure(figure=figname, bgcolor=(0, 0, 0), size=(1024, 768))
        mlab.clf(myFig)
        head_position = np.array((0, 0, 0))
        tail_position = np.array((0, -self.fork_length_cm, 0))
        Fish3D.fish3D(head_position, tail_position, self.species, myFig, color=tuple(abs(np.random.rand(3))), world_vertical=np.array([0, 0, 1]))
        mlab.pipeline.volume(mlab.pipeline.scalar_field(x, y, z, s), vmin=0.0, vmax=np.max(s))
        (px, py, pz) = 100 * np.transpose(np.asarray(self.fielddata['detection_positions']))
        point_radius = 0.005 * self.fork_length_cm
        d = np.repeat(2 * point_radius, px.size)  # creates an array of point diameters
        mlab.points3d(py, -px, pz, d, color=self.color, scale_factor=8, resolution=12, opacity=1.0, figure=myFig)
        mlab.show()
        return myFig

    # def roc_curve_data(self, pc):
    #     def p_false_positive(disc_thresh):
    #         return 1.0 - mg.normal_cdf(disc_thresh / self.perceptual_sigma(pc))
    #     def p_true_hit(disc_thresh):
    #         return 1.0 - mg.normal_cdf((disc_thresh - self.environment.lambda_c) / self.perceptual_sigma(pc))
    #     linsp = np.linspace(-10, 10, 100)
    #     logsp = np.logspace(-1, 10, 50)
    #     thresholds = np.sort(np.concatenate([np.flip(-logsp, axis=0), logsp, linsp], axis=0))
    #     probabilities = [(p_false_positive(thresh), p_true_hit(thresh)) for thresh in thresholds]
    #     pfps, pths = zip(*probabilities)  # unzip the list
    #     return list(pfps), list(pths)
    #
    # def plot_roc_curves(self):
    #     fig = plt.figure(figsize=(5, 5))
    #     ax = plt.axes()
    #     legend_handles = []
    #     for pc in self.environment.prey_categories:
    #         x, y = self.roc_curve_data(pc)
    #         dot_x = self.false_positive_probability(pc)
    #         dot_y = self.true_hit_probability(pc)
    #         color = np.random.rand(3,)
    #         handle = ax.plot(x, y, color=color, label=pc.description)
    #         legend_handles.append(handle[0])
    #         ax.plot([dot_x], [dot_y], marker='o', markersize=5, color=color)
    #     ax.legend(handles=legend_handles, loc=4)
    #     ax.set_xbound(lower=-0.02, upper=1.02)
    #     ax.set_ybound(lower=-0.02, upper=1.02)
    #     ax.set_aspect('equal', 'datalim')
    #     ax.set_xlabel("False positive probability")
    #     ax.set_ylabel("True hit probability")
    #     plt.tight_layout()


# VIDEO_EXPORT_PATH = "/Users/Jason/Dropbox/Drift Model Project/Presentations/2018 New Zealand/"
# fps = 60
# revolution_time = 15
# revolutions = 2
# import moviepy.video.VideoClip as vc
# import os
#
# def make_frame(t):
#     current_view = mlab.view(figure=fig)
#     mlab.view(azimuth=90 + (t / revolution_time) * 360, elevation=current_view[1], distance=current_view[2], focalpoint=current_view[3], figure=fig)
#     return mlab.screenshot(antialiased=True, figure=fig)
#
# rotating_movie = vc.VideoClip(make_frame, duration=revolution_time * revolutions - 1 / fps)
# rotating_movie.write_videofile(VIDEO_EXPORT_PATH + test_fish['fielddata']['label'] + ".mp4", fps=fps, codec='h264', bitrate='8000k', audio=False)
# mlab.close()
# os.system('open "%s"' % VIDEO_EXPORT_PATH)
