from platform import uname
import sys
from job_runner import JobRunner

IS_MAC = (uname()[0] == 'Darwin')
if IS_MAC:
    cores_per_node = 8
    runner = JobRunner('Fifteen1', cores_per_node)
else:
    cores_per_node = int(sys.argv[1])
    runner = JobRunner(sys.argv[2], cores_per_node)

runner.run_jobs()
#
# test_fish = runner.fishes[0]
# test_fish.cforager.print_parameters()
# test_fish.cforager.print_strategy()
# test_fish.evaluate_fit()
#
# test_fish.plot_detection_model()
#
# test_fish.cforager.set_parameter(test_fish.cforager.get_parameter_named('alpha_tau'), 1.0) # to reset caches
# testx=0.01
# testy=0.01
# testz=0.02
# pt = test_fish.cforager.get_favorite_prey_type()
# t_y=test_fish.cforager.time_at_y(testy, testx, testz, pt)
# test_fish.cforager.tau_components(t_y, testx, testz, pt)