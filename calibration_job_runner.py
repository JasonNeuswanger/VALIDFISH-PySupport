from platform import uname
import sys
from job_runner import JobRunner

IS_MAC = (uname()[0] == 'Darwin')
if IS_MAC:
    cores_per_node = 8
    runner = JobRunner('Fifteen2', cores_per_node)
else:
    cores_per_node = int(sys.argv[1])
    runner = JobRunner(sys.argv[2], cores_per_node)

runner.run_jobs()

