import os

# CONFIGURATION FOR A GIVEN JOB

N_NODES = 1
CORES_PER_NODE = 14
JOB_NAME = "First Cluster Test"
FISH_GROUP = "calibration_five_of_each"

# COMMON CODE FOR ANY JOB

for i in range(N_NODES):
    batch_name = "{0} Node {1}".format(JOB_NAME, i)
    batch_file_contents = """
    #PBS -S /bin/bash
    #PBS -N {job_name}
    #PBS -q batch
    #PBS -l nodes={nodes}:ppn=28:Intel
    #PBS -l walltime=96:00:00
    #PBS -l mem=30gb
    #PBS -M jasonneuswanger@gmail.com
    #PBS -m ae
    
    cd $PBS_O_WORKDIR
    
    echo
    echo "Job ID: $PBS_JOBID"
    echo "Queue:  $PBS_QUEUE"
    echo "Cores:  $PBS_NP"
    echo "Nodes:  $(cat $PBS_NODEFILE | sort -u | tr '\n' ' ')"
    echo
    
    python calibration_job_runner.py '{job_name}' '{fish_group}'
    """.format(nodes=CORES_PER_NODE, job_name=JOB_NAME, fish_group=FISH_GROUP)

    with open('{0}.sh'.format('batches/' + batch_name), 'w') as batch_file:
        batch_file.write(batch_file_contents)

    os.system("qsub 'batches/{0}.sh'".format(batch_name))