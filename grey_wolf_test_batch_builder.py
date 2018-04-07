import os
import sys
import re

batch_reps = 20  # Number of times to check each grey wolf parameter set for each fish

fish_labels = [ '2015-06-10-1 Chena - Chinook Salmon (id #1)',
                '2015-06-16-2 Panguingue - Dolly Varden (id #2)',
                '2015-07-10-1 Chena - Chinook Salmon (id #4)',
                '2015-07-15-1 Panguingue - Dolly Varden (id #1)',
                '2015-07-16-2 Panguingue - Dolly Varden (id #5)',
                '2015-07-31-1 Clearwater - Arctic Grayling (id #1)',
                '2015-08-05-2 Chena - Chinook Salmon (id #4)',
                '2015-08-13-1 Clearwater - Arctic Grayling (id #3)',
                '2016-06-03-1 Chena - Chinook Salmon (id #1)',
                '2016-06-10-1A Clearwater - Arctic Grayling (id #2)',
                '2016-06-10-2 Clearwater - Arctic Grayling (id #1)',
                '2016-06-18-1 Panguingue - Dolly Varden (id #1)',
                '2016-08-02-1 Clearwater - Arctic Grayling (id #1)',
                '2016-08-07-2 Panguingue - Dolly Varden (id #1)',
                '2016-08-14-2 Chena - Chinook Salmon (id #1)',
               ]

for fish_label in fish_labels:
    batch_name = re.sub(r'\W+', '', fish_label)  # strip out non-alphanumeric characters from fish label for job name
    batch_file_contents = """
    #PBS -S /bin/bash
    #PBS -N {1}
    #PBS -q batch
    #PBS -l nodes=1:ppn=28:Intel
    #PBS -l walltime=48:00:00
    #PBS -l mem=60gb
    #PBS -M jasonneuswanger@gmail.com
    #PBS -m ae
    
    cd $PBS_O_WORKDIR
    
    echo
    echo "Job ID: $PBS_JOBID"
    echo "Queue:  $PBS_QUEUE"
    echo "Cores:  $PBS_NP"
    echo "Nodes:  $(cat $PBS_NODEFILE | sort -u | tr '\n' ' ')"
    echo
    
    python grey_wolf_test.py {0} '{2}'
    """.format(batch_reps, batch_name, fish_label)

    with open('{0}.sh'.format('batches/' + batch_name), 'w') as batch_file:
        batch_file.write(batch_file_contents)

    os.system("qsub 'batches/{0}.sh'".format(batch_name))