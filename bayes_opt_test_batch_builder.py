import os
import re

fish_labels = ( '2015-06-10-1 Chena - Chinook Salmon (id #1)',
                '2015-08-05-2 Chena - Chinook Salmon (id #4)',
                '2016-06-18-1 Panguingue - Dolly Varden (id #1)',
                '2016-08-07-2 Panguingue - Dolly Varden (id #1)',
                '2015-07-31-1 Clearwater - Arctic Grayling (id #1)',
                '2016-06-10-2 Clearwater - Arctic Grayling (id #1)'
                )

methods = ('ucb', 'ei', 'mixed')

scalings = ('linear', 'log')

for fish_label in fish_labels:
    for method in methods:
        for scaling in scalings:
            batch_name = re.sub(r'\W+', '', fish_label) + '_' + method + '_' + scaling  # strip out non-alphanumeric characters from fish label for job name
            batch_file_contents = """
            #PBS -S /bin/bash
            #PBS -N {2}
            #PBS -q batch
            #PBS -l nodes=1:ppn=28:Intel
            #PBS -l walltime=24:00:00
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
            
            python bayes_opt_test_batch.py '{0}' '{1}' '{2}' '{3}'
            """.format(method, scaling, batch_name, fish_label)

            with open('{0}.sh'.format('batches/' + batch_name), 'w') as batch_file:
                batch_file.write(batch_file_contents)

            os.system("qsub 'batches/{0}.sh'".format(batch_name))