import csv
import sys
import field_test_fish as ftf

IS_MAC = (sys.platform == 'darwin')
if IS_MAC:
    CONVERGENCE_RESULTS_FOLDER = '/Users/Jason/Dropbox/Drift Model Project/Calculations/GWO convergence/'
else:
    CONVERGENCE_RESULTS_FOLDER = '/home/alaskajn/results/GWO Convergence/'

# 2015-06-10-1 Chena - Chinook Salmon (id #1)
# 2015-06-11-1 Chena - Chinook Salmon (id #1)
# 2015-06-11-2 Chena - Chinook Salmon (id #1)
# 2015-06-12-1 Chena - Chinook Salmon (id #1)
# 2015-06-16-1 Panguingue - Dolly Varden (id #1)
# 2015-06-16-2 Panguingue - Dolly Varden (id #1)
# 2015-06-16-2 Panguingue - Dolly Varden (id #2)
# 2015-06-17-1 Panguingue - Dolly Varden (id #1)
# 2015-06-17-1 Panguingue - Dolly Varden (id #4)
# 2015-06-18-1 Panguingue - Dolly Varden (id #1)
# 2015-06-23-1 Clearwater - Arctic Grayling (id #1)
# 2015-06-23-2 Clearwater - Arctic Grayling (id #1)
# 2015-07-10-1 Chena - Chinook Salmon (id #4)
# 2015-07-11-1 Chena - Chinook Salmon (id #4)
# 2015-07-12-2 Chena - Chinook Salmon (id #4)
# 2015-07-15-1 Panguingue - Dolly Varden (id #1)
# 2015-07-15-1 Panguingue - Dolly Varden (id #2)
# 2015-07-16-2 Panguingue - Dolly Varden (id #4)
# 2015-07-16-2 Panguingue - Dolly Varden (id #5)
# 2015-07-17-1 Panguingue - Dolly Varden (id #1)
# 2015-07-17-2 Panguingue - Dolly Varden (id #1)
# 2015-07-17-3 Panguingue - Dolly Varden (id #4)
# 2015-07-28-1 Clearwater - Arctic Grayling (id #1)
# 2015-07-29-1 Clearwater - Arctic Grayling (id #3)
# 2015-07-31-1 Clearwater - Arctic Grayling (id #1)
# 2015-07-31-1 Clearwater - Arctic Grayling (id #2)
# 2015-08-05-1 Chena - Chinook Salmon (id #4)
# 2015-08-05-2 Chena - Chinook Salmon (id #4)
# 2015-08-06-1 Chena - Chinook Salmon (id #3)
# 2015-08-12-1 Clearwater - Arctic Grayling (id #1)
# 2015-08-13-1 Clearwater - Arctic Grayling (id #3)
# 2015-09-04-1 Clearwater - Arctic Grayling (id #1)
# 2015-09-04-1 Clearwater - Arctic Grayling (id #2)
# 2016-06-01-1 Chena - Chinook Salmon (id #1)
# 2016-06-02-1 Chena - Chinook Salmon (id #1)
# 2016-06-02-2 Chena - Chinook Salmon (id #1)
# 2016-06-02-2 Chena - Chinook Salmon (id #5)
# 2016-06-03-1 Chena - Chinook Salmon (id #1)
# 2016-06-03-1 Chena - Chinook Salmon (id #2)
# 2016-06-09-1 Clearwater - Arctic Grayling (id #1)
# 2016-06-10-1A Clearwater - Arctic Grayling (id #1)
# 2016-06-10-1A Clearwater - Arctic Grayling (id #2)
# 2016-06-10-1B Clearwater - Arctic Grayling (id #3)
# 2016-06-10-2 Clearwater - Arctic Grayling (id #1)
# 2016-06-11-1 Clearwater - Arctic Grayling (id #1)
# 2016-06-16-1 Panguingue - Dolly Varden (id #1)
# 2016-06-17-1 Panguingue - Dolly Varden (id #1)
# 2016-06-17-1 Panguingue - Dolly Varden (id #2)
# 2016-06-17-1 Panguingue - Dolly Varden (id #3)
# 2016-06-17-2 Panguingue - Dolly Varden (id #1)
# 2016-06-17-2 Panguingue - Dolly Varden (id #3)
# 2016-06-17-3 Panguingue - Dolly Varden (id #1)
# 2016-06-17-3 Panguingue - Dolly Varden (id #2)
# 2016-06-18-1 Panguingue - Dolly Varden (id #1)
# 2016-06-19-1 Panguingue - Dolly Varden (id #1)
# 2016-06-19-1 Panguingue - Dolly Varden (id #2)
# 2016-07-07-1 Clearwater - Arctic Grayling (id #1)
# 2016-07-07-1 Clearwater - Arctic Grayling (id #5)
# 2016-07-08-1 Clearwater - Arctic Grayling (id #1)
# 2016-07-08-2 Clearwater - Arctic Grayling (id #1)
# 2016-07-09-1 Clearwater - Arctic Grayling (id #1)
# 2016-07-09-1 Clearwater - Arctic Grayling (id #5)
# 2016-07-09-2 Clearwater - Arctic Grayling (id #1)
# 2016-08-01-1 Clearwater - Arctic Grayling (id #1)
# 2016-08-02-1 Clearwater - Arctic Grayling (id #1)
# 2016-08-02-2 Clearwater - Arctic Grayling (id #1)
# 2016-08-02-2 Clearwater - Arctic Grayling (id #2)
# 2016-08-07-1 Panguingue - Dolly Varden (id #1)
# 2016-08-07-1 Panguingue - Dolly Varden (id #2)
# 2016-08-07-2 Panguingue - Dolly Varden (id #1)
# 2016-08-08-1 Panguingue - Dolly Varden (id #1)
# 2016-08-08-2 Panguingue - Dolly Varden (id #1)
# 2016-08-08-3 Panguingue - Dolly Varden (id #1)
# 2016-08-08-3 Panguingue - Dolly Varden (id #2)
# 2016-08-08-4 Panguingue - Dolly Varden (id #1)
# 2016-08-12-1 Chena - Chinook Salmon (id #1)
# 2016-08-13-1 Chena - Chinook Salmon (id #1)
# 2016-08-13-2 Chena - Chinook Salmon (id #1)
# 2016-08-14-1 Chena - Chinook Salmon (id #1)
# 2016-08-14-1 Chena - Chinook Salmon (id #2)
# 2016-08-14-2 Chena - Chinook Salmon (id #1)

candidate_strategies = [  # never want the last two to be true at the same time
    # (False, True, False, False, False, True),
    # (False, True, False, False, False, False),
    # (False, False, True, False, False, False),
    (False, True, True, False, False, True),
    (False, True, True, False, False, False),
    (False, False, False, False, False, True)
]

def test_algorithm(fish_label, nreps=1):
    niters = 1000
    pack_size = 26
    forager = ftf.FieldTestFish(fish_label)
    with open(CONVERGENCE_RESULTS_FOLDER + 'GWO convergence ' + fish_label + '.csv', 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for rep in range(nreps):
            for strategy in candidate_strategies:
                use_chaos, use_dynamic_C, use_exponential_decay, use_levy, use_only_alpha, use_weighted_alpha = strategy
                key = (pack_size, use_chaos, use_dynamic_C, use_exponential_decay, use_levy, use_only_alpha, use_weighted_alpha)
                fitnesses = forager.optimize(niters, pack_size, True, use_chaos, use_dynamic_C, use_exponential_decay, use_levy, use_only_alpha, use_weighted_alpha)
                writer.writerow([rep] + list(key) + fitnesses)
                csvfile.flush()
                print("Completed calculation ", rep+1, " of ", nreps, " for key ", key, " for fish ", fish_label)

# fish_label = '2016-07-09-1 Clearwater - Arctic Grayling (id #5)'
# fish_label = '2016-08-12-1 Chena - Chinook Salmon (id #1)' # -- BAD FISH, lots of infinite NREIs
# fish_label = '2016-08-14-2 Chena - Chinook Salmon (id #1)'
# fish_label = '2015-07-10-1 Chena - Chinook Salmon (id #4)'
# fish_label = '2015-07-15-1 Panguingue - Dolly Varden (id #1)'

# for i in range(10):
#     fish_label = '2016-08-14-2 Chena - Chinook Salmon (id #1)'
#     test_algorithm(1)
#     fish_label = '2015-07-10-1 Chena - Chinook Salmon (id #4)'
#     test_algorithm(1)
#     fish_label = '2015-07-15-1 Panguingue - Dolly Varden (id #1)'
#     test_algorithm(1)
#     fish_label = '2016-06-03-1 Chena - Chinook Salmon (id #1)'
#     test_algorithm(1)
#     fish_label = '2016-08-07-2 Panguingue - Dolly Varden (id #1)'
#     test_algorithm(1)
#     fish_label = '2015-07-31-1 Clearwater - Arctic Grayling (id #1)'
#     test_algorithm(1)

nreps = int(sys.argv[1])
fish_label_arg = sys.argv[2]
test_algorithm(fish_label_arg, nreps)

# USAGE:
# On Mac, do 'source activate driftmodelenv' in terminal if not done already.
# Then python grey_wolf_test.py 2 '2016-08-12-1 Chena - Chinook Salmon (id #1)')
