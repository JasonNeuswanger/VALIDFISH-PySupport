import json
import random
import math

# Generate groups

FIELD_DATA_FILE = "/Users/Jason/Dropbox/Drift Model Project/Calculations/VALIDFISH/PySupport/Model_Testing_Data.json"
json_data = json.load(open(FIELD_DATA_FILE, 'r'))

all_fish_labels = sorted(list(json_data.keys()))
chinook_labels = [label for label in all_fish_labels if 'Chinook' in label]
dolly_labels = [label for label in all_fish_labels if 'Dolly' in label]
grayling_labels = [label for label in all_fish_labels if 'Grayling' in label]

calibration_chinook_labels = random.sample(chinook_labels, math.ceil(len(chinook_labels)/2))
calibration_dolly_labels = random.sample(dolly_labels, math.ceil(len(dolly_labels)/2))
calibration_grayling_labels = random.sample(grayling_labels, math.ceil(len(grayling_labels)/2))

testing_chinook_labels = [label for label in chinook_labels if label not in calibration_chinook_labels]
testing_dolly_labels = [label for label in dolly_labels if label not in calibration_dolly_labels]
testing_grayling_labels = [label for label in grayling_labels if label not in calibration_grayling_labels]

calibration_fish_labels = calibration_chinook_labels + calibration_dolly_labels + calibration_grayling_labels
testing_fish_labels = testing_chinook_labels + testing_dolly_labels + testing_grayling_labels

calibration_five_chinook = random.sample(calibration_chinook_labels, 5)
calibration_five_dollies = random.sample(calibration_dolly_labels, 5)
calibration_five_grayling = random.sample(calibration_grayling_labels, 5)
calibration_five_of_each = calibration_five_chinook + calibration_five_dollies + calibration_five_grayling

calibration_one_chinook = random.sample(calibration_chinook_labels, 1)

print("Calibration group for Chinook has {0} fish, with {1} reserved for testing.".format(len(calibration_chinook_labels), len(testing_chinook_labels)))
print("Calibration group for Grayling has {0} fish, with {1} reserved for testing.".format(len(calibration_grayling_labels), len(testing_grayling_labels)))
print("Calibration group for Dollies has {0} fish, with {1} reserved for testing.".format(len(calibration_dolly_labels), len(testing_dolly_labels)))

# Save groups to file

# first element of each dict entry is the fish labels, second is the species (or 'all') contained

fish_groups_dict = {
    'calibration_chinook' : (calibration_chinook_labels, 'all'),
    'calibration_dollies' : (calibration_dolly_labels, 'dollies'),
    'calibration_grayling' : (calibration_grayling_labels, 'grayling'),
    'testing_chinook' : (testing_chinook_labels, 'chinook'),
    'testing_dollies' : (testing_dolly_labels, 'dollies'),
    'testing_grayling' : (testing_grayling_labels, 'grayling'),
    'calibration_fish' : (calibration_fish_labels, 'all'),
    'testing_fish' : (testing_fish_labels, 'all'),
    'calibration_five_chinook' : (calibration_five_chinook, 'chinook'),
    'calibration_five_dollies' : (calibration_five_dollies, 'dollies'),
    'calibration_five_grayling' : (calibration_five_grayling, 'grayling'),
    'calibration_five_of_each' : (calibration_five_of_each, 'all'),
    'calibration_one_chinook' : (calibration_one_chinook, 'chinook')
}

with open("Fish_Groups.json", "w") as f:
    json.dump(fish_groups_dict, f)