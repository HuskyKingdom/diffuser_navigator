
import json
import numpy as np

# Assuming the JSON data is stored in a file named 'data.json'
file_path = '../train_gt.json'

# Load the data from the JSON file
with open(file_path, 'r') as file:
    data_from_file = json.load(file)

# Extract all locations from the JSON
all_locations_from_file = []

for key, value in data_from_file.items():
    locations = value['locations']
    all_locations_from_file.extend(locations)

# Convert locations to a numpy array for calculation
locations_np_from_file = np.array(all_locations_from_file)

flattened_locations = locations_np_from_file.flatten()

print(flattened_locations.shape)

# Calculate mean and variance for x, y, z
means_from_file = np.mean(flattened_locations, axis=0)
variances_from_file = np.var(flattened_locations, axis=0)

max_ = np.max(flattened_locations, axis=0)
min_ = np.min(flattened_locations, axis=0)

print(max_, min_)
