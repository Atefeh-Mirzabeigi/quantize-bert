'''
This script extracts article items from 'combined_data.json' and saving to 'filtered_data.json'.
It selects 6000 items from each bias category, ensuring no duplicates. And shuffling before saving.
'''

import json
import random

with open('combined_data.json', 'r') as file:
    data = json.load(file)

# Filter data by bias
bias_0 = [item for item in data if item['bias'] == 0]
bias_1 = [item for item in data if item['bias'] == 1]
bias_2 = [item for item in data if item['bias'] == 2]

# Randomly select 6000 items from each bias, no dublicates
random_selected_bias_0 = random.sample(bias_0, 6000)
random_selected_bias_1 = random.sample(bias_1, 6000)
random_selected_bias_2 = random.sample(bias_2, 6000)

# Combine all items and shuffle
selected_items = random_selected_bias_0 + random_selected_bias_1 + random_selected_bias_2
random.shuffle(selected_items)

with open('filtered_data.json', 'w') as new_file:
    json.dump(selected_items, new_file, indent=4)
