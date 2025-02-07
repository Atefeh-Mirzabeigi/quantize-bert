import pandas as pd

data = pd.read_json('filtered_data.json')

data['bias'] = data['bias'].astype(int)

bias_counts = data['bias'].value_counts()

print("no. articles with bias 0:", bias_counts.get(0, 0))
print("no. articles with bias 1:", bias_counts.get(1, 0))
print("no. articles with bias 2:", bias_counts.get(2, 0))