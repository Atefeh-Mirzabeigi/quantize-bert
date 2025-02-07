import json
import os
from tqdm import tqdm

def load_seen_ids(filtered_file):
    """ Load the filtered data file and extract all seen IDs. """
    with open(filtered_file, 'r') as file:
        filtered_data = json.load(file)
    seen_ids = {item['ID'] for item in filtered_data}
    return seen_ids

def find_unseen_data(json_folder, seen_ids):
    """ Traverse the json files and collect unseen items with specific fields. """
    unseen_data = []
    files = [f for f in os.listdir(json_folder) if f.endswith('.json')]
    for filename in tqdm(files, desc="Processing JSON files"):
        filepath = os.path.join(json_folder, filename)
        with open(filepath, 'r') as file:
            try:
                data = json.load(file)
                # Check if the ID in the current file is in the set of seen IDs
                if data['ID'] not in seen_ids:
                    # Collect only the required fields
                    filtered_item = {
                        "title": data["title"],
                        "content": data["content"],
                        "date": data["date"],
                        "bias": data["bias"],
                        "topic": data["topic"],
                        "ID": data["ID"]
                    }
                    unseen_data.append(filtered_item)
            except json.JSONDecodeError:
                print(f"Error decoding JSON from file: {filename}")
    return unseen_data


def save_unseen_data(unseen_data, output_file):
    """ Save the unseen data to a new JSON file. """
    with open(output_file, 'w') as file:
        json.dump(unseen_data, file, indent=4)

def main():
    filtered_file = 'filtered_data.json'
    json_folder = 'data/jsons'
    output_file = 'unseen_data.json'

    # Load seen IDs from filtered data
    seen_ids = load_seen_ids(filtered_file)

    # Find unseen data in the folder
    unseen_data = find_unseen_data(json_folder, seen_ids)

    # Save the unseen data to a new file
    save_unseen_data(unseen_data, output_file)
    print(f"Unseen data saved to {output_file}")

if __name__ == "__main__":
    main()
