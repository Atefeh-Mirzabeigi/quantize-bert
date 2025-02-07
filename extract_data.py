'''
To extract the required fields from the JSON files and merge them into a single JSON file.
This script reads JSON files from a specified input directory, extracts the fields (default: title, date, bias, and ID), and writes the extracted data to a single output JSON file. 
The extracted data is stored as a list of dictionaries, where each dictionary represents a single JSON entry with the required fields.
Assumption: The files from https://github.com/ramybaly/Article-Bias-Prediction are already downloaded and this script is placed within that folder.
Input: input_dir (str): Path to the directory containing JSON files
        output_file (str): Path to the output JSON file
        fields (list): List of fields to extract from the JSON files
Example: "python preprocessing.py data/jsons/ data/combined_data.json title,date,bias,ID"
'''
import os
import json
import sys

def merge_json_files(input_dir = "data/jsons/", output_file = "./combined_data.json", fields = ['title', 'content', 'date', 'bias', 'topic', 'ID']):
    # Initialize an empty list to store extracted data
    combined_data = []

    # Iterate through each file in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(input_dir, filename)

            # Read JSON data from the file
            with open(file_path, 'r') as file:
                data = json.load(file)
                # Extract required fields from each JSON entry
                
                extracted_data = []
                data_obj = {}
                for field in fields:
                    data_obj[field] = data[field]
                
                extracted_data.append(data_obj)

                # Append extracted data to combined_data list
                combined_data.extend(extracted_data)
                
    # Write the combined extracted data to the output JSON file
    with open(output_file, 'w') as outfile:
        json.dump(combined_data, outfile, indent=4)


if __name__ == '__main__':
    # Define input directory containing JSON files and output file path
    input_directory = "data/jsons/"
    output_directory = "./combined_data.json"
    fields = "title,content,date,bias,topic,ID"
    # Call the function to merge JSON files and extract required columns
    merge_json_files(input_dir = input_directory, output_file = output_directory, fields=fields.split(','))
