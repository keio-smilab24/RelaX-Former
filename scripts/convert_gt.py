import json
from collections import defaultdict

# Paths to your input and output files
input_file_path = "../data/hm3d/hm3d_dataset/hm3d_database.json"
output_file_path = "../data/hm3d/hm3d_dataset/expanded_hm3d_database.json"


def read_json_file(file_path):
    """Reads a JSON file and returns its content."""
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError as e:
        print(f"Error reading the file: {e}")
        return []


def group_by_mode(data):
    """
    Groups unique 'gt_bbox_id' values by either 'instruction_chatgpt' or 'llm_data.destination'
    based on the 'mode' of the entry, excluding error items with a key of 'NA'.
    """
    groups = defaultdict(set)
    for entry in data:
        gt_bbox_id = entry["gt_bbox_id"][0]  # Assuming there's a 'gt_bbox_id' field in each entry
        mode = entry["mode"]
        key = None

        # Determine the grouping key based on the mode
        if mode == "<target>":
            key = entry["instruction_chatgpt"]
        elif mode == "<destination>":
            key = entry["llm_data"]["destination"]

        # Skip error items with a key of 'NA'
        if key == "NA":
            continue

        if key:  # Only proceed if a valid key was determined
            # Extract environment from 'gt_bbox_id' and include it in the grouping key
            environment = gt_bbox_id.split("_")[1]
            # Use a tuple of key and environment as the composite key for grouping
            groups[(key, environment)].add(gt_bbox_id)

    return groups


def update_gt_bbox_id(data, groups):
    """
    Updates the 'gt_bbox_id' field for each entry with unique values based on the groups formed,
    excluding error items with a key of 'NA'.
    """
    for entry in data:
        gt_bbox_id = entry["gt_bbox_id"][0]
        environment = gt_bbox_id.split("_")[1]
        mode = entry["mode"]
        key = None

        # Determine the grouping key based on the mode, as before
        if mode == "<target>":
            key = entry["instruction_chatgpt"]
        elif mode == "<destination>":
            key = entry["llm_data"]["destination"]

        # Skip error items with a key of 'NA'
        if key == "NA":
            continue

        if key:  # Only proceed if a valid key was determined
            # Update 'gt_bbox_id' with unique values from the group
            entry["gt_bbox_id"] = list(groups[(key, environment)])

    return data


def write_json_file(file_path, data):
    """Writes the provided data to a JSON file."""
    try:
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)
    except Exception as e:
        print(f"Error writing to file: {e}")


# Read the input file
data = read_json_file(input_file_path)

# Group 'gt_bbox_id' values by the specified key and environment, excluding error items
groups = group_by_mode(data)

# Update the 'gt_bbox_id' field for each entry with unique values based on the groups, excluding error items
updated_data = update_gt_bbox_id(data, groups)

# Write the modified data to the output file
write_json_file(output_file_path, updated_data)

print("Data extraction, conditional grouping excluding error items, and file writing completed successfully.")
