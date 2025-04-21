#!/usr/bin/env python

import json

from tqdm import tqdm

input_file_path = "../data/ltrpo_dataset/original_ltrpo_database.json"

output_file_path = "../data/ltrpo_dataset/modified_ltrpo_data.json"


def read_json_file(file_path):
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return []


input_data = read_json_file(input_file_path)

modified_objects = []

for obj in tqdm(input_data, desc="Processing", unit="obj"):
    obj_copy = obj.copy()

    if obj_copy.get("image_path") and len(obj_copy["image_path"]) > 0:
        first_image_path_parts = obj_copy["image_path"][0].split("/")
        new_gt_bbox_id = ["image_" + "_".join(first_image_path_parts[-3:]).replace(".jpg", "")]

        obj_copy["image_bbox_id"] = obj_copy["gt_bbox_id"]
        obj_copy["gt_bbox_id"] = new_gt_bbox_id

        modified_objects.append(obj_copy)

with open(output_file_path, "w") as file:
    json.dump(modified_objects, file)

print(f"modified data saved to {output_file_path}")
