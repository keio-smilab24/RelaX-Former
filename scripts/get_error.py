import json


def load_jsonl(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
    # Parse each line as a separate JSON array
    json_objects = [json.loads(line) for line in lines]
    return json_objects


def find_matching_objects(instructions, data):
    matching_objects = []
    for instruction in instructions:
        for obj in data:
            if instruction["instruction"] == obj["mode"] + " " + obj["instruction"]:
                # Add rank to the matched object
                obj["rank"] = instruction["rank"]
                matching_objects.append(obj)
    return matching_objects


def save_json(data, file_path):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)


def main():
    input_file = "../log/id1770_mp3d_error.json"
    # data_file = "../data/hm3d/hm3d_dataset/hm3d_database.json"
    data_file = "../data/ltrpo_dataset/ltrpo_database.json"
    output_file = "output.json"

    # Load instructions from input.jsonl
    instruction_lines = load_jsonl(input_file)

    instructions = [
        {"instruction": item["instruction"], "rank": item["mrr_values"][0]}
        for sublist in instruction_lines
        for item in sublist
    ]

    # Load data from data.json
    with open(data_file, "r") as file:
        data = json.load(file)

    # Find matching objects in data.json
    matching_objects = find_matching_objects(instructions, data)

    # Save the matched objects to output.json
    save_json(matching_objects, output_file)

    print(f"Matching objects have been saved to {output_file}")


if __name__ == "__main__":
    main()
