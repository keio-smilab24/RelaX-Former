"""val unseen, test作成のため，環境ごとに画像数をカウント"""
import json
import os


def main():
    database_path = "data/hm3d/hm3d_dataset/hm3d_database.json"
    database = None
    with open(database_path, "r") as json_file:
        database = json.load(json_file)
    print(len(database))

    # TODO: better: overlapを排除
    env_to_num_images_with_overlap = {}

    for data in database:
        if data["other"]["original_split"] == "val":
            if data["environment"] in env_to_num_images_with_overlap.keys():
                env_to_num_images_with_overlap[data["environment"]] = (
                    env_to_num_images_with_overlap[data["environment"]] + 1
                )
            else:
                env_to_num_images_with_overlap[data["environment"]] = 1

    os.makedirs("data/hm3d/hm3d_dataset", exist_ok=True)
    with open("data/hm3d/hm3d_dataset/hm3d_env_to_num_images_with_overlap.json", "w") as json_file:
        json.dump(env_to_num_images_with_overlap, json_file, indent=4)


# poetry run python src/count_sample_per_env.py
if __name__ == "__main__":
    main()
