"""環境数をカウント"""
import json


def main():
    # database_path = "data/hm3d/hm3d_dataset/hm3d_database.json"
    database_path = "data/ltrpo_dataset/ltrpo_database.json"
    database = None
    with open(database_path, "r") as json_file:
        database = json.load(json_file)
    # print(len(database))

    env_name = set()

    for data in database:
        env_name.add(data["environment"])

    print(f"len(env_name): {len(env_name)}")


# poetry run python src/count_env.py
if __name__ == "__main__":
    main()
