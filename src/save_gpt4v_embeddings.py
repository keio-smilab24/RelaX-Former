import json
import os

import numpy as np
from tqdm import tqdm


def read_jsonl_file(filepath):
    """Reads a JSONL file and returns a list of dictionaries."""
    data = []
    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line))
    return data


def main():
    database = None
    embedding_database = None
    # mp3d
    # with open(
    #     "/home/initial/dev/switching_reverie_retrieval/data/ltrpo_dataset/ltrpo_database.json",
    #     "r",
    # ) as database_file:
    #     database = json.load(database_file)
    #
    # embedding_database = read_jsonl_file(
    #     "/home/initial/dev/switching_reverie_retrieval/data/mp3d_gpt4v_embeddings.jsonl"
    # )

    # for embedding_data in tqdm(embedding_database):
    #     custom_id = embedding_data["custom_id"]
    #     embeddings = embedding_data["response"]["body"]["data"][0]["embedding"]
    #     if not isinstance(embeddings, np.ndarray):
    #         embedding_array = np.array(embeddings)
    #     else:
    #         embedding_array = embeddings
    #
    #     matching_data = [data for data in database if data["image_path"][0] == custom_id]
    #
    #     if len(matching_data) > 1:
    #         for data in matching_data:
    #             save_path = data["image_path"][0].replace("EXTRACTED_IMGS_", "gpt4v_embeddings").replace(".jpg", ".npy")
    #             os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #             np.save(save_path, embedding_array)
    #     elif len(matching_data) == 1:
    #         save_path = (
    #             matching_data[0]["image_path"][0].replace("EXTRACTED_IMGS_", "gpt4v_embeddings").replace(".jpg", ".npy")
    #         )
    #         os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #         np.save(save_path, embedding_array)
    #     else:
    #         print(f"Could not find matching data for {custom_id}")

    # hm3d
    with open(
        "/home/initial/dev/switching_reverie_retrieval/data/hm3d/hm3d_dataset/hm3d_database.json",
        "r",
    ) as database_file:
        database = json.load(database_file)

    embedding_database = read_jsonl_file(
        "/home/initial/dev/switching_reverie_retrieval/data/hm3d/hm3d_gpt4v_embeddings.jsonl"
    )

    for embedding_data in tqdm(embedding_database):
        custom_id = embedding_data["custom_id"]
        embeddings = embedding_data["response"]["body"]["data"][0]["embedding"]
        if not isinstance(embeddings, np.ndarray):
            embedding_array = np.array(embeddings)
        else:
            embedding_array = embeddings

        matching_data = [data for data in database if data["image_path"][0] == custom_id]

        if len(matching_data) > 1:
            for data in matching_data:
                save_path = data["full_image_feature_path"][0].replace("image_features", "gpt4v_embeddings")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                np.save(save_path, embedding_array)
        elif len(matching_data) == 1:
            save_path = matching_data[0]["full_image_feature_path"][0].replace("image_features", "gpt4v_embeddings")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, embedding_array)
        else:
            print(f"Could not find matching data for {custom_id}")


if __name__ == "__main__":
    # pass
    main()
