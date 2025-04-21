import json

MP3D_DATABASE = "data/ltrpo_dataset/ltrpo_database.json"
HM3D_DATABASE = "data/hm3d/hm3d_dataset/hm3d_database.json"


NEW_MP3D_DATABASE = "data/ltrpo_dataset/new_ltrpo_database.json"
NEW_HM3D_DATABASE = "data/hm3d/hm3d_dataset/new_hm3d_database.json"


def main():
    with open(MP3D_DATABASE, "r") as f:
        mp3d_database = json.load(f)

    with open(HM3D_DATABASE, "r") as f:
        hm3d_database = json.load(f)

    new_mp3d_database = []
    for data in mp3d_database:
        data["gpt4v_embeddings"] = (
            data["image_path"][0].replace("EXTRACTED_IMGS_", "gpt4v_embeddings").replace(".jpg", ".npy")
        )
        new_mp3d_database.append(data)

    new_hm3d_database = []
    for data in hm3d_database:
        data["gpt4v_embeddings"] = data["full_image_feature_path"][0].replace("image_features", "gpt4v_embeddings")
        new_hm3d_database.append(data)

    with open(NEW_MP3D_DATABASE, "w") as f:
        json.dump(new_mp3d_database, f)

    with open(NEW_HM3D_DATABASE, "w") as f:
        json.dump(new_hm3d_database, f)


if __name__ == "__main__":
    main()
