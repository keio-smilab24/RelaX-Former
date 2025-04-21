"""データセットの統計情報を表示"""
import json


def main():
    mp3d_path = "data/ltrpo_dataset/ltrpo_database.json"
    hm3d_path = "data/hm3d/hm3d_dataset/hm3d_database.json"

    with open(mp3d_path, "r") as json_file:
        mp3d_database = json.load(json_file)
    with open(hm3d_path, "r") as json_file:
        hm3d_database = json.load(json_file)

    instructions = []
    vocab = set()
    total_word = 0
    image_names = set()
    env_names = set()

    for database in [mp3d_database, hm3d_database]:
        for data in database:
            instructions.append(data["instruction"])
            for image in data["image_path"]:
                image_names.add(image)
            env_names.add(data["environment"])

    for instr in instructions:
        words = instr.replace(".", "").replace("?", "").split()
        for word in words:
            vocab.add(word)
        total_word += len(words)

    # 1つのinstructionに対してtarget/destinationで2つのデータがあることを前提
    sample = len(instructions) // 2
    total_word //= 2

    print(f"sample: {sample}")
    print(f"vocabulary size: {len(vocab)}")
    print(f"total word: {total_word}")
    print(f"average sentence length: {total_word / sample}")
    print(f"image: {len(image_names)}")
    print(f"environment: {len(env_names)}")


# poetry run python src/show_dataset_statistics.py
if __name__ == "__main__":
    main()
