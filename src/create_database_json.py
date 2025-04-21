"""hm3d_database.jsonを作成"""
import json
import os


def create_per_version(version, split="train"):
    print(version, split)
    assert version in ["ver.3", "ver.4"]
    file_path = "data/hm3d/csv/hm3d-carry_part1_clean.csv" if version == "ver.3" else "data/hm3d/csv/hm3d-carry_part2_clean.csv"
    header = None
    data = []
    with open(file_path, "r", encoding="utf-16") as file:
        print(f"load: {file_path}")
        # 各行を読み込む
        for idx, line in enumerate(file):
            # タブで分割
            columns = line.strip().split("\t")
            if idx == 0:
                # CASE	SERIAL	REF	QUESTNNR	MODE	STARTED	BX01_01	BX30_01	BX29_01	BX28_01	BX27_01	BX26_01	BX25_01	BX24_01	BX23_01	BX22_01	BX21_01	BX20_01	BX19_01	BX18_01	BX17_01	BX16_01	BX15_01	BX14_01	BX13_01	BX12_01	BX11_01	BX10_01	BX09_01	BX08_01	BX07_01	BX06_01	BX05_01	BX04_01	BX03_01	BX02_01	IV01_01	IV01_02	IV01_03	IV01_04	IV01_05	IV01_06	IV01_07	IV01_08	IV01_09	IV01_10	IV01_11	IV01_12	IV01_13	IV01_14	IV01_15	IV01_16	IV01_17	IV01_18	IV01_19	IV01_20	IV01_21	IV01_22	IV01_23	IV01_24	IV01_25	IV01_26	IV01_27	IV01_28	IV01_29	IV01_30	OT01_01	OT02_01	OT06	RD01_01	TIME001	TIME002	TIME003	TIME004	TIME005	TIME006	TIME007	TIME008	TIME009	TIME010	TIME011	TIME012	TIME013	TIME014	TIME015	TIME016	TIME017	TIME018	TIME019	TIME020	TIME021	TIME022	TIME023	TIME024	TIME025	TIME026	TIME027	TIME028	TIME029	TIME030	TIME031	TIME032	TIME033	TIME034	TIME035	TIME_SUM	MAILSENT	LASTDATA	FINISHED	Q_VIEWER	LASTPAGE	MAXPAGE	MISSING	MISSREL	TIME_RSI
                header = columns
            else:
                data.append(columns)

    print(f"len(data): {len(data)}")

    cloud_json_path = "data/hm3d/ver.3/dataset_cloud.json" if version == "ver.3" else f"data/hm3d/{version}/dataset_cloud_{split}.json"
    print(cloud_json_path)
    cloud_samples = None
    with open(cloud_json_path, "r") as json_file:
        cloud_samples = json.load(json_file)

    all_samples = []

    for d in data:
        if version == "ver.3":
            loop_range = range(1, 31)
        else:
            if split == "val":
                loop_range = range(1, 11)
            else:
                loop_range = range(11, 31)
        for idx in loop_range:
            instruction = d[header.index(f"BX{idx:02d}_01")]
            cloud_image_path = d[header.index(f"IV01_{idx:02d}")]
            serial_number = d[header.index(f"RD01_01")]

            sample_id_cloud_str = cloud_image_path.split("/")[-1].split(".")[0]
            cloud_sample = cloud_samples[int(sample_id_cloud_str)]
            assert cloud_sample["sample_id_cloud"] == sample_id_cloud_str

            old_targ_image_path = cloud_sample["targ"]["image_path"]
            targ_image_id_str = old_targ_image_path.split("/")[-1].split(".")[0]
            old_dest_image_path = cloud_sample["dest"]["image_path"]
            dest_image_id_str = old_dest_image_path.split("/")[-1].split(".")[0]
            env = cloud_sample["scene"]

            targ_sample = {
                "instruction_id": f"{env}_target_{targ_image_id_str}_to_dest_{dest_image_id_str}_{serial_number}",
                "mode": "<target>",
                "instruction": instruction,
                "gpt3_embeddings": f"data/hm3d/gpt3_embeddings/{env}/target_{targ_image_id_str}_to_dest_{dest_image_id_str}_{serial_number}.npy",
                "gt_bbox_id": [f"image_{env}_{targ_image_id_str}"],
                "image_path": [f"data/hm3d/{version}/{split}/{env}/raw/{targ_image_id_str}.jpg"],
                "environment": env,
                "full_image_feature_path": [f"data/hm3d/image_features/{env}/{targ_image_id_str}.npy"],
                "full_image_feature_path_2d": [f"data/hm3d/image_features_2d/{env}/{targ_image_id_str}.npy"],
                "other": {
                    "original_split": cloud_sample["original_split"],
                    "targ": cloud_sample["targ"],
                    "concat_image_path": cloud_sample["concat_image_path"],
                },
            }
            dest_sample = {
                "instruction_id": f"{env}_target_{targ_image_id_str}_to_dest_{dest_image_id_str}_{serial_number}",
                "mode": "<destination>",
                "instruction": instruction,
                "gpt3_embeddings": f"data/hm3d/gpt3_embeddings/{env}/target_{targ_image_id_str}_to_dest_{dest_image_id_str}_{serial_number}.npy",
                "gt_bbox_id": [f"image_{env}_{dest_image_id_str}"],
                "image_path": [f"data/hm3d/{version}/{split}/{env}/raw/{dest_image_id_str}.jpg"],
                "environment": env,
                "full_image_feature_path": [f"data/hm3d/image_features/{env}/{dest_image_id_str}.npy"],
                "full_image_feature_path_2d": [f"data/hm3d/image_features_2d/{env}/{dest_image_id_str}.npy"],
                "other": {
                    "original_split": cloud_sample["original_split"],
                    "dest": cloud_sample["dest"],
                    "concat_image_path": cloud_sample["concat_image_path"],
                },
            }
            all_samples.append(targ_sample)
            all_samples.append(dest_sample)

    print(f"len(all_samples): {len(all_samples)}")

    return all_samples


def main():
    database_v3_train = create_per_version("ver.3")
    database_v4_train = create_per_version("ver.4")
    database_v4_val = create_per_version("ver.4", split="val")

    # concatしてjsonとして保存
    database_all = database_v3_train + database_v4_train + database_v4_val
    print(f"len(database_all): {len(database_all)}")
    os.makedirs("data/hm3d/hm3d_dataset", exist_ok=True)
    with open("data/hm3d/hm3d_dataset/hm3d_database_no_gpt.json", 'w') as json_file:
        json.dump(database_all, json_file, indent=4)


# poetry run python src/create_database_json.py
if __name__ == "__main__":
    main()

# ver.3 train
# load: data/hm3d/csv/hm3d-carry_part1_clean.csv
# len(data): 72
# data/hm3d/ver.3/dataset_cloud.json
# len(all_samples): 4320
# ver.4 train
# load: data/hm3d/csv/hm3d-carry_part2_clean.csv
# len(data): 38
# data/hm3d/ver.4/dataset_cloud_train.json
# len(all_samples): 1520
# ver.4 val
# load: data/hm3d/csv/hm3d-carry_part2_clean.csv
# len(data): 38
# data/hm3d/ver.4/dataset_cloud_val.json
# len(all_samples): 760
# len(database_all): 6600
