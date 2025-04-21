"""上位20画像を保存"""
import json
import os
from PIL import Image


def save_images(experimental_id, dataset, result_path, id2path_path, instruction_ids, instructions=None):
    assert dataset in ["mp3d", "hm3d"]
    if dataset == "mp3d":
        assert instructions is not None

    with open(result_path, "r") as json_file:
        result_dict = json.load(json_file)
    with open(id2path_path, "r") as json_file:
        id2path_dict = json.load(json_file)

    for env in result_dict.keys():
        for result in result_dict[env]:
            if result["instruction_id"] in instruction_ids:
                # MP3Dはinstruction_idが一意ではない
                if dataset == "mp3d" and " ".join(result["instruction"].split(" ")[1:]) not in instructions:
                    continue

                identifier = f"_{''.join(result['instruction'].split(' ')[1:])}" if dataset == "mp3d" else ""
                os.makedirs(f"result/{experimental_id}/{dataset}/{result['instruction_id']}{identifier}/{'target' if result['instruction'].startswith('<target>') else 'destination'}", exist_ok=True)
                img = Image.open(id2path_dict[result["gt_image_id"][0]]["img_path"])
                img.save(f"result/{experimental_id}/{dataset}/{result['instruction_id']}{identifier}/{'target' if result['instruction'].startswith('<target>') else 'destination'}/00_gt.jpg")
                for rank, image_id in enumerate(result["top20"]):
                    img = Image.open(id2path_dict[image_id]["img_path"])
                    img.save(f"result/{experimental_id}/{dataset}/{result['instruction_id']}{identifier}/{'target' if result['instruction'].startswith('<target>') else 'destination'}/{rank+1:02d}.jpg")

    print(f"=== finished {dataset} ===")


def main():
    experimental_id = "id104_rec"

    # MP3D
    save_images(
        experimental_id=experimental_id,
        dataset="mp3d",
        result_path="result/id104_rec_mp3d_test.json",
        id2path_path="data/ltrpo_dataset/ltrpo_test_id2path.json",
        instruction_ids=[
            "QUCTc6BB5sX_target_158_bbox_267_to_dest_007",
            "QUCTc6BB5sX_target_196_bbox_304_to_dest_024",
            "QUCTc6BB5sX_target_215_bbox_172_to_dest_003",
            "QUCTc6BB5sX_target_230_bbox_174_to_dest_005",
            "QUCTc6BB5sX_target_254_bbox_204_to_dest_021",
            "QUCTc6BB5sX_target_254_bbox_209_to_dest_025",
            "QUCTc6BB5sX_target_264_bbox_208_to_dest_023",
            "QUCTc6BB5sX_target_312_bbox_331_to_dest_010",
            "QUCTc6BB5sX_target_350_bbox_181_to_dest_009",
            "X7HyMhZNoso_target_71_bbox_422_to_dest_011",
        ],
        instructions=[  # MP3Dのみ必要
            "Pick up the vase on the black shelf and put it on the brown square table.",
            "Pick up the picture on the wall and put it on the closet.",
            "Please put the ball on the desk.",
            "Pick up a picture and put it on the kitchen table.",
            "Put the red towel on the round wooden table",
            "Pick up the red towel and put it on the brown shelf.",
            "Pick up a white towel locating left of you and put it on the dark brown chest.",
            "Bring the wooden chair in the wine cellar to the wooden table with the lamp on it.",
            "Pick up a small picture frame on the upper left and put it on the white table .",
            "Pick up a dispenser and put it on the table alnong side on the wall in the livingroom.",
        ],
    )

    # HM3D
    save_images(
        experimental_id=experimental_id,
        dataset="hm3d",
        result_path="result/id104_rec_hm3d_test.json",
        id2path_path = "data/hm3d/hm3d_dataset/hm3d_test_id2path.json",
        instruction_ids=[
            "3t8DB4Uzvkt_target_000198_to_dest_000243_571211",
            "7GAhQPFzMot_target_000117_to_dest_000039_921919",
            "y9hTuugGdiq_target_000432_to_dest_000311_217963",
            "T6nG3E2Uui9_target_001713_to_dest_001371_850650",
            "T6nG3E2Uui9_target_001716_to_dest_001383_608261",
            "T6nG3E2Uui9_target_001733_to_dest_001824_434655",
            "bCPU9suPUw9_target_000936_to_dest_000644_482198",
            "LNg5mXe1BDj_target_000027_to_dest_000007_352507",
            "Nfvxx8J5NCo_target_000018_to_dest_000044_571211",
            "a8BtkwhxdRV_target_000004_to_dest_000077_571211",
        ],
    )


# poetry run python src/save_qualitative.py
if __name__ == "__main__":
    main()
