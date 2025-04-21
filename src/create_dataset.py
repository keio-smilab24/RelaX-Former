"""学習用のデータセットを作成"""

import json
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

import clip
from stanford_parser import get_all_np, get_stanford_parser_proc


def crop_bbox(img_path, x1, y1, x2, y2):
    # 画像の読み込み
    im = Image.open(img_path)
    # plt.imshow(im)

    # Crop
    im = im.crop((x1, y1, x2, y2))

    # 表示
    # im_list = np.asarray(im)
    # plt.imshow(im_list)
    # plt.show()

    return im


def create_bbox_images(db_path, img_dir, bbox_save_dir, bbox_npy_save_dir, img_npy_save_dir, ref_db_path, bbox=True):
    clip_base_model = "ViT-L/14"
    clip_model, preprocess_clip = clip.load(clip_base_model, device="cuda:0")

    db = json.load(open(db_path, "r"))

    path_db = []

    for env, v1 in tqdm(db.items()):
        for waypoint, v2 in v1.items():
            for view, v3 in v2.items():
                if bbox:
                    for objId, data in v3.items():
                        # create_bbox
                        img_path = f"{img_dir}/{env}/{waypoint}/id{view}.jpg"
                        x1, y1, x2, y2 = (
                            data["bbox"][0],
                            data["bbox"][1],
                            data["bbox"][0] + data["bbox"][2],
                            data["bbox"][1] + data["bbox"][3],
                        )
                        cropped_img = crop_bbox(img_path, x1, y1, x2, y2)
                        save_path = f"{bbox_save_dir}/{env}/{waypoint}/id{view}"
                        os.makedirs(save_path, exist_ok=True)
                        cropped_img.save(f"{save_path}/{objId}.jpg")

                        # create npy file for bbox
                        clip_feature = clip_model.encode_image(preprocess_clip(cropped_img).unsqueeze(0).to("cuda:0"))
                        output_data = clip_feature.to("cpu").detach().numpy().copy()

                        output_dir = f"{bbox_npy_save_dir}/{env}/{waypoint}/id{view}"
                        os.makedirs(output_dir, exist_ok=True)
                        image_npy_file = f"{output_dir}/{objId}.npy"
                        np.save(image_npy_file, output_data)

                # create npy file for entire image with side
                for side_view in side_image_view(int(view)):
                    img_path = f"{img_dir}/{env}/{waypoint}/id{side_view:02}.jpg"
                    if img_path not in path_db:
                        clip_feature = clip_model.encode_image(
                            preprocess_clip(Image.open(img_path)).unsqueeze(0).to("cuda:0")
                        )
                        output_data = clip_feature.to("cpu").detach().numpy().copy()

                        output_dir = f"{img_npy_save_dir}/{env}/{waypoint}"
                        os.makedirs(output_dir, exist_ok=True)
                        image_npy_file = f"{output_dir}/id{side_view:02}.npy"
                        np.save(image_npy_file, output_data)

                        path_db.append(img_path)


def side_image_view(view):
    if view % 12 == 0:
        left = view + 11
        right = view + 1
    elif view % 12 == 11:
        left = view - 1
        right = view - 11
    else:
        left = view - 1
        right = view + 1
    return left, view, right


def side_image_idx(view, side):
    idx = int(view[-2:])
    if side == "left":
        if idx % 12 == 0:
            side_idx = idx + 11
        else:
            side_idx = idx - 1
    else:
        if idx % 12 == 11:
            side_idx = idx - 11
        else:
            side_idx = idx + 1
    return side_idx


def create_dataset_by_bbox(split, input_db_path, img_dir, output_dataset_path, ref_db_path, bbox_dir):

    db = json.load(open(input_db_path, "r"))

    bbox_id = 1

    # create ref db
    ref_db = {}
    for data in json.load(open(ref_db_path, "r")):
        if data["label"] == 1:
            if data["instruction"] not in ref_db.keys():
                ref_db[data["instruction"]] = []
            ref_db[data["instruction"]].append(data["view"])

    # create tmp database objId 2 bboxId
    # bbox_db = { "bbox_ids" : [id1, id2, ...], "bboxes" : [ [x,y,w,h], [x,y,w,h], ... ]
    bbox_db = {}
    id2path = {}
    for env, v1 in tqdm(db.items()):
        for waypoint, v2 in v1.items():
            for view, v3 in v2.items():
                for objId, data in v3.items():
                    img_path = f"{img_dir}/{env}/{waypoint}/id{view}.jpg"
                    bbox_path = f"{bbox_dir}/{env}/{waypoint}/id{view}/{objId}.jpg"
                    if objId not in bbox_db.keys():
                        bbox_db[str(objId)] = {
                            "bbox_ids": [f"bbox_{env}_{bbox_id}"],
                            "bboxes": [data["bbox"]],
                            "img_path": [img_path],
                        }
                    else:
                        bbox_db[str(objId)]["bbox_ids"].append(f"bbox_{env}_{bbox_id}")
                        bbox_db[str(objId)]["bboxes"].append(data["bbox"])
                        bbox_db[str(objId)]["img_path"].append(img_path)
                    id2path[f"bbox_{env}_{bbox_id}"] = {
                        "img_path": img_path,
                        "bbox_path": bbox_path,
                        "bbox": data["bbox"],
                        "target_id": objId,
                    }
                    bbox_id += 1
    # print(bbox_db)
    # sys.exit()
    with open(f"{output_dataset_path}_by_bbox_id2path.json", "w") as wf:
        json.dump(id2path, wf, indent=2)

    # create dataset for eval
    if split == "train":
        instruction_id = 1
        instructions = []
        output_json = []
        for env, v1 in tqdm(db.items()):
            for waypoint, v2 in v1.items():
                for view, v3 in v2.items():
                    for objId, data in v3.items():
                        # if data["instructions"][0] not in instructions:
                        for inst in data["instructions"]:
                            if inst not in instructions:
                                dump = {}
                                dump["instruction"] = inst
                                dump["noun_phrases"] = list(get_all_np(inst))
                                dump["environment"] = env
                                dump["gt_waypoint"] = waypoint
                                dump["gt_view"] = f"id{view}"
                                dump["target_objId"] = objId

                                dump["nearby_object_bboxId"] = []
                                for nearby_objId in v3.keys():
                                    if nearby_objId != objId:
                                        for idx, path in enumerate(bbox_db[nearby_objId]["img_path"]):
                                            if view in path and waypoint in path:
                                                dump["nearby_object_bboxId"].append(
                                                    bbox_db[nearby_objId]["bbox_ids"][idx]
                                                )

                                output_json.append(dump)
                                instructions.append(inst)
                                # instructions.append(data["instructions"][0])
                                instruction_id += 1
        with open(f"{output_dataset_path}.json", "w") as wwf:
            json.dump(output_json, wwf, indent=2)
    else:
        instruction_id = 1
        instructions = []
        for env, v1 in tqdm(db.items()):
            output_json = []
            for waypoint, v2 in v1.items():
                for view, v3 in v2.items():
                    for objId, data in v3.items():
                        for inst in data["instructions"]:
                            if inst not in instructions:
                                dump = {}
                                dump["gt_bbox_id"] = bbox_db[objId]["bbox_ids"]
                                dump["instruction_id"] = instruction_id
                                dump["instruction"] = inst
                                dump["noun_phrases"] = list(get_all_np(inst))
                                dump["environment"] = env
                                dump["gt_waypoint"] = waypoint
                                dump["gt_view"] = f"id{view}"
                                dump["target_objId"] = objId
                                dump["target_object"] = data["target_object"]
                                dump["gt_bbox"] = bbox_db[objId]["bboxes"]
                                dump["image_path"] = bbox_db[objId]["img_path"]

                                dump["nearby_object_bboxId"] = []
                                for nearby_objId in v3.keys():
                                    if nearby_objId != objId:
                                        for idx, path in enumerate(bbox_db[nearby_objId]["img_path"]):
                                            if view in path and waypoint in path:
                                                dump["nearby_object_bboxId"].append(
                                                    bbox_db[nearby_objId]["bbox_ids"][idx]
                                                )

                                output_json.append(dump)
                                instructions.append(inst)
                                instruction_id += 1

                                # check bbox ========================
                                # id2path = json.load(open(f"dataset/reverie_retrieval_dataset/reverie_test_by_image_id2path.json", "r"))
                                # for idx, bbox_id in enumerate(dump["gt_bbox_id"]):
                                #     # image_path = id2path[bbox_id]
                                #     image_path = dump["image_path"][idx]
                                #     im = Image.open(image_path)
                                #     im_list = np.asarray(im)
                                #     plt.figure(figsize=(8, 6), dpi=300)
                                #     plt.imshow(im_list)
                                #     plt.axis("off")
                                #     bbox = dump["gt_bbox"][idx]
                                #     x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                                #     ax = plt.gca()
                                #     rect = patches.Rectangle(xy=(x, y), width=w, height=h, ec='cyan', linewidth=3, fill=False)
                                #     ax.add_patch(rect)
                                #     plt.show()

                                # check bbox ========================
            with open(f"{output_dataset_path}_by_bbox_{env}.json", "w") as wwf:
                json.dump(output_json, wwf, indent=2)
            # sys.exit()
    print(f"{split} #inst : {instruction_id-1}")


def create_eval_bbox_features(
    split, output_path, input_dataset_dir, full_img_npy_dir, bbox_npy_dir, dataset_environment
):
    for env in dataset_environment:
        output_data = {}
        image_features = []
        left_images = []
        right_images = []
        bboxId_list = []
        bbox_images = []
        for bboxId, data in json.load(open(f"{input_dataset_dir}_by_bbox_id2path.json", "r")).items():
            if env in data["img_path"]:

                # center
                tmp_path = data["img_path"].replace("data/EXTRACTED_IMGS_", full_img_npy_dir).replace("jpg", "npy")
                img_feature = np.load(tmp_path)
                image_features.append(img_feature)

                # left
                tmp_path = tmp_path.replace(
                    data["img_path"][-8:-4], f'id{side_image_idx(data["img_path"][-6:-4], side="left"):02}'
                )
                img_feature = np.load(tmp_path)
                left_images.append(img_feature)

                # right
                tmp_path = tmp_path.replace(
                    data["img_path"][-8:-4], f'id{side_image_idx(data["img_path"][-6:-4], side="right"):02}'
                )
                img_feature = np.load(tmp_path)
                right_images.append(img_feature)

                # imgId
                bboxId_list.append(bboxId)

                # bbox
                tmp_path = (
                    data["img_path"]
                    .replace("data/EXTRACTED_IMGS_", bbox_npy_dir)
                    .replace(".jpg", f'/{data["target_id"]}.npy')
                )
                img_feature = np.load(tmp_path)
                bbox_images.append(img_feature)

        output_data["data_path"] = f"{output_path}/eval_features_{split}_{env}.npy"
        output_data["bboxId_list"] = bboxId_list

        print(f"{split} #bbox : {len(bboxId_list)}")

        # save features
        np.save(output_data["data_path"], np.array(image_features))
        np.save(output_data["data_path"].replace(".npy", "_left.npy"), np.array(left_images))
        np.save(output_data["data_path"].replace(".npy", "_right.npy"), np.array(right_images))
        np.save(output_data["data_path"].replace(".npy", "_bbox.npy"), np.array(bbox_images))

        with open(f"{output_path}/eval_features_bbox_list_{split}_{env}.json", "w") as wf:
            json.dump(output_data, wf, indent=2)

    return 0


def create_dataset_by_full_image(split, input_db_path, img_dir, output_dataset_path):
    db = json.load(open(input_db_path, "r"))

    img_id = 1

    bbox_id = 1
    img_id = 0
    bboxid_2_imgid = {}
    id2path = {}

    # create tmp database objId 2 bboxId
    # bbox_db = [ { "bbox_ids" : [id1, id2, ...]   }  ]
    bbox_db = {}
    for env, v1 in tqdm(db.items()):
        for waypoint, v2 in v1.items():
            for view, v3 in v2.items():
                for objId, data in v3.items():
                    # print(data)
                    if objId not in bbox_db.keys():
                        bbox_db[str(objId)] = {"bbox_ids": [f"bbox_{env}_{bbox_id}"], "bboxes": [data["bbox"]]}
                    else:
                        bbox_db[str(objId)]["bbox_ids"].append(f"bbox_{env}_{bbox_id}")
                        bbox_db[str(objId)]["bboxes"].append(data["bbox"])
                    bboxid_2_imgid[f"bbox_{env}_{bbox_id}"] = img_id
                    bbox_id += 1
                id2path[f"image_{env}_{img_id}"] = f"{img_dir}/{env}/{waypoint}/id{view}.jpg"
                img_id += 1

                # # check bbox ========================
                # image_path = f"{img_dir}/{env}/{waypoint}/id{view}.jpg"
                # im = Image.open(image_path)
                # im_list = np.asarray(im)
                # plt.figure(figsize=(8, 6), dpi=300)
                # plt.imshow(im_list)
                # plt.axis("off")
                # bbox = data["bbox"]
                # x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                # ax = plt.gca()
                # rect = patches.Rectangle(xy=(x, y), width=w, height=h, ec='cyan', linewidth=3, fill=False)
                # ax.add_patch(rect)
                # plt.show()
                # # check bbox ========================

    # print(bboxid_2_imgid)

    with open(f"{output_dataset_path}_by_image_id2path.json", "w") as wf:
        json.dump(id2path, wf, indent=2)

    # create dataset for eval
    if split != "train":
        instruction_id = 1
        instructions = []
        for env, v1 in tqdm(db.items()):
            output_json = []
            for waypoint, v2 in v1.items():
                for view, v3 in v2.items():
                    for objId, data in v3.items():
                        for inst in data["instructions"]:
                            if inst not in instructions:
                                dump = {}
                                tmp = set()
                                for bbox_id in bbox_db[objId]["bbox_ids"]:
                                    tmp.add(f"image_{env}_{bboxid_2_imgid[str(bbox_id)]}")

                                dump["gt_img_id"] = list(tmp)
                                dump["instruction_id"] = instruction_id
                                dump["instruction"] = inst
                                dump["noun_phrases"] = list(get_all_np(inst))
                                dump["environment"] = env
                                dump["gt_waypoint"] = waypoint
                                dump["gt_view"] = f"id{view}"
                                dump["target_objId"] = objId
                                dump["target_object"] = data["target_object"]
                                dump["gt_bbox"] = bbox_db[objId]["bboxes"]

                                output_json.append(dump)
                                instructions.append(inst)
                                instruction_id += 1

            with open(f"{output_dataset_path}_by_image_{env}.json", "w") as wwf:
                json.dump(output_json, wwf, indent=2)
    else:  # train
        instruction_id = 1
        instructions = []
        output_json = []
        for env, v1 in tqdm(db.items()):
            for waypoint, v2 in v1.items():
                for view, v3 in v2.items():
                    for objId, data in v3.items():
                        # if data["instructions"][0] not in instructions:
                        for inst in data["instructions"]:
                            if inst not in instructions:
                                dump = {}
                                dump["instruction"] = inst
                                dump["noun_phrases"] = list(get_all_np(inst))
                                dump["environment"] = env
                                dump["gt_waypoint"] = waypoint
                                dump["gt_view"] = f"id{view}"
                                dump["target_objId"] = objId

                                output_json.append(dump)
                                instructions.append(inst)
                                instruction_id += 1
        with open(f"{output_dataset_path}.json", "w") as wwf:
            json.dump(output_json, wwf, indent=2)

    print(f"{split} #inst : {instruction_id-1}")


def create_eval_image_features(split, output_path, input_dataset_dir, full_img_npy_dir, dataset_environment):
    # 1210
    for env in dataset_environment:
        output_data = {}
        image_features = []
        left_images = []
        right_images = []
        imageId_list = []
        for imgId, img_path in json.load(open(f"{input_dataset_dir}_by_image_id2path.json", "r")).items():
            if env in img_path:
                # center
                tmp_path = img_path.replace("data/EXTRACTED_IMGS_", full_img_npy_dir).replace("jpg", "npy")
                img_feature = np.load(tmp_path)
                image_features.append(img_feature)

                # left
                tmp_path = tmp_path.replace(img_path[-8:-4], f'id{side_image_idx(img_path[-6:-4], side="left"):02}')
                img_feature = np.load(tmp_path)
                left_images.append(img_feature)

                # right
                tmp_path = tmp_path.replace(img_path[-8:-4], f'id{side_image_idx(img_path[-6:-4], side="right"):02}')
                img_feature = np.load(tmp_path)
                right_images.append(img_feature)

                # imgId
                imageId_list.append(imgId)

        output_data["data_path"] = f"{output_path}/eval_features_{split}_{env}.npy"
        output_data["imgId_list"] = imageId_list

        # save features
        np.save(output_data["data_path"], np.array(image_features))
        np.save(output_data["data_path"].replace(".npy", "_left.npy"), np.array(left_images))
        np.save(output_data["data_path"].replace(".npy", "_right.npy"), np.array(right_images))

        with open(f"{output_path}/eval_features_full_image_list_{split}_{env}.json", "w") as wf:
            json.dump(output_data, wf, indent=2)

        print(f"{split} #img : {len(imageId_list)}")

    return 0


def main():
    # 1215
    environment = {}
    environment["val_seen"] = ["8WUmhLawc2A", "1pXnuDYAj8r", "VzqfbhrpDEA", "ac26ZMwG7aT", "rPc6DW4iMge"]
    environment["val_unseen"] = ["EU6Fwq7SyZv", "x8F5xyUWy9e", "zsNo4HB9uLZ", "oLBMNvg9in8"]
    environment["test"] = ["2azQ1b91cZZ", "QUCTc6BB5sX", "X7HyMhZNoso", "TbHJrupSAjP"]

    db_path = "data/reverie_database/reverie_retrieval_database"
    img_dir = "data/EXTRACTED_IMGS_"
    bbox_save_dir = "data/bbox_images"
    bbox_npy_save_dir = "data/bbox_features"
    image_npy_save_dir = "data/full_image_features"
    dataset_path = "data/reverie_retrieval_dataset/reverie"
    ref_dataset_path = "data/reverie"
    eval_features_path = "data/reverie_retrieval_dataset"

    bbox_flag = False

    create_bbox_flag = True
    create_bbox_dataset_flag = False
    create_bbox_eval_flag = False
    create_full_image_dataset_flag = True
    create_full_image_eval_flag = True

    os.makedirs(eval_features_path, exist_ok=True)

    if create_bbox_flag:  #
        for split in ["train", "val_seen", "val_unseen", "test"]:
            create_bbox_images(
                f"{db_path}_{split}.json",
                img_dir,
                bbox_save_dir,
                bbox_npy_save_dir,
                image_npy_save_dir,
                f"{ref_dataset_path}_{split}.json",
                bbox=bbox_flag,
            )

    if create_bbox_dataset_flag:
        for split in ["train", "val_seen", "val_unseen", "test"]:
            create_dataset_by_bbox(
                split,
                f"{db_path}_{split}.json",
                img_dir,
                f"{dataset_path}_{split}",
                f"{ref_dataset_path}_{split}.json",
                bbox_save_dir,
            )

    if create_bbox_eval_flag:
        for split in ["val_seen", "val_unseen", "test"]:
            create_eval_bbox_features(
                split,
                eval_features_path,
                f"{dataset_path}_{split}",
                image_npy_save_dir,
                bbox_npy_save_dir,
                dataset_environment=environment[split],
            )

    if create_full_image_dataset_flag:
        for split in ["test", "val_seen", "train", "val_unseen"]:
            create_dataset_by_full_image(split, f"{db_path}_{split}.json", img_dir, f"{dataset_path}_{split}")

    if create_full_image_eval_flag:
        for split in ["val_unseen", "val_seen", "test"]:
            create_eval_image_features(
                split,
                eval_features_path,
                f"{dataset_path}_{split}",
                image_npy_save_dir,
                dataset_environment=environment[split],
            )


if __name__ == "__main__":
    with get_stanford_parser_proc():
        main()
