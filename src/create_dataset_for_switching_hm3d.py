"""学習データとなるデータベースを作成"""

import argparse
import copy
import json
import os

import numpy as np
import torch
from PIL import Image
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    ViTModel,
)

import clip
from llava.llava import LLaVAImageCaptioner
from sam.sam_server import SimpleSegmentAnything
from stanford_parser import get_all_np, get_stanford_parser_proc


def create_np(database):
    new_database = copy.deepcopy(database)
    for i, data in enumerate(tqdm(database)):
        inst = data["instruction"]
        new_database[i]["noun_phrases"] = list(get_all_np(inst))

    output_file = "data/hm3d/hm3d_dataset/hm3d_database_np.json"
    with open(output_file, "w") as wf:
        json.dump(new_database, wf, indent=2)

    return new_database


def create_full_image_feature(database, enable_clip=False, enable_sam=False, enable_ViT=False, enable_llava=False):
    if enable_clip:
        clip_base_model = "ViT-L/14"
        clip_model, preprocess_clip = clip.load(clip_base_model, device="cuda:0")

    if enable_sam:
        segment_anything = SimpleSegmentAnything()
        segment_anything.load_model(sam_checkpoint="src/sam/sam_vit_h_4b8939.pth")

    if enable_ViT:
        image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224")
        vit_model.eval()  # type: ignore
        vit_model.to("cuda:0")

    if enable_llava:
        llava_processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        llava_model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf",
            torch_dtype=torch.float16,
        )
        llava_model.eval()  # type: ignore
        llava_model.to("cuda:0")

    image_path_set = set()

    jpg_to_npy_dict = {}

    for data in database:
        instruction = data["instruction"]
        for image_path, feature_path in zip(data["image_path"], data["full_image_feature_path"]):
            image_path_instruction_pair = (image_path, instruction)
            image_path_set.add(image_path_instruction_pair)
            jpg_to_npy_dict[image_path] = feature_path

    for img_path, instruction in tqdm(sorted(list(image_path_set))):
        img = Image.open(img_path)
        feature_vectors = []

        # Llava
        if enable_llava:
            resized_img = img.resize((224, 224))
            prompt = "[INST] <image>\nThis is an image of a certain house. Please describe in detail the location of objects in this image. In doing so, be sure to mention the following object given in the instructions.\nAfter describing the object within the directive, continue to elaborate on the arrangement of other objects throughout the house, ensuring a thorough and comprehensive description.\nIn the description, refrain from using personal pronouns such as 'I', 'we', or 'you'.\nAlso, avoid including sensory details.\nFocus solely on providing a clear description.\nBegin the description with the phrase 'In the image'\nThe directions should be as indicated in the image\n[/INST]"

            _input = llava_processor(images=resized_img, text=prompt, return_tensors="pt").to("cuda:0")  # type: ignore

            with torch.no_grad():
                _llava_output = llava_model(**_input, output_hidden_states=True)  # type: ignore

            llava_feature = _llava_output.hidden_states[-1][:, 0, :].to("cpu").detach().numpy().copy()
            print(llava_feature.shape)

            llava_npy_path = jpg_to_npy_dict[img_path].replace("features", "llava_features_latent_full_bit")
            llava_dir = os.path.dirname(llava_npy_path)
            os.makedirs(llava_dir, exist_ok=True)
            np.save(llava_npy_path, llava_feature)

        if enable_clip or enable_ViT or enable_sam:
            if enable_clip:
                # CLIP
                clip_feature = clip_model.encode_image(preprocess_clip(img).unsqueeze(0).to("cuda:0"))  # type: ignore
                feature_vectors.append(clip_feature.to("cpu").detach().numpy().copy())

            # SAM
            if enable_clip and enable_sam:
                masks = segment_anything.mask_generator.generate(np.asarray(img))  # type: ignore
                processed_masks = segment_anything.process_anns(masks)  # type: ignore
                overlaid = segment_anything.merge_masks(img, processed_masks)  # type: ignore
                sam = Image.fromarray(overlaid)
                sam_clip_feature = clip_model.encode_image(preprocess_clip(sam).unsqueeze(0).to("cuda:0"))  # type: ignore
                feature_vectors.append(sam_clip_feature.to("cpu").detach().numpy().copy())

            # ViT
            if enable_ViT:
                resized_img = img.resize((224, 224))

                _image = image_processor(images=resized_img, return_tensors="pt").pixel_values.to("cuda:0")  # type: ignore
                with torch.no_grad():
                    _vit_output = vit_model(_image)  # type: ignore
                vit_feature = _vit_output.last_hidden_state[:, 0, :].to("cpu").detach().numpy().copy()
                feature_vectors.append(vit_feature)

            output_data = np.vstack(feature_vectors)

            npy_path = jpg_to_npy_dict[img_path]
            dir = "/".join(npy_path.split("/")[:-1])
            os.makedirs(dir, exist_ok=True)
            np.save(npy_path, output_data)


def create_full_image_feature_2d(database):
    clip_base_model = "RN50x4"
    clip_model, preprocess_clip = clip.load(clip_base_model, device="cuda:0")

    image_path_set = set()

    jpg_to_npy_dict = {}

    for data in database:
        for image_path, feature_path_2d in zip(data["image_path"], data["full_image_feature_path_2d"]):
            image_path_set.add(image_path)
            jpg_to_npy_dict[image_path] = feature_path_2d

    for img_path in tqdm(sorted(list(image_path_set))):
        img = Image.open(img_path)
        img_tensor = preprocess_clip(img).unsqueeze(0).to("cuda:0")  # type: ignore
        clip_feature = clip_model.encode_image(img_tensor, layer=3)

        clip_feature = torch.flatten(clip_feature, 2)
        p_enc_1d = Summer(PositionalEncoding1D(clip_feature.shape[-1])).to("cuda:0")
        clip_feature_pe = p_enc_1d(clip_feature)
        clip_feature_pe = clip_feature_pe.view(
            clip_feature.shape[0],
            clip_feature.shape[1],
            int(np.sqrt(clip_feature.shape[2])),
            int(np.sqrt(clip_feature.shape[2])),
        )

        output_data = clip_feature_pe.to("cpu").detach().numpy().copy()

        npy_path = jpg_to_npy_dict[img_path]
        dir = "/".join(npy_path.split("/")[:-1])
        os.makedirs(dir, exist_ok=True)
        np.save(npy_path, output_data)


def create_dataset_per_env(database):
    # eval_features_bbox_list_{split}_{env}.json
    # hm3d_{split}_{env}.json
    # hm3d_{split}_id2path.json
    envs_dict = {
        "val_unseen": [
            "TEEsavR23oF-W7k2QWzBrFY-7UrtFsADwob-bxsVRursffK",
            "66seV3BWPoX-L53DsRRk4Ch-HaxA7YrQdEC-yr17PDCnDDW-mma8eWq3nNQ-XNeHsjL6nBB",
            "4ok3usBNeis-rsggHU7g7dh-cvZr5TUy5C5",
            "c5eTyR3Rxyh-LT9Jq6dN3Ea-h1zeeAwLh9Z-7MXmsvcQjpJ-q5QZSEeHe5g-BAbdmeyTvMZ",
            "Qpor2mEya8F-cYkrGrCg2kB-7Ukhou1GxYi",
            "58NLZxWBSpk-MHPLjHsuG27-bzCsHPLDztK-5cdEh9F2hJL-hyFzGGJCSYs-u1bkiGRVyu9",
            "nrA1tAA17Yp-LEFTm3JecaC-fsQtJ8t3nTf",
            "5jp3fCRSRjc-k1cupFYWXJ6-svBbv1Pavdk-F7EAMsdDASd-kJJyRFXVpx2",
            "XMHNu9rRQ1y-BFRyYbPCCPE-AWUFxHEyV3T-z9YwN9M8FpG",
        ],
        "test": [
            "VBzV5z6i1WS-7GAhQPFzMot-3t8DB4Uzvkt-y9hTuugGdiq-tQ5s4ShP627",
            "mL8ThkuaVTM-CrMo8WxCyVb-X7gTkoDHViv",
            "GLAQ4DNUx5U-QHhQZWdMpGJ-T6nG3E2Uui9",
            "FnSn2KSrALj-bCPU9suPUw9",
            "6D36GQHuP8H-YRUkbU5xsYj-LNg5mXe1BDj-Nfvxx8J5NCo-HMkoS756sz6-SByzJLxpRGn",
            "wcojb4TFT35-ziup5kvtCCR-a8BtkwhxdRV-dHwjuKfkRUR-jgPBycuV1Jq-6s7QHgap2fW",
            "rXXL6twQiWc-hkr2MGpHD6B-q3hn1WQ12rz-u8ug2rtNARf-uSKXQ5fFg6u-DYehNKdT76V",
            "eF36g7L6Z9M-XB4GS9ShBRE-QaLdnwvtxbs-vBMLrTe4uLA",
            "uLz9jNga3kC-vd3HHTEpmyA-dVW2D7TDctW-X4qjx5vquwH-rJhMRvNn4DS",
        ],
        "train": ["train"],
    }

    for split in envs_dict.keys():

        output_by_split = {}
        for env in envs_dict[split]:
            # create hm3d_{split}_{env}.json
            env_dataset = []
            for data in database:
                if split == "train":
                    # リークに注意
                    if all(
                        [data["environment"] not in val_envs.split("-") for val_envs in envs_dict["val_unseen"]]
                    ) and all([data["environment"] not in test_envs.split("-") for test_envs in envs_dict["test"]]):
                        env_dataset.append(data)
                else:  # val_unseen or test
                    if data["environment"] in env.split("-"):
                        env_dataset.append(data)

            # output hm3d_{split}_{env}.json
            output_file = f"data/hm3d/hm3d_dataset/hm3d_{split}_{env}.json"
            with open(output_file, "w") as wf:
                json.dump(sorted(env_dataset, key=lambda x: x["instruction_id"]), wf, indent=2)

            # create hm3d_{split}_id2path.json
            output_by_env = {}
            for data in sorted(env_dataset, key=lambda x: x["instruction_id"]):
                for i, bbox_id in enumerate(data["gt_bbox_id"]):
                    if bbox_id not in output_by_env.keys():
                        dump = {}
                        dump["img_path"] = data["image_path"][i]
                        dump["full_image_feature_path"] = data["full_image_feature_path"][i]
                        dump["full_image_feature_path_2d"] = data["full_image_feature_path_2d"][i]
                        dump["gpt4v_embeddings"] = data["gpt4v_embeddings"]
                        dump["pseudo_gt"] = data["pseudo_gt"] if "pseudo_gt" in data else []
                        output_by_env[bbox_id] = dump
            output_by_split.update(output_by_env)

            # output eval_features_{}.npy
            if split != "train":
                bboxId_list = []
                image_features = []
                image_features_2d = []
                llava_features = []
                gpt4v_features = []
                # alpha_clip_features = []
                for bbox_id in sorted(output_by_env.keys()):
                    # center
                    img_feature = np.load(output_by_env[bbox_id]["full_image_feature_path"])
                    image_features.append(img_feature)

                    # center 2d
                    img_feature_2d = np.load(output_by_env[bbox_id]["full_image_feature_path_2d"])
                    image_features_2d.append(img_feature_2d)  # typo...

                    # llava image
                    llava_npy_path = output_by_env[bbox_id]["full_image_feature_path"].replace(
                        "features", "llava_features_latent_full_bit"
                    )
                    llava_feature = np.load(llava_npy_path)
                    llava_features.append(llava_feature)

                    # gpt4v
                    gpt4v_feature = np.load(output_by_env[bbox_id]["gpt4v_embeddings"])
                    gpt4v_features.append(gpt4v_feature)

                    # imgId
                    bboxId_list.append(bbox_id)

                output_data = {}
                output_data["bboxId_list"] = bboxId_list

                os.makedirs("data/hm3d/eval_features", exist_ok=True)
                base_path = f"data/hm3d/eval_features/eval_features_{split}_{env}"
                np.save(f"{base_path}_full.npy", np.array(image_features))
                np.save(f"{base_path}_full_2d.npy", np.array(image_features_2d))
                np.save(f"{base_path}_llava.npy", np.array(llava_features))
                np.save(f"{base_path}_gpt4v_embeddings.npy", np.array(gpt4v_features))
                with open(f"{base_path}_bbox_list.json", "w") as wf:
                    json.dump(output_data, wf, indent=2)

        # 定性的結果で必要
        # output hm3d_{split}_id2path.json
        output_file = f"data/hm3d/hm3d_dataset/hm3d_{split}_id2path.json"
        with open(output_file, "w") as wf:
            json.dump(output_by_split, wf, indent=2)


def create_pseudo_gt_with_llava(database, cos_similarity_data):
    llava_processor = LLaVAImageCaptioner()
    output = []

    def split_string(s):
        parts = s.split(">", 1)
        return [parts[0] + ">", parts[1].strip()] if len(parts) == 2 else [s]

    for instruction_obj in tqdm(cos_similarity_data["output"]):
        instruction_parts = split_string(instruction_obj["instruction"])
        data = next(
            (
                d
                for d in database
                if d["mode"] == ("<target>" if instruction_parts[0] == "<target>" else "<destination>")
                and d["instruction"].strip() == instruction_parts[1]
            ),
            None,
        )
        if data is None:
            print(f"Data not found for instruction: {instruction_parts}")
            continue
        top20_img_ids = instruction_obj["top40"]
        pseudo_gt_img_ids = []
        for img_id in top20_img_ids:
            img_id_parts = img_id.split("_")
            waypoint_name, id = img_id_parts[1], img_id_parts[2]
            img_path = f"data/hm3d/ver.4/train/{waypoint_name}/raw/{id}.jpg"
            if os.path.exists(img_path):
                try:
                    img = Image.open(img_path)
                except FileNotFoundError:
                    print(f"Image not found for ver.4: {img_path} for {img_id}")
                    continue
            else:
                img_path = f"data/hm3d/ver.3/train/{waypoint_name}/raw/{id}.jpg"
                if os.path.exists(img_path):
                    try:
                        img = Image.open(img_path)
                    except FileNotFoundError:
                        print(f"Image not found for ver.3: {img_path} for {img_id}")
                        continue
                else:
                    img_path = f"data/hm3d/ver.4/val/{waypoint_name}/raw/{id}.jpg"
                    if os.path.exists(img_path):
                        try:
                            img = Image.open(img_path)
                        except FileNotFoundError:
                            print(f"Image not found for ver.4 val: {img_path} for {img_id}")
                            continue
                    else:
                        print(f"Image not found for both ver.3 and ver.4: {img_path} for {img_id}")
                    continue

            target_or_destination = (
                data["other"]["targ"]["label"] if data["mode"] == "<target>" else data["llm_data"]["destination"]
            )
            prompt = f"[INST] <image>\nCan you see a {target_or_destination} in the image? Tell me with ‘true’ or ‘false’.[/INST]"
            caption = llava_processor.generate_caption(img, prompt)
            if "true" in caption.lower():
                pseudo_gt_img_ids.append({"gt_img_id": img_id, "rank": top20_img_ids.index(img_id)})
        data["pseudo_gt"] = pseudo_gt_img_ids
        output.append(data)
    with open("data/hm3d/hm3d_dataset/hm3d_pseudo_gt.json", "w") as wf:
        json.dump(output, wf, indent=2)


def create_dataset_for_pseudo_gt(database):
    # eval_features_bbox_list_{split}_{env}.json
    # hm3d_{split}_{env}.json
    # hm3d_{split}_id2path.json
    reference_env_dict = {
        "val_unseen": [
            "TEEsavR23oF-W7k2QWzBrFY-7UrtFsADwob-bxsVRursffK",
            "66seV3BWPoX-L53DsRRk4Ch-HaxA7YrQdEC-yr17PDCnDDW-mma8eWq3nNQ-XNeHsjL6nBB",
            "4ok3usBNeis-rsggHU7g7dh-cvZr5TUy5C5",
            "c5eTyR3Rxyh-LT9Jq6dN3Ea-h1zeeAwLh9Z-7MXmsvcQjpJ-q5QZSEeHe5g-BAbdmeyTvMZ",
            "Qpor2mEya8F-cYkrGrCg2kB-7Ukhou1GxYi",
            "58NLZxWBSpk-MHPLjHsuG27-bzCsHPLDztK-5cdEh9F2hJL-hyFzGGJCSYs-u1bkiGRVyu9",
            "nrA1tAA17Yp-LEFTm3JecaC-fsQtJ8t3nTf",
            "5jp3fCRSRjc-k1cupFYWXJ6-svBbv1Pavdk-F7EAMsdDASd-kJJyRFXVpx2",
            "XMHNu9rRQ1y-BFRyYbPCCPE-AWUFxHEyV3T-z9YwN9M8FpG",
        ],
        "test": [
            "VBzV5z6i1WS-7GAhQPFzMot-3t8DB4Uzvkt-y9hTuugGdiq-tQ5s4ShP627",
            "mL8ThkuaVTM-CrMo8WxCyVb-X7gTkoDHViv",
            "GLAQ4DNUx5U-QHhQZWdMpGJ-T6nG3E2Uui9",
            "FnSn2KSrALj-bCPU9suPUw9",
            "6D36GQHuP8H-YRUkbU5xsYj-LNg5mXe1BDj-Nfvxx8J5NCo-HMkoS756sz6-SByzJLxpRGn",
            "wcojb4TFT35-ziup5kvtCCR-a8BtkwhxdRV-dHwjuKfkRUR-jgPBycuV1Jq-6s7QHgap2fW",
            "rXXL6twQiWc-hkr2MGpHD6B-q3hn1WQ12rz-u8ug2rtNARf-uSKXQ5fFg6u-DYehNKdT76V",
            "eF36g7L6Z9M-XB4GS9ShBRE-QaLdnwvtxbs-vBMLrTe4uLA",
            "uLz9jNga3kC-vd3HHTEpmyA-dVW2D7TDctW-X4qjx5vquwH-rJhMRvNn4DS",
        ],
    }
    envs_dict = {
        "pseudo_gt": ["pseudo_gt"],
    }

    batch_size = 100  # Each batch contains 100 items

    for split in envs_dict.keys():
        output_by_split = {}
        for env in envs_dict[split]:
            env_dataset = []
            for data in tqdm(database):
                if all(
                    data["environment"] not in val_envs.split("-") for val_envs in reference_env_dict["val_unseen"]
                ) and all(data["environment"] not in test_envs.split("-") for test_envs in reference_env_dict["test"]):
                    env_dataset.append(data)

            sorted_env_dataset = sorted(env_dataset, key=lambda x: x["instruction_id"])
            total_parts = len(sorted_env_dataset) // batch_size + (len(sorted_env_dataset) % batch_size > 0)

            for part in range(total_parts):
                start_index = part * batch_size
                end_index = start_index + batch_size
                part_dataset = sorted_env_dataset[start_index:end_index]

                output_file = f"data/hm3d/hm3d_dataset/hm3d_{split}_{env}_batch{part}.json"
                with open(output_file, "w") as wf:
                    json.dump(part_dataset, wf, indent=2)

                output_by_env = {}
                for data in part_dataset:
                    for i, bbox_id in enumerate(data["gt_bbox_id"]):
                        if bbox_id not in output_by_env:
                            dump = {
                                "img_path": data["image_path"][i],
                                "full_image_feature_path": data["full_image_feature_path"][i],
                                "full_image_feature_path_2d": data["full_image_feature_path_2d"][i],
                                "gpt4v_embeddings": data["gpt4v_embeddings"],
                            }
                            output_by_env[bbox_id] = dump

                # Compile feature files for each batch
                bboxId_list, image_features, image_features_2d, llava_features, gpt4v_features = [], [], [], [], []
                for bbox_id in sorted(output_by_env):
                    img_feature = np.load(output_by_env[bbox_id]["full_image_feature_path"])
                    image_features.append(img_feature)

                    img_feature_2d = np.load(output_by_env[bbox_id]["full_image_feature_path_2d"])
                    image_features_2d.append(img_feature_2d)

                    llava_npy_path = output_by_env[bbox_id]["full_image_feature_path"].replace(
                        "features", "llava_features_latent_full_bit"
                    )
                    llava_feature = np.load(llava_npy_path)
                    llava_features.append(llava_feature)

                    gpt4v_feature = np.load(output_by_env[bbox_id]["gpt4v_embeddings"])
                    gpt4v_features.append(gpt4v_feature)

                    bboxId_list.append(bbox_id)

                output_data = {"bboxId_list": bboxId_list}
                base_path = f"data/hm3d/eval_features/eval_features_{split}_{env}_batch{part}"
                np.save(f"{base_path}_full.npy", np.array(image_features))
                np.save(f"{base_path}_full_2d.npy", np.array(image_features_2d))
                np.save(f"{base_path}_llava.npy", np.array(llava_features))
                np.save(f"{base_path}_gpt4v_embeddings.npy", np.array(gpt4v_features))

                with open(f"{base_path}_bbox_list.json", "w") as wf:
                    json.dump(output_data, wf, indent=2)

        # Output hm3d_{split}_id2path.json for entire split, not by batches
        output_file = f"data/hm3d/hm3d_dataset/hm3d_{split}_id2path.json"
        with open(output_file, "w") as wf:
            json.dump(output_by_split, wf, indent=2)
        print(f"Output {output_file}")


def interpolate_features(base_database, target_database):
    """
    add missing data from base_database to target_database
    """
    target_dict = {(data["instruction"], data["mode"]): data for data in target_database}

    for base_data in base_database:
        key = (base_data["instruction"], base_data["mode"])
        if key not in target_dict:
            target_database.append(base_data)

    open("data/hm3d/hm3d_dataset/__hm3d_pseudo_gt.json", "w").write(json.dumps(target_database, indent=2))


# poetry run python src/create_dataset_for_switching_hm3d.py --np --full --dataset --full_2d --baseline_dataset --sam
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--np", action="store_true")
    parser.add_argument("--use_np_cached", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--dataset", action="store_true")
    parser.add_argument("--baseline_dataset", action="store_true")
    parser.add_argument("--full_2d", action="store_true")
    parser.add_argument("--sam", action="store_true")
    parser.add_argument("--vit", action="store_true")
    parser.add_argument("--clip", action="store_true")
    parser.add_argument("--llava", action="store_true")
    parser.add_argument("--pseudo_gt", action="store_true")
    parser.add_argument("--pseudo_gt_with_llava", action="store_true")
    parser.add_argument("--interpolate", action="store_true")

    args = parser.parse_args()

    dataset_file = "data/hm3d/hm3d_dataset/hm3d_database.json"

    with open(dataset_file, "r") as f:
        database = json.load(f)

    if args.use_np_cached:
        with open("data/hm3d/hm3d_dataset/hm3d_database_np.json", "r") as f:
            database = json.load(f)

    if args.np:
        database = create_np(database)

    if args.full:
        create_full_image_feature(
            database, enable_clip=args.clip, enable_sam=args.sam, enable_ViT=args.vit, enable_llava=args.llava
        )

    if args.full_2d:
        create_full_image_feature_2d(database)

    if args.dataset:
        create_dataset_per_env(database)

    if args.pseudo_gt_with_llava:
        cos_similarity_data = "./result/hm3d_cos_similarity_clip_40_without_similarity.json"
        with open(cos_similarity_data, "r") as f:
            cos_similarity_data = json.load(f)
        create_pseudo_gt_with_llava(database, cos_similarity_data)

    if args.interpolate:
        base_database_file = "data/hm3d/hm3d_dataset/hm3d_database_20_pseudo_gt.json"
        target_database_file = "data/hm3d/hm3d_dataset/hm3d_database.json"
        with open(base_database_file, "r") as f:
            base_database = json.load(f)
        with open(target_database_file, "r") as f:
            target_database = json.load(f)
        interpolate_features(base_database, target_database)

    # if args.baseline_dataset:
    #     create_dataset_per_env_for_baseline(database, "target")
    #     create_dataset_per_env_for_baseline(database, "destination")

    if args.pseudo_gt:
        create_dataset_for_pseudo_gt(database)


if __name__ == "__main__":
    with get_stanford_parser_proc():
        main()
