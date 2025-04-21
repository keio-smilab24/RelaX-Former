import argparse
import json
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataloader
from model import ClipReverie

warnings.simplefilter("ignore")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="model.pth")
    parser.add_argument("--clip_base_model", default="ViT-L/14")
    parser.add_argument("--N", default=20)
    parser.add_argument("--baseline_dataset", default="mp3d")
    parser.add_argument("--cut_environment", action="store_true")

    args = parser.parse_args(args=argv)
    return args


class CreateUnlabeledPositive:

    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.args = args
        self.TEST_ENVIRONMENT_MP3D = [f"pseudo_gt_batch{i}" for i in range(58)]
        self.TEST_ENVIRONMENT_HM3D = [f"pseudo_gt_batch{i}" for i in range(58)]

        self.model = ClipReverie(
            args.clip_base_model,
            self.device,
        ).cuda(self.device)

    @torch.no_grad()
    def calculate_embeddings(self, split, environments, file_output=None, is_hm3d=False):
        """
        Calculate embeddings for all text and images in the dataset
        """
        self.model.eval()
        embeddings_output = {}
        with torch.no_grad():
            print("start calculating embeddings")
            if self.args.cut_environment:
                environments = environments[1 : len(environments)]

            for env in environments:
                eval_dataset = dataloader.reverie_dataset(
                    self.model,
                    self.args,
                    split=split,
                    env=env,
                    num_bbox=16,
                    eval=True,
                    N=int(self.args.N),
                    baseline_dataset=self.args.baseline_dataset,
                    is_hm3d=is_hm3d,
                )
                if len(eval_dataset) == 0:
                    print(f"skip env {env} because there's no sample.")
                    continue
                eval_dataloader = DataLoader(eval_dataset, batch_size=1)

                _embeddings_output = []
                for (
                    all_image_features,
                    tokenized_instruction_clip,
                    tokenized_instruction_modified_clip,
                    tokenized_instruction_chatgpt_clip,
                    tokenized_np_clip,
                    gpt3_embeddings,
                    raw_instruction,
                    gt_img_id,
                    imageId_list,
                    instId,
                    llava_feature,
                    gpt4v_embeddings,
                ) in tqdm(eval_dataloader):
                    all_tokenized_instruction_clip = tokenized_instruction_clip.to(self.device).repeat(
                        all_image_features.shape[1], 1
                    )
                    all_tokenized_instruction_modified_clip = tokenized_instruction_modified_clip.to(
                        self.device
                    ).repeat(all_image_features.shape[1], 1)
                    all_tokenized_instruction_chatgpt_clip = tokenized_instruction_chatgpt_clip.to(self.device).repeat(
                        all_image_features.shape[1], 1
                    )
                    all_tokenized_np_clip = tokenized_np_clip.to(self.device).repeat(all_image_features.shape[1], 1, 1)
                    all_gpt3_embeddings = gpt3_embeddings.to(self.device).repeat(all_image_features.shape[1], 1)
                    all_gpt4v_embeddings = gpt4v_embeddings.to(self.device).squeeze(0)

                    _, image_embeddings, text_embeddings = self.model.calc_logits(
                        all_image_features.to(self.device).squeeze(0),
                        all_tokenized_instruction_clip,
                        all_tokenized_instruction_modified_clip,
                        all_tokenized_instruction_chatgpt_clip,
                        all_tokenized_np_clip,
                        all_gpt3_embeddings,
                        llava_feature.to(self.device).squeeze(0),
                        all_gpt4v_embeddings,
                        _eval=True,
                    )

                    # save image embeddings and text embeddings
                    if file_output:
                        gt_idx = [imageId_list.index(x) for x in gt_img_id]
                        _dump = {}
                        _dump["instruction_id"] = instId[0]
                        _dump["instruction"] = raw_instruction[0]
                        _dump["gt_image_id"] = gt_img_id
                        _dump["image_embeddings"] = image_embeddings.cpu().numpy().tolist()[gt_idx[0]]
                        _dump["text_embeddings"] = text_embeddings.cpu().numpy().tolist()[0]
                        _dump["batch_id"] = env
                        _embeddings_output.append(_dump)

                if file_output:
                    embeddings_output[env] = _embeddings_output

            if file_output:
                path = "./result/mp3d_embeddings.json" if not is_hm3d else "./result/hm3d_embeddings.json"
                with open(path, "w") as wf:
                    json.dump(embeddings_output, wf, indent=2, ensure_ascii=False)

        print("done")
        return True

    def create_dataset_for_pseudo_gt(self, database):
        envs_dict = {
            "pseudo_gt": ["pseudo_gt"],
        }

        batch_size = 100  # Each batch contains 100 items

        for split in envs_dict.keys():
            output_by_split = {}
            for env in envs_dict[split]:
                env_dataset = [data for data in database]
                sorted_env_dataset = sorted(env_dataset, key=lambda x: x["instruction_id"])
                total_batches = len(sorted_env_dataset) // batch_size + (len(sorted_env_dataset) % batch_size > 0)

                for batch in range(total_batches):
                    start_index = batch * batch_size
                    end_index = start_index + batch_size
                    batch_dataset = sorted_env_dataset[start_index:end_index]
                    if not batch_dataset:
                        continue

                    output_file = f"data/ltrpo_dataset/pseudo_gt/ltrpo_{split}_{env}_batch{batch}.json"
                    with open(output_file, "w") as wf:
                        json.dump(batch_dataset, wf, indent=2)

                    output_by_env = {}
                    for data in batch_dataset:
                        for i, bbox_id in enumerate(data["gt_bbox_id"]):
                            if bbox_id not in output_by_env:
                                dump = {
                                    "img_path": data["image_path"][i],
                                    "full_image_feature_path": data["full_image_feature_path"][i],
                                    "full_image_feature_path_2d": data["full_image_feature_path_2d"][i],
                                    "gpt4v_embeddings": data["gpt4v_embeddings"],
                                }
                                output_by_env[bbox_id] = dump
                    output_by_split.update(output_by_env)

                    # Create feature files for each batch
                    bboxId_list = []
                    image_features, image_features_2d, llava_features, gpt4v_features = [], [], [], []
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
                    base_path = f"data/eval_features/pseudo_gt/eval_features_{split}_{env}_batch{batch}"
                    np.save(f"{base_path}_full.npy", np.array(image_features))
                    np.save(f"{base_path}_full_2d.npy", np.array(image_features_2d))
                    np.save(f"{base_path}_llava.npy", np.array(llava_features))
                    np.save(f"{base_path}_gpt4v_embeddings.npy", np.array(gpt4v_features))

                    with open(f"{base_path}_bbox_list.json", "w") as wf:
                        json.dump(output_data, wf, indent=2)

            output_file = f"data/ltrpo_dataset/pseudo_gt/ltrpo_{split}_id2path.json"
            with open(output_file, "w") as wf:
                json.dump(output_by_split, wf, indent=2)

    @torch.no_grad()
    def create_cos_similarity(self, is_hm3d=False):
        """
        Calculate cosine similarity between text instructions and all image embeddings.
        """
        self.model.eval()
        output = {"output": []}

        if is_hm3d:
            with open("./result/hm3d_embeddings.json") as rf:
                embeddings = json.load(rf)
        else:
            with open("./result/mp3d_embeddings.json") as rf:
                embeddings = json.load(rf)

        all_image_embeddings = []
        all_image_ids = []

        for env in embeddings:
            for inst in embeddings[env]:
                all_image_embeddings.append(inst["image_embeddings"])
                all_image_ids.append(inst["gt_image_id"][0])

        all_image_embeddings = np.array(all_image_embeddings)

        for env in embeddings:
            print(f"==========  {env} ===========")
            for inst in embeddings[env]:
                instruction_id = inst["instruction_id"]
                instruction = inst["instruction"]
                text_embedding = np.array(inst["text_embeddings"])

                cos_similarities = self.calc_cos_similarity(
                    torch.tensor(all_image_embeddings), torch.tensor(text_embedding)
                )

                cos_similarities = cos_similarities.flatten()

                similarities = []
                for i, img_id in enumerate(all_image_ids):
                    similarities.append({"gt_image_id": img_id, "cos_similarity": cos_similarities[i]})

                unique_gt_image_ids = set()
                top20_image_ids = []
                for idx in np.argsort(cos_similarities)[::-1]:
                    gt_image_id = all_image_ids[idx][0]
                    if gt_image_id not in unique_gt_image_ids:
                        unique_gt_image_ids.add(gt_image_id)
                        top20_image_ids.append(gt_image_id)
                    if len(top20_image_ids) >= 20:
                        break

                output["output"].append(
                    {
                        "instruction_id": instruction_id,
                        "instruction": instruction,
                        # "similarity": similarities,
                        "top20": top20_image_ids,
                    }
                )

        if is_hm3d:
            with open("./result/hm3d_cos_similarity_output.json", "w") as wf:
                json.dump(output, wf, indent=4)
        else:
            with open("./result/mp3d_cos_similarity_output.json", "w") as wf:
                json.dump(output, wf, indent=4)

    def calc_cos_similarity(self, image_embeddings, text_embeddings):
        """calculate cosine similarity between image and text embeddings"""
        logits = self.model.calc_logits_from_embeddings(
            image_embeddings.to(self.device),
            text_embeddings.to(self.device),
        )
        score = logits.detach().cpu().numpy().copy()
        return score


def main(args):
    create_unlabeled_positive = CreateUnlabeledPositive(args)

    # Create dataset for pseudo gt
    # with open("data/ltrpo_dataset/pseudo_gt/pseudo_gt.json") as rf:
    #     database = json.load(rf)
    # create_unlabeled_positive.create_dataset_for_pseudo_gt(database)

    # Calculate embeddings
    # create_unlabeled_positive.calculate_embeddings(
    #     split="pseudo_gt",
    #     environments=create_unlabeled_positive.TEST_ENVIRONMENT_MP3D,
    #     file_output=True,
    #     is_hm3d=False,
    # )
    # create_unlabeled_positive.calculate_embeddings(
    #     split="pseudo_gt",
    #     environments=create_unlabeled_positive.TEST_ENVIRONMENT_HM3D,
    #     file_output=True,
    #     is_hm3d=True,
    # )

    # Calculate cosine similarity
    create_unlabeled_positive.create_cos_similarity(is_hm3d=False)
    create_unlabeled_positive.create_cos_similarity(is_hm3d=True)


if __name__ == "__main__":
    args = parse_args()
    main(args)
