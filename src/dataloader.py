"""REVERIEデータセットのデータローダモジュール"""

import json

import numpy as np
import torch
from torch.utils.data import Dataset

import clip


class reverie_dataset(Dataset):
    """REVERIEデータセットを管理するクラス"""

    def __init__(
        self, model, args, split="train", env="train", num_bbox=16, eval=False, N=30, baseline_dataset="", is_hm3d=False
    ):
        self.model = model
        self.args = args
        self.bbox = True
        self.num_bbox = num_bbox
        self.eval = eval
        self.N = N
        self.is_hm3d = is_hm3d

        if baseline_dataset != "":
            self.baseline = f"_baseline_{baseline_dataset}"
        else:
            self.baseline = ""

        dataset_path = f"data/ltrpo_dataset{self.baseline}/ltrpo_{split}_{env}.json"
        hm3d_datset_path = f"data/hm3d/hm3d_dataset{self.baseline}/hm3d_{split}_{env}.json"

        self.clip_model, self.preprocess_clip = clip.load("ViT-L/14", device="cuda:0")

        # val/test
        if self.eval:
            if self.is_hm3d:
                self.get_dataset(hm3d_datset_path)
                eval_features_path_base = f"data/hm3d/eval_features{self.baseline}/eval_features_{split}_{env}"
                eval_id_path = f"data/hm3d/eval_features{self.baseline}/eval_features_{split}_{env}_bbox_list.json"
            else:
                self.get_dataset(dataset_path)
                eval_features_path_base = f"data/eval_features{self.baseline}/eval_features_{split}_{env}"
                eval_id_path = f"data/eval_features{self.baseline}/eval_features_{split}_{env}_bbox_list.json"

            self.all_image_features = self.open_npy(f"{eval_features_path_base}_full.npy")
            self.imageId_list = json.load(open(eval_id_path, "r"))["bboxId_list"]
            self.all_llava_features = self.open_npy(f"{eval_features_path_base}_llava.npy")
            self.all_gpt4v_features = self.open_npy(f"{eval_features_path_base}_gpt4v_embeddings.npy")
            # self.all_alpha_clip_features = self.open_npy(f"{eval_features_path_base}_alpha.npy")

        # train
        else:
            self.get_dataset(dataset_path, optional_path=hm3d_datset_path)

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        mode = self.db[idx]["mode"]
        raw_instruction_chatgpt = "NA"
        if mode == "<destination>":
            mode = "<receptacle>"
        raw_instruction = mode + " " + self.db[idx]["instruction"]
        modified_instruction = mode + " " + self.db[idx]["llm_data"]["modified_instruction"]
        if mode == "<target>":
            raw_instruction_chatgpt = self.db[idx]["instruction_chatgpt"]
        elif mode == "<receptacle>":
            raw_instruction_chatgpt = self.db[idx]["llm_data"]["destination"]

        if raw_instruction_chatgpt == "NA":
            raw_instruction_chatgpt = ""
            modified_instruction = raw_instruction

        # npy or 直書きの両方に対応
        if type(self.db[idx]["gpt3_embeddings"]) is str:
            gpt3_embeddings = torch.from_numpy(
                self.open_npy(self.db[idx]["gpt3_embeddings"]).astype(np.float32)
            ).clone()
        else:
            gpt3_embeddings = torch.tensor(self.db[idx]["gpt3_embeddings"])

        gpt4v_embeddings = self.open_npy((self.db[idx]["gpt4v_embeddings"]))
        llava_feature = self.open_npy(
            self.db[idx]["full_image_feature_path"][0].replace("features", "llava_features_latent_full_bit")
        )

        tokenized_instruction_clip = clip.tokenize(raw_instruction).squeeze(0)
        tokenized_instruction_modified_clip = clip.tokenize(modified_instruction).squeeze(0)
        tokenized_instruction_chatgpt_clip = clip.tokenize("A photo of " + raw_instruction_chatgpt).squeeze(0)

        tokenized_np_clip = self.get_noun_phrases_clip(self.db[idx]["noun_phrases"], n=self.N)

        gt_img_ids = self.db[idx]["gt_bbox_id"]

        if not self.eval:
            entire_image_feature = self.open_npy(self.db[idx]["full_image_feature_path"][0])

            # Ensure each rank from 0 to 19 is filled, if not, add an empty string
            complete_pseudo_gt = {rank: "" for rank in range(20)}
            for pseudo_gt in self.db[idx]["pseudo_gt"]:
                complete_pseudo_gt[pseudo_gt["rank"]] = pseudo_gt["gt_img_id"]
            pseudo_gt_img_ids = [complete_pseudo_gt[rank] for rank in range(20)]

            ret = (
                entire_image_feature,
                tokenized_instruction_clip,
                tokenized_instruction_modified_clip,
                tokenized_instruction_chatgpt_clip,
                tokenized_np_clip,
                gpt3_embeddings,
                llava_feature,
                gpt4v_embeddings,
                gt_img_ids,
                pseudo_gt_img_ids,
                # alpha_clip_feature,
            )
        else:
            instId = self.db[idx]["instruction_id"]
            ret = (
                self.all_image_features,
                tokenized_instruction_clip,
                tokenized_instruction_modified_clip,
                tokenized_instruction_chatgpt_clip,
                tokenized_np_clip,
                gpt3_embeddings,
                raw_instruction,
                gt_img_ids,
                self.imageId_list,
                instId,
                self.all_llava_features,
                self.all_gpt4v_features,
                # self.all_alpha_clip_features,
            )

        return ret

    @staticmethod
    def get_unique_np(noun_phrases):
        """名詞句リストから重複を削除する"""
        unique_phrases = []
        for phrase in sorted(noun_phrases, key=len, reverse=True):
            if phrase not in " ".join(unique_phrases):
                unique_phrases.append("A photo of " + phrase)
        return unique_phrases

    @classmethod
    def get_noun_phrases_clip(cls, noun_phrases, n=1):
        """名詞句をtensorとして取得"""
        noun = cls.get_unique_np(noun_phrases)
        tokenized_noun_phrases_clip = clip.tokenize(noun)

        # id 4019 : np を n つに制限
        if tokenized_noun_phrases_clip.shape[0] > n:
            return tokenized_noun_phrases_clip[:n, :].to(torch.int32)

        tokenized_noun_phrases_clip = torch.cat(
            [
                tokenized_noun_phrases_clip,
                torch.zeros(n - tokenized_noun_phrases_clip.shape[0], tokenized_noun_phrases_clip.shape[1]),
            ],
            dim=0,
        )
        return tokenized_noun_phrases_clip.to(torch.int32)

    def get_dataset(self, dataset_path, optional_path=None):
        """JSON形式のデータセット情報を読み出す"""
        if optional_path is not None:
            self.db = json.load(open(dataset_path, "r")) + json.load(open(optional_path, "r"))
        else:
            self.db = json.load(open(dataset_path, "r"))

    def open_npy(self, path):
        """numpy形式データを読み出す"""
        return np.load(path, allow_pickle=True)
