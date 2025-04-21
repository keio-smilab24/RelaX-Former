"""
Text-Image Retrieval

TO ADD :
- Gradient Checkpointing
- Filter out bias from weight decay
- Decaying learning rate with cosine schedule
- Half-precision Adam statistics
- Half-precision stochastically rounded text encoder weights were used

Note:
1. BATCH_SIZE must larger than 1
"""

import argparse
import itertools
import json
import os
import warnings

import faiss
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

import callback_server
import clip
import dataloader
import performance_timer
import wandb
from double_relaxed_contrastive_loss import DoubleRelaxedContrastiveLoss
from model import ClipReverie
from openai_api import AskToOpenaiApiEmbeddings, AskToOpenaiChatCompletion
from stanford_parser import get_all_np, get_stanford_parser_proc

warnings.simplefilter("ignore")


def parse_args(argv=None):
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str)

    parser.add_argument("--log_wandb", action="store_true")
    parser.add_argument("--wandb_name", "-w", default="")

    parser.add_argument("--clip_base_model", default="ViT-L/14")

    parser.add_argument("--frcnn", action="store_true")
    parser.add_argument("--bbox", action="store_true")
    parser.add_argument("--num_bbox", default=16)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", default="5e-4")
    # change batch size
    parser.add_argument("--bs", default=128)
    parser.add_argument("--epochs", default=20)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval_metric", default="recall10")
    parser.add_argument("--eval_test", action="store_true")
    parser.add_argument("--eval_seen", action="store_true")
    parser.add_argument("--eval_unseen", action="store_true")

    parser.add_argument("--model_output_prefix", default="model/model_tir")
    parser.add_argument("--infer_model_path", default=None)
    # results/id4024 (no .json)  # 現状不使用
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--server_config", default="config/server_config.json")

    parser.add_argument("--N", default="30")
    # target or destination
    parser.add_argument("--baseline_dataset", default="")
    parser.add_argument("--cut_environment", action="store_true")
    parser.add_argument("--no_2d", action="store_true")

    parser.add_argument("--profiling", action="store_true")
    parser.add_argument("--use_faiss", action="store_true")
    parser.add_argument("--use_openai_api", action="store_true")

    parser.add_argument("--show_torchinfo", action="store_true")

    parser.add_argument("--alpha", type=float, default="0.7")
    parser.add_argument("--lambda_neg", type=float, default="0.7")
    parser.add_argument("--gamma", type=float, default="0.7")

    parser.add_argument("--use_offline_data", action="store_true")

    args = parser.parse_args(args=argv)

    if args.wandb_name != "":
        args.log_wandb = True

    if not args.infer_model_path:
        args.infer_model_path = args.model_output_prefix + "_best.pth"

    if args.mode == "start_server":
        args.use_faiss = True

    return args


class TextImageRetrievalMain:
    """Text-Image Retrievalを実行するクラス"""

    VAL_UNSEEN_ENVIRONMENT_MP3D = ["oLBMNvg9in8", "zsNo4HB9uLZ"]
    TEST_ENVIRONMENT_MP3D = ["QUCTc6BB5sX", "X7HyMhZNoso"]
    # TEST_ENVIRONMENT_MP3D = [f"pseudo_gt_batch{i}" for i in range(58)]
    VAL_UNSEEN_ENVIRONMENT_HM3D = [
        "TEEsavR23oF-W7k2QWzBrFY-7UrtFsADwob-bxsVRursffK",
        "66seV3BWPoX-L53DsRRk4Ch-HaxA7YrQdEC-yr17PDCnDDW-mma8eWq3nNQ-XNeHsjL6nBB",
        "4ok3usBNeis-rsggHU7g7dh-cvZr5TUy5C5",
        "c5eTyR3Rxyh-LT9Jq6dN3Ea-h1zeeAwLh9Z-7MXmsvcQjpJ-q5QZSEeHe5g-BAbdmeyTvMZ",
        "Qpor2mEya8F-cYkrGrCg2kB-7Ukhou1GxYi",
        "58NLZxWBSpk-MHPLjHsuG27-bzCsHPLDztK-5cdEh9F2hJL-hyFzGGJCSYs-u1bkiGRVyu9",
        "nrA1tAA17Yp-LEFTm3JecaC-fsQtJ8t3nTf",
        "5jp3fCRSRjc-k1cupFYWXJ6-svBbv1Pavdk-F7EAMsdDASd-kJJyRFXVpx2",
        "XMHNu9rRQ1y-BFRyYbPCCPE-AWUFxHEyV3T-z9YwN9M8FpG",
    ]
    TEST_ENVIRONMENT_HM3D = [
        "VBzV5z6i1WS-7GAhQPFzMot-3t8DB4Uzvkt-y9hTuugGdiq-tQ5s4ShP627",
        "mL8ThkuaVTM-CrMo8WxCyVb-X7gTkoDHViv",
        "GLAQ4DNUx5U-QHhQZWdMpGJ-T6nG3E2Uui9",
        "FnSn2KSrALj-bCPU9suPUw9",
        "6D36GQHuP8H-YRUkbU5xsYj-LNg5mXe1BDj-Nfvxx8J5NCo-HMkoS756sz6-SByzJLxpRGn",
        "wcojb4TFT35-ziup5kvtCCR-a8BtkwhxdRV-dHwjuKfkRUR-jgPBycuV1Jq-6s7QHgap2fW",
        "rXXL6twQiWc-hkr2MGpHD6B-q3hn1WQ12rz-u8ug2rtNARf-uSKXQ5fFg6u-DYehNKdT76V",
        "eF36g7L6Z9M-XB4GS9ShBRE-QaLdnwvtxbs-vBMLrTe4uLA",
        "uLz9jNga3kC-vd3HHTEpmyA-dVW2D7TDctW-X4qjx5vquwH-rJhMRvNn4DS",
    ]
    # TEST_ENVIRONMENT_HM3D = [f"pseudo_gt_batch{i}" for i in range(59)]
    REAL_TEST_ENVIRONMENT = [
        "real_20230118T172959",
        "real_20230118T173924",
        "real_20230119T153429",
        "real_20230119T154113",
        "real_20230119T155732",
        "real_20230119T160333",
        "real_20230119T161046",
        "real_20230119T161732",
        "real_20230119T162429",
        "real_20230119T163059",
    ]

    def __init__(self, args_):
        self.faiss_index = None
        self.device = "cuda:0"
        self.args = args_

        self.model = ClipReverie(
            args_.clip_base_model,
            self.device,
        ).cuda(self.device)

        if self.args.use_faiss:
            inner_product_dimension = self.model.fc2.out_features
            self.faiss_index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(), 0, faiss.IndexFlatIP(inner_product_dimension)
            )
        if self.args.use_openai_api:
            self.openai_embeddings = AskToOpenaiApiEmbeddings()
            self.openai_completion = AskToOpenaiChatCompletion()

    def load_model(self, path):
        """モデルをロードする"""
        self.model.load_state_dict(torch.load(path))
        print(f"model file was loaded from {path}.")

    def save_model(self, path):
        """モデルを保存する"""
        torch.save(self.model.state_dict(), path)

    def train_model(self):
        """学習を実行する"""
        print("Currently loading train dataset ... ")
        train_dataset = dataloader.reverie_dataset(
            self.model,
            self.args,
            split="train",
            env="train",
            eval=False,
            N=int(self.args.N),
            baseline_dataset=self.args.baseline_dataset,
        )
        train_dataloader = DataLoader(train_dataset, batch_size=int(self.args.bs))

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.args.lr),
            betas=(0.9, 0.98),
            eps=1e-6,
            weight_decay=0.2,
        )  # Params from paper

        best_score = 0
        best_epoch = 0
        for epoch in range(int(self.args.epochs)):
            print(f"\n==== Epoch {epoch} ================")
            loss = self.train_epoch(train_dataloader, optimizer, frcnn_flag=self.args.frcnn)
            print(f"Epoch: {epoch},  Loss: {loss:.4f}")

            if self.args.eval_unseen:
                # MP3D
                val_unseen_mp = self.evaluate("val_unseen", self.VAL_UNSEEN_ENVIRONMENT_MP3D)
                val_unseen_mrr_mp = val_unseen_mp[0]
                val_unseen_recall1_mp = val_unseen_mp[1]
                val_unseen_recall5_mp = val_unseen_mp[2]
                val_unseen_recall10_mp = val_unseen_mp[3]
                val_unseen_recall20_mp = val_unseen_mp[4]
                print(", ".join([f"{value:.2f}" for value in val_unseen_mp]))

                # HM3D
                val_unseen_hm = self.evaluate("val_unseen", self.VAL_UNSEEN_ENVIRONMENT_HM3D, is_hm3d=True)
                val_unseen_mrr_hm = val_unseen_hm[0]
                val_unseen_recall1_hm = val_unseen_hm[1]
                val_unseen_recall5_hm = val_unseen_hm[2]
                val_unseen_recall10_hm = val_unseen_hm[3]
                val_unseen_recall20_hm = val_unseen_hm[4]
                print(", ".join([f"{value:.2f}" for value in val_unseen_hm]))

                if self.args.eval_metric == "recall10+mrr":
                    eval_score = val_unseen_recall10_mp + val_unseen_mrr_mp + val_unseen_recall10_hm + val_unseen_mrr_hm
                elif self.args.eval_metric == "recall10+recall5":
                    eval_score = (
                        val_unseen_recall10_mp + val_unseen_recall5_mp + val_unseen_recall10_hm + val_unseen_recall5_hm
                    )
                elif self.args.eval_metric == "recall10":
                    eval_score = val_unseen_recall10_mp + val_unseen_recall10_hm
                elif self.args.eval_metric == "recall5":
                    eval_score = val_unseen_recall5_mp + val_unseen_recall5_hm
                elif self.args.eval_metric == "recall1":
                    eval_score = val_unseen_recall1_mp + val_unseen_recall1_hm
                elif self.args.eval_metric == "mrr":
                    eval_score = val_unseen_mrr_mp + val_unseen_mrr_hm

                if eval_score > best_score:
                    best_epoch = epoch
                    best_score = eval_score
                    model_path = f"{self.args.model_output_prefix}_best.pth"
                    print(f"save model file as best.: {model_path}")
                    self.save_model(model_path)

                if self.args.log_wandb:
                    wandb.log(
                        {
                            "training loss per epoch": loss,
                            "val_unseen_mp": {
                                "mrr": val_unseen_mp[0],
                                "R@1": val_unseen_mp[1],
                                "R@5": val_unseen_mp[2],
                                "R@10": val_unseen_mp[3],
                                "R@20": val_unseen_mp[4],
                            },
                            "val_unseen_hm": {
                                "mrr": val_unseen_hm[0],
                                "R@1": val_unseen_hm[1],
                                "R@5": val_unseen_hm[2],
                                "R@10": val_unseen_hm[3],
                                "R@20": val_unseen_hm[4],
                            },
                        }
                    )

            self.save_model(f"{self.args.model_output_prefix}_{epoch:03d}.pth")

            # TODO: 一旦機能停止
            # if self.args.eval_test:
            #     test_result = self.evaluate("test", self.TEST_ENVIRONMENT_MP3D)
            #     print(", ".join([f"{value:.2f}" for value in test_result]))

        print(f"\n==== RESULTS for best epoch {best_epoch} =====")
        self.load_model(self.args.infer_model_path)
        self.test_model()

    def test_model(self):
        """評価を実行する"""
        header = "mrr, recall1, recall5, recall10, recall20 = "

        # MP3D
        # TODO: idを逐次変更
        test_result_mp = self.evaluate(
            "test", self.TEST_ENVIRONMENT_MP3D, file_output=f"result/{self.args.wandb_name}_mp3d"
        )
        # test_result_mp = self.evaluate(
        #     "pseudo_gt", self.TEST_ENVIRONMENT_MP3D, file_output=f"result/{self.args.wandb_name}_mp3d"
        # )
        print(header + ", ".join([f"{val:.2f}" for val in test_result_mp]))

        # HM3D
        # TODO: idを逐次変更
        test_result_hm = self.evaluate(
            "test", self.TEST_ENVIRONMENT_HM3D, file_output=f"result/{self.args.wandb_name}_hm3d", is_hm3d=True
        )
        # test_result_hm = self.evaluate(
        #     "pseudo_gt", self.TEST_ENVIRONMENT_HM3D, file_output=f"result/{self.args.wandb_name}_hm3d", is_hm3d=True
        # )
        print(header + ", ".join([f"{val:.2f}" for val in test_result_hm]))

        if self.args.log_wandb:
            wandb.log(
                {
                    "test_mp": {
                        "mrr": test_result_mp[0],
                        "R@1": test_result_mp[1],
                        "R@5": test_result_mp[2],
                        "R@10": test_result_mp[3],
                        "R@20": test_result_mp[4],
                    },
                    "test_hm": {
                        "mrr": test_result_hm[0],
                        "R@1": test_result_hm[1],
                        "R@5": test_result_hm[2],
                        "R@10": test_result_hm[3],
                        "R@20": test_result_hm[4],
                    },
                }
            )

    def test_model_real(self):
        """評価を実行する"""
        header = "mrr, recall1, recall5, recall10, recall20 = "
        test_result = self.evaluate("real_test", self.REAL_TEST_ENVIRONMENT)
        print(header + ", ".join([f"{val:.2f}" for val in test_result]))

    def predict_oneshot(self, image, instruction, mode, is_offline=False, offline_data_path=None):
        """一発打ちを実行する"""
        self.model.eval()

        if is_offline and offline_data_path:
            image_feature_path = f"{offline_data_path}/image_feature.npy"
            llava_feature_path = f"{offline_data_path}/llava_feature.npy"
            seem_feature_path = f"{offline_data_path}/seem_feature.npy"
            with open(image_feature_path, "rb") as rf:
                image_feature = np.load(rf)
                image_feature = torch.tensor(image_feature).to(self.device)
            with open(llava_feature_path, "rb") as rf:
                llava_feature = np.load(rf)
                llava_feature = torch.tensor(llava_feature).to(self.device)
            with open(seem_feature_path, "rb") as rf:
                seem_feature = np.load(rf)
                seem_feature = torch.tensor(seem_feature).to(self.device)
            image_embeddings = self.model.image_encoder(image_feature, llava_feature, seem_feature)
        else:
            image_embeddings = self.embed_image(image)
        text_embeddings = self.embed_instruction(instruction, mode)

        with performance_timer.get_timer("forward with model", self.args.profiling):
            score = self.calc_cos_similarity(image_embeddings, text_embeddings)
        return score

    def embed_image(self, image, np_return=False):

        import requests

        """画像をTIR向けにembeddingする"""
        server_ip = os.environ.get("EMBED_IMAGE_SERVER_IP")
        server_port = os.environ.get("EMBED_IMAGE_SERVER_PORT", 5000)
        server_base_url = f"http://{server_ip}:{server_port}"
        files = {"image": image}

        image_features = requests.post(f"{server_base_url}/embed_all", files=files)
        image_features = image_features.json()
        vit_feature = torch.tensor(image_features["vit_features"]).to(self.device)
        clip_feature = torch.tensor(image_features["clip_features"]).to(self.device)
        llava_feature = torch.tensor(image_features["llava_features"]).to(self.device).unsqueeze(0)
        sam_feature = torch.tensor(image_features["sam_features"]).to(self.device)
        seem_feature = torch.tensor(image_features["seem_features"]).to(self.device)

        image_feature = torch.stack([clip_feature, sam_feature, vit_feature], dim=1)
        image_embeddings = self.model.image_encoder(image_feature, llava_feature, seem_feature)

        if np_return:
            return image_embeddings.to("cpu").detach().numpy().copy()
        return image_embeddings

    def embed_instruction(self, instruction, mode, np_return=False):
        """指示文をTIR向けにembeddingする"""
        if hasattr(instruction, "read"):
            raw_st_instruction = json.load(instruction)["instruction"]
        elif isinstance(instruction, str):
            raw_st_instruction = instruction
        else:
            raise RuntimeError("unsupported type for argument 'instruction' of function 'predict_oneshot'")

        with performance_timer.get_timer("get information from ChatGPT", self.args.profiling):
            chat_gpt_data = self.interpret_with_chat_gpt(raw_st_instruction)

        with performance_timer.get_timer("compose data about instruction", self.args.profiling):
            raw_instruction = mode + " " + raw_st_instruction
            if mode == "<target>":
                raw_instruction_chatgpt = chat_gpt_data["instruction_chatgpt"]
            elif mode == "<destination>":
                raw_instruction_chatgpt = chat_gpt_data["llm_data"]["destination"]
            else:
                raise RuntimeError(f"unknown mode for {mode}")

            if raw_instruction_chatgpt == "NA":
                raw_instruction_chatgpt = ""
                modified_instruction = raw_instruction
            else:
                modified_instruction = mode + " " + chat_gpt_data["llm_data"]["modified_instruction"]

            with performance_timer.get_timer("tokenize with clip tokenizer", self.args.profiling):
                tokenized_instruction_clip = clip.tokenize(raw_instruction)
                tokenized_instruction_modified_clip = clip.tokenize(modified_instruction)
                tokenized_instruction_chatgpt_clip = clip.tokenize("A photo of " + raw_instruction_chatgpt)
            with performance_timer.get_timer("get noun phrase with stanford parser", self.args.profiling):
                tokenized_np_clip = dataloader.reverie_dataset.get_noun_phrases_clip(
                    list(get_all_np(raw_st_instruction)), n=self.model.N
                )
            gpt3_embeddings = torch.tensor(chat_gpt_data["gpt3_embeddings"])

        text_embeddings = self.model.text_encoder(
            tokenized_instruction_clip.to(self.device),
            tokenized_instruction_modified_clip.to(self.device),
            tokenized_instruction_chatgpt_clip.to(self.device),
            tokenized_np_clip.unsqueeze(0).to(self.device),
            gpt3_embeddings.unsqueeze(0).to(self.device),
        )

        if np_return:
            return text_embeddings.to("cpu").detach().numpy().copy()
        return text_embeddings

    def calc_cos_similarity(self, image_embeddings, text_embeddings):
        """TIRモデルのインファレンスを実行する関数"""
        logits = self.model.calc_logits_from_embeddings(
            image_embeddings.to(self.device),
            text_embeddings.to(self.device),
        )
        # score = logits.detach().cpu().numpy().tolist()[0]
        score = logits.detach().cpu().numpy().copy()
        # return score[0]
        return score

    def calc_cos_similarity_with_faiss(self, image_embeddings, text_embeddings, num_k=10):
        """TIRモデルのインファレンスを実行する関数"""
        self.faiss_index.reset()
        self.faiss_index.add(image_embeddings)
        distance, indice = self.faiss_index.search(text_embeddings, num_k)
        ret = [None] * image_embeddings.shape[0]
        for dist, idx in zip(distance[0], indice[0]):
            if idx >= 0:
                ret[idx] = float(dist)
        return ret

    def interpret_with_chat_gpt(self, inst):
        """ChatGPTに解釈させたデータを生成する"""
        if self.args.use_openai_api:
            embeddings = self.openai_embeddings.process_instruction(inst)
            target, dest = self.openai_completion.process_instruction(inst)
            ret = {
                "instruction_chatgpt": target,
                "llm_data": {"destination": dest, "modified_instruction": f"Carry {target} to {dest}"},
                "gpt3_embeddings": embeddings,
            }

        else:
            ret = {
                "instruction_chatgpt": "unknown",
                "llm_data": {"destination": "unknow", "modified_instruction": "Carry unknown to unknow"},
                "gpt3_embeddings": [0.0] * 1536,
            }

        return ret

    def cross_entropy(self, preds, targets, reduction="none"):
        """Cross entropyを定義"""
        log_softmax = torch.nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()
        else:
            return loss

    def train_epoch(self, dataloader, optimizer, frcnn_flag=False):
        """1epoch分のtrainを実行する"""
        self.model.train()
        t_loss = 0
        n_ex = 0
        for (
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
        ) in tqdm(dataloader):
            optimizer.zero_grad()

            # batchごとに出力されてしまうので適当にctrl+c
            if self.args.show_torchinfo:
                summary(
                    self.model,
                    input_data=[
                        entire_image_feature.to(self.device),
                        tokenized_instruction_clip.to(self.device),
                        tokenized_instruction_modified_clip.to(self.device),
                        tokenized_instruction_chatgpt_clip.to(self.device),
                        tokenized_np_clip.to(self.device),
                        gpt3_embeddings.to(self.device),
                        llava_feature.to(self.device),
                        gpt4v_embeddings.to(self.device),
                    ],
                )

            logits, targets, image_embeddings, text_embeddings = self.model(
                entire_image_feature.to(self.device),
                tokenized_instruction_clip.to(self.device),
                tokenized_instruction_modified_clip.to(self.device),
                tokenized_instruction_chatgpt_clip.to(self.device),
                tokenized_np_clip.to(self.device),
                gpt3_embeddings.to(self.device),
                llava_feature.to(self.device),
                gpt4v_embeddings.to(self.device),
            )
            criterion = DoubleRelaxedContrastiveLoss(
                alpha=self.args.alpha, lambda_neg=self.args.lambda_neg, gamma_semi=self.args.gamma
            )
            loss = criterion(text_embeddings, image_embeddings, gt_img_ids, pseudo_gt_img_ids)

            # orignal cross_entropy loss

            # loss = self.cross_entropy(logits, targets).mean()
            # texts_loss = self.cross_entropy(logits, targets)
            # images_loss = self.cross_entropy(logits.T, targets.T)
            # loss = (images_loss + texts_loss) / 2.0  # shape: (batch_size)
            # loss = loss.mean()

            t_loss += loss
            loss.backward()
            optimizer.step()
            n_ex += 1

        return loss / n_ex
        # return t_loss

    def calc_score(self, probs, gt_id, imgId_list_full, eval_baseline=False):
        """Scoreを計算"""
        mrr, recall1, recall5, recall10, recall20 = 0, 0, 0, 0, 0

        ranks = []
        if not eval_baseline:
            imgId_list_full = [i[0] for i in imgId_list_full]

        # imgId_list = [i.split("_")[-1] for i in imgId_list_full]
        imgId_list = [i for i in imgId_list_full]

        for gt in gt_id:
            idx_list = [imgId_list.index(gt_item) for gt_item in gt]
            ranks_for_gt = [sorted(probs, reverse=True).index(probs[idx]) for idx in idx_list]
            rank = min(ranks_for_gt)
            while rank in ranks:  # 同じ値の場合に
                rank += 1
            ranks.append(rank)

        # find top 20
        top20 = []
        top20_rank = np.argsort(probs)[-20:][::-1]
        for i in top20_rank:
            top20.append(imgId_list_full[i])

        for i, rank in enumerate(sorted(ranks)):
            if rank < 20:
                recall20 += 1
            if rank < 10:
                recall10 += 1
            if rank < 5:
                recall5 += 1
            if rank < 1:
                recall1 += 1

            if i == 0:  # first time
                mrr = 100 / (rank + 1)

        recall20 = 100 * recall20 / len(ranks)
        recall10 = 100 * recall10 / len(ranks)
        recall5 = 100 * recall5 / len(ranks)
        recall1 = 100 * recall1 / len(ranks)

        return mrr, recall1, recall5, recall10, recall20, ranks, top20

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
                        "similarity": similarities,
                        "top20": top20_image_ids,
                    }
                )

        if is_hm3d:
            with open("./result/hm3d_cos_similarity_output.json", "w") as wf:
                json.dump(output, wf, indent=4)
        else:
            with open("./result/mp3d_cos_similarity_output.json", "w") as wf:
                json.dump(output, wf, indent=4)

    @torch.no_grad()
    def evaluate(self, split, environments, file_output=None, is_hm3d=False):
        """
        評価処理を実行

        返り値は、mrr, recall1, recall5, recall10, recall20の順。
        """
        self.model.eval()
        output = {}
        # embeddings_output = {}
        with torch.no_grad():
            mrr, recall1, recall5, recall10, recall20 = 0, 0, 0, 0, 0
            print(f"==========  {split.upper()} on {'HM3D' if is_hm3d else 'MP3D'} ===========")
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

                n_ex = 0
                env_mrr, env_recall1, env_recall5, env_recall10, env_recall20 = 0, 0, 0, 0, 0
                env_output = []
                # for pseudo_gt
                # _embeddings_output = []
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
                    # alpha_clip_feature,
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

                    logits_per_text, image_embeddings, text_embeddings = self.model.calc_logits(
                        all_image_features.to(self.device).squeeze(0),
                        all_tokenized_instruction_clip,
                        all_tokenized_instruction_modified_clip,
                        all_tokenized_instruction_chatgpt_clip,
                        all_tokenized_np_clip,
                        all_gpt3_embeddings,
                        llava_feature.to(self.device).squeeze(0),
                        all_gpt4v_embeddings,
                        # alpha_clip_feature.to(self.device).squeeze(0),
                        _eval=True,
                    )

                    # save image embeddings and text embeddings
                    # if file_output:
                    #     gt_idx = [imageId_list.index(x) for x in gt_img_id]
                    #     _dump = {}
                    #     _dump["instruction_id"] = instId[0]
                    #     _dump["instruction"] = raw_instruction[0]
                    #     _dump["gt_image_id"] = gt_img_id
                    #     _dump["image_embeddings"] = image_embeddings.cpu().numpy().tolist()[gt_idx[0]]
                    #     _dump["text_embeddings"] = text_embeddings.cpu().numpy().tolist()[0]
                    #     _dump["batch_id"] = env
                    #     _embeddings_output.append(_dump)

                    _mrr, _recall1, _recall5, _recall10, _recall20, ranks, top20 = self.calc_score(
                        np.diag(logits_per_text.cpu().numpy()), gt_img_id, imageId_list
                    )

                    if file_output:
                        dump = {}

                        dump["instruction_id"] = instId[0]
                        dump["instruction"] = raw_instruction[0]
                        # dump["gt_image_id"] = [x[0] for x in gt_img_id]
                        dump["gt_image_id"] = gt_img_id
                        dump["mrr"] = str(_mrr)
                        dump["ranks"] = [str(x) for x in ranks]
                        dump["top20"] = top20

                        env_output.append(dump)

                    n_ex += 1
                    env_mrr += _mrr
                    env_recall1 += _recall1
                    env_recall5 += _recall5
                    env_recall10 += _recall10
                    env_recall20 += _recall20

                mrr += env_mrr / n_ex
                recall1 += env_recall1 / n_ex
                recall5 += env_recall5 / n_ex
                recall10 += env_recall10 / n_ex
                recall20 += env_recall20 / n_ex

                print(
                    ", ".join(
                        [
                            f"num_inst : {n_ex}",
                            f"num_img : {len(imageId_list)} ... {env_mrr/n_ex:.2f}",
                            f"{env_recall1/n_ex:.2f}",
                            f"{env_recall5/n_ex:.2f}",
                            f"{env_recall10/n_ex:.2f}",
                            f"{env_recall20/n_ex:.2f}",
                        ]
                    )
                )

                if file_output:
                    output[env] = env_output
            #         embeddings_output[env] = _embeddings_output
            #
            # if file_output:
            #     path = "./result/mp3d_embeddings.json" if not is_hm3d else "./result/hm3d_embeddings.json"
            #     with open(path, "w") as wf:
            #         json.dump(embeddings_output, wf, indent=2, ensure_ascii=False)

            if file_output:
                path = f"{file_output}_{split}.json"
                with open(path, "w") as wf:
                    json.dump(output, wf, indent=2, ensure_ascii=False)

        n_envs = len(environments)
        return (mrr / n_envs, recall1 / n_envs, recall5 / n_envs, recall10 / n_envs, recall20 / n_envs)


def tir_main():
    """Text-Image Retrievalを実行する"""
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if args.log_wandb:
        wandb.init(project="clip-reverie", name=args.wandb_name)

    with performance_timer.get_timer("build model", args.profiling):
        tir = TextImageRetrievalMain(args)

    if args.mode == "train":
        tir.train_model()
    elif args.mode == "test":
        tir.load_model(args.infer_model_path)
        tir.test_model()
    elif args.mode == "real_test":
        tir.load_model(args.infer_model_path)
        tir.test_model_real()
    elif args.mode == "oneshot":
        with performance_timer.get_timer("load model", args.profiling):
            tir.load_model(args.infer_model_path)
            tir.model.eval()

        sample_data_prefix = os.path.join("scripts", "sample_data")
        posi_sample_img = os.path.join(sample_data_prefix, "posi_center.jpg")
        neg_sample_img = os.path.join(sample_data_prefix, "neg_center.jpg")
        instruction_path = os.path.join(sample_data_prefix, "instruction.json")

        if args.use_offline_data:
            result_str = ""
            for img_path, mode in itertools.product([posi_sample_img, neg_sample_img], ["<target>", "<destination>"]):
                with open(img_path, "rb") as img, open(instruction_path, "r", encoding="utf-8") as inst:
                    with performance_timer.get_timer("predict_oneshot", args.profiling):
                        with performance_timer.get_timer("embeding image", args.profiling):
                            score = tir.predict_oneshot(
                                img, inst, mode, is_offline=True, offline_data_path="./scripts/sample_data"
                            )
                    result_str += "\n{0:30}: {1:.04f}".format(os.path.basename(img_path) + "," + mode, score.item())

            print("=" * 20 + " RESULT " + "=" * 20 + result_str)
            return

        if args.use_faiss:
            with performance_timer.get_timer("embed instruction", args.profiling):
                with open(instruction_path, "r", encoding="utf-8") as inst:
                    text_embeddings = tir.embed_instruction(inst, "<target>", np_return=True)

            with performance_timer.get_timer("open and embed image files", args.profiling):
                image_embeddings = []
                for img_path in [posi_sample_img, neg_sample_img] * 3:
                    with open(img_path, "rb") as img:
                        image_embeddings.append(tir.embed_image(img_path, np_return=True))
                image_embeddings = np.concatenate(image_embeddings)

            with performance_timer.get_timer("calculate cos similarity", args.profiling):
                result = tir.calc_cos_similarity_with_faiss(image_embeddings, text_embeddings, 4)
            print(result)
            return

        result_str = ""
        for img_path, mode in itertools.product([posi_sample_img, neg_sample_img], ["<target>", "<destination>"]):
            with open(img_path, "rb") as img, open(instruction_path, "r", encoding="utf-8") as inst:
                with performance_timer.get_timer("predict_oneshot", args.profiling):
                    with performance_timer.get_timer("embeding image", args.profiling):
                        score = tir.predict_oneshot(img, inst, mode)
                result_str += "\n{0:30}: {1:.04f}".format(os.path.basename(img_path) + "," + mode, score.item())

        print("=" * 20 + " RESULT " + "=" * 20 + result_str)

    elif args.mode == "real_oneshot":
        import glob
        import json
        from collections import defaultdict

        # Load the model
        with performance_timer.get_timer("load model", args.profiling):
            tir.load_model(args.infer_model_path)
            tir.model.eval()
        results = defaultdict(lambda: defaultdict(list))
        # Process each environment
        for env_num in range(1, 5):
            # Read images
            image_paths = glob.glob(f"./data/real_dataset/env_info/{env_num}/*.png")
            print("image_paths:", image_paths)
            # Read queries
            with open(f"./data/real_dataset/env_info/{env_num}/query.txt", "r") as query_file:
                queries = query_file.read().splitlines()
                print("queries:", queries)
            # Process each query
            for query in queries:
                query_scores = []
                # Calculate similarity for each image
                for img_path in image_paths:
                    with open(img_path, "rb") as img:
                        with performance_timer.get_timer("predict_real_oneshot", args.profiling):
                            score = tir.predict_oneshot(img, query, "<target>")
                    query_scores.append((img_path, score))
                # Sort scores and get rankings
                ranked_scores = sorted(query_scores, key=lambda x: x[1], reverse=True)
                rankings = [img_path for img_path, _ in ranked_scores]
                # Store results
                results[env_num][query] = rankings
        # Write results to JSON file
        output_file = "real_oneshot_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

    elif args.mode == "start_server":
        tir.load_model(args.infer_model_path)
        tir.model.eval()

        with open(args.server_config, "r", encoding="utf-8") as server_conf:
            conf = json.load(server_conf)
        callback_server.start(
            conf, tir.predict_oneshot, tir.embed_image, tir.embed_instruction, tir.calc_cos_similarity_with_faiss
        )

    elif args.mode == "create_cos_similarity":
        tir.load_model(args.infer_model_path)
        tir.create_cos_similarity()
    else:
        raise RuntimeError(f"unknown mode of [{args.mode}]")


if __name__ == "__main__":
    with get_stanford_parser_proc():
        tir_main()
