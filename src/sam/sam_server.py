#!/usr/bin/env python
"""
SAMをFastAPIで提供するモジュール
"""
import argparse
import json
import logging
import sys

import cv2
import numpy as np
from sam.segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

import callback_server
import sam.image_plotter as iplt


class SimpleSegmentAnything:
    """SegmentAnythingをFastAPIで提供するサーバ"""

    def __init__(self):
        self.logger = self.get_logger()
        self.sam = None
        self.device = None
        self.mask_generator = None
        self.predictor = None
        self.device = "cuda"
        np.random.seed(seed=0)

    @staticmethod
    def get_logger():
        """本クラス向けのloggerを生成する"""
        logger = logging.getLogger("sam_server")
        logger.setLevel(logging.INFO)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(logging.Formatter("%(levelname)s,sam: %(message)s"))
        logger.addHandler(stream_handler)

        return logger

    def __predict_overlaid(self, image):
        """
        一発打ちを実行する(overlaidモード)

        fileに対して自動でセグメンテーションを付与し対象を透過maskで描画
        """
        self.logger.info("start predict_overlaid")

        masks = self.mask_generator.generate(image)
        processed_masks = self.process_anns(masks)
        for mask in masks:
            mask.pop("segmentation", None)
        ret = {"info": masks, "overlaid": self.merge_masks(image, processed_masks)}
        self.logger.info("end predict_overlaid")
        return ret

    def __predict_with_prompt(self, image, json_data):
        """
        一発打ちを実行する(with_promptモード)

        SamPredictorを利用し、指定した条件のmaskを取得
        """
        self.logger.info("start predict_with_prompt")
        self.predictor.set_image(image)

        point, label, bbox = self.extract_prompt(json_data)
        self.logger.info(f"prompt point={point}")
        self.logger.info(f"prompt label={label}")
        self.logger.info(f"prompt bbox={bbox}")

        masks, score, _ = self.predictor.predict(
            point_coords=point, point_labels=label, box=bbox, multimask_output=False
        )

        ret = None
        if masks.size > 0 and score.size > 0:
            ret = {
                "mask": masks[0].astype(np.uint8),
                "score": float(score[0]),
            }
        self.logger.info("end predict_with_prompt")
        return ret

    @staticmethod
    def merge_masks(image, masks):
        """
        imageに対して透過mask情報をマージする処理

        出力結果はuint8にキャストするが、内部ではfloatで扱う
        """
        overlaid = np.copy(image).astype(np.float32)
        for mask in masks:
            mask_rgb = mask[:, :, :3].astype(np.float32)
            mask_alpha = mask[:, :, 3].astype(np.float32) / 255  # アルファチャンネルを0-1の範囲に変換

            mask_alpha = np.expand_dims(mask_alpha, axis=-1)
            overlaid = (overlaid * (1 - mask_alpha)) + (mask_rgb * mask_alpha)

        return overlaid.astype(np.uint8)  # 最後にuint8にキャストして戻す

    @staticmethod
    def process_anns(anns):
        """透過マスク情報を付与する。"""
        processed_anns = []
        if len(anns) == 0:
            return processed_anns
        sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
        for ann in sorted_anns:
            seg = ann["segmentation"]
            img = np.ones((seg.shape[0], seg.shape[1], 3))
            color_mask = (np.random.random((1, 3)) * 255).astype(np.uint8).tolist()[0]
            for i in range(3):
                img[:, :, i] = color_mask[i]
            # imgとアルファチャンネルを結合し、uint8にキャスト
            processed_anns.append(np.dstack((img.astype(np.uint8), (seg * 0.35 * 255).astype(np.uint8))))
        return processed_anns

    def load_model(self, sam_checkpoint, model_type="vit_h"):
        """モデルをロードする"""
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=self.device)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)
        self.predictor = SamPredictor(self.sam)

    def predict_oneshot(self, file, prompt=None):
        """predict実行判定と実行する"""
        image = np.asarray(bytearray(file.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if prompt:
            return self.__predict_with_prompt(image, prompt)
        return self.__predict_overlaid(image)

    def extract_prompt(self, json_data):
        """
        dictからpromptに必要なデータを取得する。
        """
        if json_data["points"] is None and json_data["bbox"] is None:
            raise ValueError("Either 'points' or 'bbox' in the prompt must be set the valid value.")

        coords, labels = None, None
        if json_data["points"]:
            coords = np.array(json_data["points"]["coords"])
            labels = np.array(json_data["points"]["labels"])

        bbox = np.array(json_data["bbox"]) if json_data["bbox"] else None
        return (coords, labels, bbox)


def main():
    """コマンドライン引数をパースして、SAMサーバを開始する"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="path to model (checkpoint)")
    parser.add_argument("--model_type", type=str, required=True, help="model_type of segment-anything")
    subparsers = parser.add_subparsers(help="sub-command help", required=True)

    # oneshot mode
    oneshot_parser = subparsers.add_parser("oneshot", help="inference one image wiht Segment Anything.")
    oneshot_parser.set_defaults(mode="oneshot")
    oneshot_parser.add_argument(
        "--oneshot_image_file", type=str, required=True, help="a file path of the image to inference."
    )
    oneshot_parser.add_argument(
        "--output", "-o", type=str, required=True, help="a file path for the resulting inference image."
    )

    # start_server mode
    sserver_parser = subparsers.add_parser("start_server", help="start Segment Anything.")
    sserver_parser.set_defaults(mode="start_server")
    sserver_parser.add_argument("--server_config", type=str, required=True, help="a path to config file of server")

    args = parser.parse_args()

    segment_anything = SimpleSegmentAnything()
    segment_anything.load_model(args.model_path, args.model_type)

    if args.mode == "oneshot":
        with open(args.oneshot_image_file, "rb") as file:
            images = segment_anything.predict_oneshot(file)
        drawer = iplt.ImagePlotter()
        drawer.set_image(images["overlaid"])
        drawer.save(args.output)
        drawer.show()
    elif args.mode == "start_server":
        with open(args.server_config, "r", encoding="utf-8") as file:
            server_config = json.load(file)
        srv = callback_server.CallbackServer(server_config)
        srv.start_server(segment_anything.predict_oneshot)


if __name__ == "__main__":
    main()
