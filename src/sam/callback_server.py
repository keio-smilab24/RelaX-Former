#!/usr/bin/env python
"""
callbackサーバを起動する
"""
import base64
import json

import cv2
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse


class CallbackServer:
    """一発打ちのためのcallback処理を定義し、サーバを起動するクラス"""

    def __init__(self, conf):
        self.conf = conf

    @staticmethod
    def encode_npimg_to_base64(img):
        """Base64エンコーディングされたPNG画像の文字列に変換する関数"""
        encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
        _, encimg = cv2.imencode(".png", img, encode_param)
        return base64.b64encode(encimg).decode("utf-8")

    def start_server(self, callback):
        """callback関数を定義して、サーバを起動する"""
        fapi = FastAPI()

        @fapi.post("/oneshot/overlaid", responses={200: {"content": {"application/json": {"example": {}}}}})
        def execute_oneshot(input_img: UploadFile = File(..., description="入力画像")):
            """SAMの検出結果を返す"""
            ret = callback(input_img.file)
            ret["overlaid"] = self.encode_npimg_to_base64(ret["overlaid"])
            return JSONResponse(content=ret)

        @fapi.post("/oneshot/with_prompt", responses={200: {"content": {"application/json": {"example": {}}}}})
        def execute_prompt(
            input_img: UploadFile = File(..., description="入力画像"),
            prompt: UploadFile = File(..., description="SamPredictor.predictにおける引数情報を持つjsonファイル"),
        ):
            """SAMの検出結果を返す"""
            # 文字列をデコードしてJSONデータに変換する
            file_contents = prompt.file.read()
            json_data = json.loads(file_contents.decode("utf-8"))
            ret = callback(input_img.file, json_data)
            ret["mask"] = self.encode_npimg_to_base64(ret["mask"] * 255)
            return JSONResponse(content=ret)

        uvicorn.run(fapi, host=self.conf["server"]["hostname"], port=int(self.conf["server"]["port"]))
