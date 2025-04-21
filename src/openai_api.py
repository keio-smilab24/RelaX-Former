"""OpenAI APIにアクセスして、指示文の処理を行うモジュール"""

import os
import re
import time

import openai

import logger
from performance_timer import get_timer

_logger = logger.get_logger_with_default_conf("openai_api")


class OpenaiApiClient:
    """環境変数から設定を取得し、OpenAIにリクエストを送信するクライアント"""

    def __init__(self):
        # self.api_org = os.getenv("OPENAI_API_ORG", default="smilab")
        api_key = os.getenv("OPENAI_API_KEY")

        # if not self.api_org:
        #     msg = "Please specify a valid ORGANIZATION or unset the environment variable 'OPENAI_API_ORG'."
        #     _logger.error(msg)
        #     raise ValueError(msg)
        if not api_key:
            msg = "Please specify a API_KEY using the environment variable 'OPENAI_API_KEY'."
            _logger.error(msg)
            raise ValueError(msg)

        # openai.organization = self.api_org
        openai.api_key = api_key

        masked_key = f"{api_key[:4]}...{api_key[-3:]}"
        # client_info = f"ORG: {self.api_org}\n KEY: {masked_key}"
        # _logger.info(f"OpenAI API Client started with the following configurations:\n {client_info}")


class AskToOpenaiChatCompletion(OpenaiApiClient):
    """OpenAI APIのChat Completionを利用して、指示文を処理する"""

    def __init__(self, model="gpt-3.5-turbo", max_tokens=1024, n=1, stop=None, temperature=0):
        super().__init__()
        self.config = {
            "model": model,  # モデルを選択
            "max_tokens": max_tokens,  # 生成する文章の最大単語数
            "n": n,  # いくつの返答を生成するか
            "stop": stop,  # 指定した単語が出現した場合、文章生成を打ち切る
            "temperature": temperature,  # 出力する単語のランダム性（0から2の範囲） 0であれば毎回返答内容固定
        }

    def ask_to_openai(self, message):
        """OpenAI APIにリクエストを送信する"""
        prompt = [
            {
                "role": "system",
                "content": "",
            },
            {
                "role": "user",
                "content": message,
            },
        ]

        try:
            res = openai.ChatCompletion.create(messages=prompt, **self.config)
            response = res.choices[0].message.content
            return response
        except openai.error.RateLimitError:
            _logger.warn("Request to OpenAI API was timed out.")
            return None

    def process_instruction(self, inst):
        """Chat Completionで指示文を解析する"""
        msg_to_openai = f'Instruction is "{inst}".'
        msg_to_openai += " What is the target object and destination in the above instruction?"
        msg_to_openai += " Answer in the following format: TARGET: #XXX#, DESTINATION: ##YYY##"

        # OpenAI APIからデータを取得できるまで繰り返す
        while True:
            llm_output = self.ask_to_openai(msg_to_openai)
            if llm_output:
                break
            time.sleep(1)

        target_match = re.search(r"TARGET: #([^#]+)#", llm_output)
        if target_match:
            target_object = target_match.group(1)
        else:
            target_object = "NA"

        destination_match = re.search(r"DESTINATION: ##([^#]+)##", llm_output)
        if destination_match:
            destination = destination_match.group(1)
        else:
            destination = "NA"

        return target_object, destination


class AskToOpenaiApiEmbeddings(OpenaiApiClient):
    """OpenAI APIのChat Completionを利用して、指示文を処理する"""

    def __init__(self, model="text-embedding-ada-002"):
        super().__init__()
        self.config = {
            "model": model,  # モデルを選択
        }

    def ask_to_openai(self, input):
        """OpenAI APIにリクエストを送信する"""
        try:
            res = openai.Embedding.create(input=input, **self.config)
            return res
        except openai.error.RateLimitError:
            _logger.warn("Request to OpenAI API was timed out.")
            return None

    def process_instruction(self, inst):
        """OpenAI APIにリクエストを送信して、指示文の前処理を実行する"""
        # OpenAI APIからデータを取得できるまで繰り返す
        while True:
            embeddings = self.ask_to_openai(inst)
            if embeddings:
                return embeddings["data"][0]["embedding"]
            time.sleep(1)


if __name__ == "__main__":
    completion_ai = AskToOpenaiChatCompletion()
    embeddings_ai = AskToOpenaiApiEmbeddings()

    with get_timer("request to OpenAI API", True):
        inst = "Pick up a cussion on the single sofa and put it on the stairs."
        print(completion_ai.process_instruction(inst))
        print(embeddings_ai.process_instruction(inst))
