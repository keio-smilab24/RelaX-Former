"""HM3Dデータセットに対してOpenAI APIを実行した結果を保存"""

import json
import os

import numpy as np
from tqdm import tqdm

from openai_api import AskToOpenaiApiEmbeddings, AskToOpenaiChatCompletion


def main():
    database_no_gpt_path = "data/hm3d/hm3d_dataset/hm3d_database_no_gpt.json"
    database_no_gpt = None
    with open(database_no_gpt_path, "r") as json_file:
        database_no_gpt = json.load(json_file)

    completion_ai = AskToOpenaiChatCompletion()
    embeddings_ai = AskToOpenaiApiEmbeddings()

    database_with_gpt = []

    instruction_ids = []
    # TODO: better: 実行済み以外のみ再実行可能にする
    llm_results = []

    for data in tqdm(database_no_gpt):
        # 1文に1回実行で十分
        if data["instruction_id"] in instruction_ids:
            inst_idx = instruction_ids.index(data["instruction_id"])
            assert data["instruction_id"] == llm_results[inst_idx]["instruction_id"]
            data["instruction_chatgpt"] = llm_results[inst_idx]["instruction_chatgpt"]
            data["llm_data"] = llm_results[inst_idx]["llm_data"]
            database_with_gpt.append(data)
            # print("continued")
            continue

        inst = data["instruction"]
        target, dest = completion_ai.process_instruction(inst)
        embeddings = embeddings_ai.process_instruction(inst)

        data["instruction_chatgpt"] = target
        data["llm_data"] = {"destination": dest, "modified_instruction": f"Carry {target} to {dest}"}

        # ada埋め込みをnpyで保存
        ada_save_dir = "/".join(data["gpt3_embeddings"].split("/")[:-1])
        os.makedirs(ada_save_dir, exist_ok=True)
        np.save(data["gpt3_embeddings"], np.array(embeddings))

        database_with_gpt.append(data)
        instruction_ids.append(data["instruction_id"])

        llm_result = {
            "instruction_id": data["instruction_id"],
            "instruction": data["instruction"],
            "instruction_chatgpt": data["instruction_chatgpt"],
            "llm_data": data["llm_data"],
            "gpt3_embeddings": data["gpt3_embeddings"],
        }
        llm_results.append(llm_result)

    # 新たなdatabase.jsonを保存
    os.makedirs("data/hm3d/hm3d_dataset", exist_ok=True)
    with open("data/hm3d/hm3d_dataset/hm3d_database.json", "w") as json_file:
        json.dump(database_with_gpt, json_file, indent=4)

    # llm関連の出力のみの別ファイルも念のため保存
    with open("data/hm3d/hm3d_dataset/hm3d_llm_output.json", "w") as json_file:
        json.dump(llm_results, json_file, indent=4)


# poetry run python src/save_openai_api_results.py
if __name__ == "__main__":
    main()
