import json
import os

from openai import OpenAI
from tqdm import tqdm

MP3D_DATABASE_PATH = "data/ltrpo_dataset/ltrpo_database.json"
HM3D_DATABASE_PATH = "data/hm3d/hm3d_dataset/hm3d_database.json"

MP3D_COS_SIMILARITY_PATH = "result/mp3d_cos_similarity_without_similarity.json"
HM3D_COS_SIMILARITY_PATH = "result/hm3d_cos_similarity_without_similarity.json"


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI()


def create_jsonl_mp3d():
    with(MP3D_DATABASE_PATH, "r") as f:
        database = json.load(f)
    for data in tqdm(database):

    pass
