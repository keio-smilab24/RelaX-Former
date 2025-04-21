from typing import Tuple

import numpy as np
import torch
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor


class LLaVAImageCaptioner:
    def __init__(self, model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf", device: str = "cuda:0"):
        self.processor = LlavaNextProcessor.from_pretrained(model_name)
        self.processor.tokenizer.padding_side = "left"
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        ).to(device)
        self.model.eval()
        self.device = device

    def resize_image(self, image: Image.Image, size: Tuple[int, int] = (224, 224)) -> Image.Image:
        return image.resize(size)

    def generate_caption(self, image: Image.Image, prompt: str) -> str:
        resized_img = self.resize_image(image)
        inputs = self.processor(images=resized_img, text=prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=5, pad_token_id=2)

        caption = self.processor.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return caption.split("[/INST]")[1]

    def extract_features(self, image: Image.Image, prompt: str) -> np.ndarray:
        resized_img = self.resize_image(image)
        inputs = self.processor(images=resized_img, text=prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model(**inputs)

        features = output.hidden_states[-1].to("cpu").detach().numpy().copy()
        return features
