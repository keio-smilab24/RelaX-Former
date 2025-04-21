"""create image features for predict_one_shot"""

import argparse
import copy
import json
import os

import numpy as np
import torch
from PIL import Image
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
from torchvision import transforms
from transformers import (
    AutoImageProcessor,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
    ViTModel,
)

import clip
from llava.llava import LLaVAImageCaptioner
from sam.sam_server import SimpleSegmentAnything


class ImageFeatureExtractor:

    def create_clip_image_feature(self, image: Image.Image, np_return=False):
        """
        create clip image features
        """
        clip_model, preprocess_clip = clip.load("ViT-L/14", device="cuda:0")

        with torch.no_grad():
            image = preprocess_clip(image).unsqueeze(0).to("cuda:0")
            image_features = clip_model.encode_image(image)
        if np_return:
            return image_features.cpu().numpy()
        return image_features

    def create_vit_image_feature(self, image, np_return=False):
        """
        create ViT image features
        """
        image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224")
        vit_model.eval()  # type: ignore
        vit_model.to("cuda:0")  # type: ignore
        image = image.resize((224, 224))  # type: ignore
        image = image_processor(image, return_tensors="pt").pixel_values.to("cuda:0")
        with torch.no_grad():
            image_features = vit_model(image)  # type: ignore

        image_features = image_features.last_hidden_state[:, 0, :]
        if np_return:
            return image_features.cpu().numpy()
        return image_features

    def create_llava_image_feature(self, image: Image.Image, np_return=False):
        llava_processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        llava_model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float32
        )
        llava_model.eval()  # type: ignore
        image = image.resize((224, 224))
        prompt = "[INST] <image>\nThis is an image of a certain house. Please describe in detail the location of objects in this image. In doing so, be sure to mention the following object given in the instructions.\nAfter describing the object within the directive, continue to elaborate on the arrangement of other objects throughout the house, ensuring a thorough and comprehensive description.\nIn the description, refrain from using personal pronouns such as 'I', 'we', or 'you'.\nAlso, avoid including sensory details.\nFocus solely on providing a clear description.\nBegin the description with the phrase 'In the image'\nThe directions should be as indicated in the image\n[/INST]"

        _input = llava_processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")  # type: ignore

        with torch.no_grad():
            llava_output = llava_model(**_input, output_hidden_states=True)  # type: ignore
        llava_features = llava_output.hidden_states[-1][:, 0, :]
        if np_return:
            return llava_features.cpu().numpy()
        return llava_features
