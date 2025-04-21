from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import torch
from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    AutoImageProcessor,
    ViTModel,
)
import io

app = FastAPI()

llava_processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
llava_model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16
)
llava_model.to("cuda:0").eval()
vit_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224")
vit_model.to("cuda:0").eval()


def create_vit_image_feature(image: Image.Image, np_return=False):
    image = image.resize((224, 224))
    _input = vit_processor(images=image, return_tensors="pt").pixel_values.to("cuda:0")
    with torch.no_grad():
        vit_output = vit_model(_input)
    return vit_output.last_hidden_state[:, 0, :].cpu().numpy()


def create_llava_image_feature(image: Image.Image, np_return=False):
    image = image.resize((224, 224))
    prompt = "[INST] <image>\nThis is an image of a certain house. Please describe in detail the location of objects in this image. In doing so, be sure to mention the following object given in the instructions.\nAfter describing the object within the directive, continue to elaborate on the arrangement of other objects throughout the house, ensuring a thorough and comprehensive description.\nIn the description, refrain from using personal pronouns such as 'I', 'we', or 'you'.\nAlso, avoid including sensory details.\nFocus solely on providing a clear description.\nBegin the description with the phrase 'In the image'\nThe directions should be as indicated in the image\n[/INST]"

    _input = llava_processor(images=image, text=prompt, return_tensors="pt").to("cuda:0")

    with torch.no_grad():
        llava_output = llava_model(**_input, output_hidden_states=True)
    llava_features = llava_output.hidden_states[-1][:, 0, :]
    return llava_features


@app.post("/vit")
async def vit(image: UploadFile = File(...)):
    try:
        img = Image.open(io.BytesIO(await image.read()))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image file: {str(e)}")

    vit_features = create_vit_image_feature(img, np_return=True)
    return JSONResponse({"features": vit_features.tolist()})


@app.post("/llava")
async def llava(image: UploadFile = File(...)):
    try:
        img = Image.open(io.BytesIO(await image.read()))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image file: {str(e)}")

    llava_features = create_llava_image_feature(img, np_return=True)
    return JSONResponse({"features": llava_features.tolist()})


if __name__ == "__main__":
    import uvicorn
    import os

    server_host = os.environ.get("EMBED_IMAGE_SERVER_HOST", "0.0.0.0")
    server_port = os.environ.get("EMBED_IMAGE_SERVER_PORT", 5000)

    uvicorn.run(app, host=server_host, port=server_port)
