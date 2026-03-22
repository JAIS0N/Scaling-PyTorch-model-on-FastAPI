import json
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, status, UploadFile, File
import torch
from PIL import Image
import numpy as np

MODEL_PATH = "./mobilenet_v2.pt"
LABELS = {}
model_loading_time = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL_PATH, LABELS, model_loading_time

    # Load model on startup
    model_loading_start_time = time.time()

    # FIX: assign model + weights_only=False
    app.state.model = torch.load(
        MODEL_PATH,
        map_location=torch.device("cpu"),
        weights_only=False
    )

    # Set evaluation mode
    app.state.model.eval()

    # Load labels
    with open("imagenet_class_index.json", "r") as f:
        LABELS = json.load(f)

    model_loading_time = time.time() - model_loading_start_time

    yield

    # Cleanup
    del app.state.model


app = FastAPI(lifespan=lifespan, title="Pytorch API")


@app.get("/", status_code=status.HTTP_200_OK)
def root():
    return "Hello World"


@app.get("/model", status_code=status.HTTP_200_OK)
def model_info():
    return f"Model loaded in {model_loading_time} s"


def preprocess_image(image):
    # Resize and normalize the image
    image = image.resize((256, 256)).crop((16, 16, 240, 240))
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))
    mean = np.array([0.485, 0.456, 0.406])[:, None, None]
    std = np.array([0.229, 0.224, 0.225])[:, None, None]
    return torch.from_numpy(((img_array - mean) / std).astype(np.float32)).unsqueeze(0)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()
    # Read and preprocess the image
    image = Image.open(file.file).convert("RGB")
    input_data = preprocess_image(image)

    # Run inference
    with torch.inference_mode():
        results = app.state.model(input_data)

    # Get the predicted class probabilities
    probabilities = torch.nn.functional.softmax(results[0], dim=0)
    top_prob, top_class = torch.max(probabilities, 0)
    label = LABELS[str(top_class.item())][-1]
    score = top_prob.item()

    inference_time = time.time() - start_time

    return {
        "class": label,
        "score": score,
        "inference_time (s)": inference_time,
        "model_loading_time (s)": model_loading_time,
    }