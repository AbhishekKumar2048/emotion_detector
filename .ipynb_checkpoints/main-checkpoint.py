from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import numpy as np
import tensorflow as tf
import io

import tensorflow as tf
from keras.utils import image_dataset_from_directory as dfd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
import tensorflow as tf


import torch

from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model

from transformers import TFViTModel
vit_model = TFViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

from keras.saving import register_keras_serializable
@register_keras_serializable()
def vit_forward(pixel_values):
    return vit_model(pixel_values=pixel_values)[0]

# Load model
model=tf.keras.models.load_model('vit_wandb_trial.keras', custom_objects={'vit_forward': vit_forward})
# Define class labels
class_labels = ['angry', 'happy', 'sad', 'nothing']

# Initialize FastAPI
app = FastAPI()

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Image preprocessing function
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))  # Resize to match model input
    image = image.convert("RGB")
    image_array = np.array(image).astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Serve HTML form
@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

# Predict route for HTML form
@app.post("/predict/")
async def predict(request: Request, file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": "Only .jpg, .jpeg or .png files are allowed."
        })

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        input_tensor = preprocess_image(image)
        predictions = model.predict(input_tensor)
        predicted_class = class_labels[np.argmax(predictions)]
        confidence = float(np.max(predictions))
        result = f"Emotion: {predicted_class} (Confidence: {confidence:.2f})"
    except Exception as e:
        result = f"Error during prediction: {str(e)}"

    return templates.TemplateResponse("index.html", {"request": request, "result": result})

# JSON API route
@app.post("/api/predict")
async def api_predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Only .jpg, .jpeg or .png files are supported.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        input_tensor = preprocess_image(image)
        predictions = model.predict(input_tensor)
        predicted_class = class_labels[np.argmax(predictions)]
        confidence = float(np.max(predictions))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    return JSONResponse(content={
        "filename": file.filename,
        "predicted_emotion": predicted_class,
        "confidence": round(confidence, 4)
    })
