from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
import torch
from transformers import TFViTModel

vit_model = TFViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

def vit_forward(pixel_values):
    return vit_model(pixel_values=pixel_values)[0]

# Load model
model=tf.keras.models.load_model('vit_wandb_final.keras', custom_objects={'vit_forward': vit_forward})

# main.py

app = FastAPI()

# Update with your actual class labels
class_names = ['angry','happy','nothing','sad']

from datetime import datetime

@app.post("/predict")
async def predict(file: UploadFile = File(...), _ts: str = str(datetime.utcnow())):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Match your model's expected input size
    image = image.resize((224,224))
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_array)
    print("Predictions:", predictions)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions))

    return {
        "filename": file.filename,
        "prediction": predicted_class,
        "confidence": confidence
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # 8000 locally, dynamic on Render
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
