from fastapi import FastAPI, UploadFile, File
from PIL import Image
import numpy as np
import tensorflow as tf
import io, os
from datetime import datetime

app = FastAPI()

vit_model = None
model = None
class_names = ['angry', 'happy', 'nothing', 'sad']

@app.on_event("startup")
def load_models():
    global vit_model, model
    from transformers import TFViTModel
    vit_model = TFViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    def vit_forward(pixel_values):
        return vit_model(pixel_values=pixel_values)[0]
    model = tf.keras.models.load_model('vit_wandb_final.keras',
                                       custom_objects={'vit_forward': vit_forward})

@app.post("/predict")
async def predict(file: UploadFile = File(...), _ts: str = str(datetime.utcnow())):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB").resize((224, 224))
    img_array = np.expand_dims(np.array(image), axis=0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions))
    return {"filename": file.filename, "prediction": predicted_class, "confidence": confidence}
