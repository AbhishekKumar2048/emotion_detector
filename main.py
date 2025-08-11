from fastapi import FastAPI, UploadFile, File
import os, io, numpy as np
from PIL import Image
import tensorflow as tf
from transformers import TFViTModel

app = FastAPI()

vit_model = None
model = None
class_names = ['angry', 'happy', 'nothing', 'sad']

@app.on_event("startup")
def load_models():
    global vit_model, model
    print("📦 Loading models...")
    vit_model = TFViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    def vit_forward(pixel_values):
        return vit_model(pixel_values=pixel_values)[0]
    model = tf.keras.models.load_model(
        'vit_wandb_final.keras',
        custom_objects={'vit_forward': vit_forward}
    )
    print("✅ Models loaded.")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = image.resize((224, 224))
    img_array = np.expand_dims(np.array(image), axis=0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions))
    return {"filename": file.filename, "prediction": predicted_class, "confidence": confidence}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
