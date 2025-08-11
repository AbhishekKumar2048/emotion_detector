from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import numpy as np
import tensorflow as tf
import io, threading, os
from datetime import datetime

app = FastAPI()

class_names = ['angry', 'happy', 'nothing', 'sad']
vit_model = None
model = None
models_loaded = False

def load_models():
    global vit_model, model, models_loaded
    from transformers import TFViTModel
    print("📦 Loading models...")
    vit_model = TFViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    def vit_forward(pixel_values):
        return vit_model(pixel_values=pixel_values)[0]
    model = tf.keras.models.load_model(
        'vit_wandb_final.keras',
        custom_objects={'vit_forward': vit_forward}
    )
    models_loaded = True
    print("✅ Models loaded.")

@app.on_event("startup")
def startup_event():
    # Load in background so the app binds to the port instantly
    threading.Thread(target=load_models).start()

@app.post("/predict")
async def predict(file: UploadFile = File(...), _ts: str = str(datetime.utcnow())):
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Model is still loading. Please try again in a moment.")
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB").resize((224, 224))
    img_array = np.expand_dims(np.array(image), axis=0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions))
    return {"filename": file.filename, "prediction": predicted_class, "confidence": confidence}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
