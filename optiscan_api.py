from fastapi import FastAPI, UploadFile, HTTPException, Request, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from pydantic import BaseModel
from typing import List, Optional
import tensorflow as tf
import numpy as np
import logging
import json
import joblib
import pandas as pd
import traceback
import os
import io

app = FastAPI()

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Mount static directory ---
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Templates setup ---
templates = Jinja2Templates(directory="templates")

# --- Serve homepage ---
@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# --- Load models ---
IMAGE_MODEL_FILE = 'my_model.keras'
DRY_EYE_MODEL_FILE = 'dry_eye_rf_model_updated.pkl'
SNELLEN_MODEL_FILE = 'snellen_model.pkl'
SNELLEN_ENCODER_FILE = 'snellen_label_encoder.pkl'

try:
    image_model = tf.keras.models.load_model(IMAGE_MODEL_FILE)
    logger.info("✅ Image model loaded.")
except Exception as e:
    logger.error(f"❌ Failed to load image model: {e}")
    image_model = None

try:
    dry_eye_model = joblib.load(DRY_EYE_MODEL_FILE)
    logger.info("✅ Dry eye model loaded.")
except Exception as e:
    logger.error(f"❌ Failed to load dry eye model: {e}")
    dry_eye_model = None

try:
    snellen_model = joblib.load(SNELLEN_MODEL_FILE)
    snellen_encoder = joblib.load(SNELLEN_ENCODER_FILE)
    logger.info("✅ Snellen model + encoder loaded.")
except Exception as e:
    logger.error(f"❌ Failed to load Snellen model or encoder: {e}")
    snellen_model = None
    snellen_encoder = None

CLASSES_NAMES = ['Uveitis', 'Normal', 'Eyelid', 'Conjunctivitis', 'Cataract']

# --- Response Models ---
class ImagePredictionOutput(BaseModel):
    probabilities: Optional[List[float]]
    predicted_class_name: Optional[str]

class DryEyeOutput(BaseModel):
    dry_eye_probability: str

class CombinedOutput(BaseModel):
    image_prediction: Optional[ImagePredictionOutput] = None
    dry_eye_prediction: Optional[DryEyeOutput] = None
    snellen_prediction: Optional[dict] = None

@app.post("/predict")
async def predict(request: Request, file: Optional[UploadFile] = File(None)):
    combined_output = CombinedOutput()
    logger.info("📥 Prediction request received.")
    
    form_data = await request.form()

    # --- Image Prediction ---
    if file and image_model:
        try:
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            image = image.resize((224, 224))
            image = np.expand_dims(np.array(image) / 255.0, axis=0)
            prediction = image_model.predict(image)
            predicted_class = int(np.argmax(prediction))
            combined_output.image_prediction = ImagePredictionOutput(
                predicted_class_name=CLASSES_NAMES[predicted_class],
                probabilities=prediction.tolist()[0]
            )
        except Exception as e:
            logger.error(f"❌ Image processing error: {e}")
            raise HTTPException(status_code=500, detail="Image prediction failed")

    # --- Snellen Prediction ---
    if "snellen_data" in form_data and snellen_model:
        try:
            snellen_data = json.loads(form_data["snellen_data"])
            line = int(snellen_data.get("line"))
            mistakes = int(snellen_data.get("mistakes"))
            age = int(snellen_data.get("age"))
            effort = float(snellen_data.get("effort"))

            df = pd.DataFrame([[line, mistakes, age, effort]], columns=["line_read", "mistakes", "age", "effort"])
            pred = snellen_model.predict(df)
            label = snellen_encoder.inverse_transform(pred)[0]
            combined_output.snellen_prediction = {"diagnosis": label}
        except Exception as e:
            logger.error(f"❌ Snellen prediction error: {e}")
            raise HTTPException(status_code=500, detail="Snellen prediction failed")

    # --- Dry Eye Prediction ---
    if "dry_eye_data" in form_data and dry_eye_model:
        try:
            dry_eye_data = json.loads(form_data["dry_eye_data"])
            column_mapping = {
                "Gender": "Gender", "Age": "Age", "Sleep_duration": "Sleep duration", "Sleep_quality": "Sleep quality",
                "Stress_level": "Stress level", "Sleep_disorder": "Sleep disorder", "Wake_up_during_night": "Wake up during night",
                "Feel_sleepy_during_day": "Feel sleepy during day", "Caffeine_consumption": "Caffeine consumption",
                "Alcohol_consumption": "Alcohol consumption", "Smoking": "Smoking", "Smart_device_before_bed": "Smart device before bed",
                "Average_screen_time": "Average screen time", "Blue_light_filter": "Blue-light filter",
                "Discomfort_Eye_strain": "Discomfort Eye-strain", "Redness_in_eye": "Redness in eye",
                "Itchiness_Irritation_in_eye": "Itchiness/Irritation in eye", "Systolic": "Systolic", "Diastolic": "Diastolic"
            }

            df = pd.DataFrame([{column_mapping[k]: v for k, v in dry_eye_data.items() if k in column_mapping}])
            df["Gender"] = df["Gender"].map({"F": 1, "M": 0})
            for col in [
                "Sleep disorder", "Wake up during night", "Feel sleepy during day", "Caffeine consumption", 
                "Alcohol consumption", "Smoking", "Smart device before bed", "Blue-light filter",
                "Discomfort Eye-strain", "Redness in eye", "Itchiness/Irritation in eye"
            ]:
                if col in df.columns:
                    df[col] = df[col].map({"Y": 1, "N": 0})
            result = dry_eye_model.predict(df)[0]
            combined_output.dry_eye_prediction = DryEyeOutput(
                dry_eye_probability="Dry Eye" if result == 1 else "No Dry Eye"
            )
        except Exception as e:
            logger.error(f"❌ Dry eye prediction error: {e}")
            raise HTTPException(status_code=500, detail="Dry eye prediction failed")

    logger.info("✅ Prediction response sent.")
    return combined_output

