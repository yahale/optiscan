from fastapi import FastAPI, UploadFile, HTTPException, Request, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
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

# --- Mount the directory containing index.html as static ---
app.mount("/static", StaticFiles(directory="."), name="static")

# --- Load models ---
IMAGE_MODEL_FILE = 'my_model.keras'
DRY_EYE_MODEL_FILE = 'dry_eye_rf_model_updated.pkl'

try:
    image_model = tf.keras.models.load_model(IMAGE_MODEL_FILE)
    logger.info("Image model (Keras) loaded successfully.")
except Exception as e:
    logger.error(f"Error loading image model (Keras): {e}")
    image_model = None

dry_eye_model = None
try:
    dry_eye_model = joblib.load(DRY_EYE_MODEL_FILE)
    logger.info("Dry eye model (PKL) loaded successfully.")
except Exception as e:
    logger.error(f"Error loading dry eye model (PKL): {e}")

# --- Load class names ---
CLASSES_NAMES = ['Bulging_Eyes', 'Cataracts', 'Crossed_Eyes', 'Glaucoma', 'Uveitis']

# --- Image Model Pydantic Models ---
class ImagePredictionOutput(BaseModel):
    probabilities: Optional[List[float]]
    predicted_class_name: Optional[str]

# --- Dry Eye Model Pydantic Models ---
class DryEyeOutput(BaseModel):
    dry_eye_probability: str

# --- Combined Output Model ---
class CombinedOutput(BaseModel):
    image_prediction: Optional[ImagePredictionOutput] = None
    dry_eye_prediction: Optional[DryEyeOutput] = None
    snellen_prediction: Optional[dict] = None

@app.post("/predict")
async def predict(
    request: Request,
    file: Optional[UploadFile] = File(None),
):
    combined_output = CombinedOutput()
    logger.info("Received prediction request")
    form_data = await request.form()  # Get form data

    # --- Image Processing ---
    if file:
        try:
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))
            image = image.resize((224, 224))
            image = np.array(image) / 255.0
            image = np.expand_dims(image, axis=0)

            prediction = image_model.predict(image)
            predicted_class_index = np.argmax(prediction)
            predicted_class_name = CLASSES_NAMES[predicted_class_index]
            probabilities = prediction.tolist()[0]

            combined_output.image_prediction = ImagePredictionOutput(
                probabilities=probabilities, predicted_class_name=predicted_class_name
            )
            logger.info(f"image_prediction: {combined_output.image_prediction}")

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            error_message = traceback.format_exc()
            logger.error(error_message)
            raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

    # --- Snellen Test Processing ---
    if "snellen_data" in form_data:
        try:
            snellen_data_str = form_data["snellen_data"]
            snellen_data_dict = json.loads(snellen_data_str)

            line = int(snellen_data_dict.get("line"))
            mistakes = int(snellen_data_dict.get("mistakes"))
            age = int(snellen_data_dict.get("age"))

            model = joblib.load("snellen_model.pkl")
            encoder = joblib.load("snellen_label_encoder.pkl")

            df = pd.DataFrame([[line, mistakes, age]], columns=["line_read", "mistakes", "age"])
            prediction = model.predict(df)
            label = encoder.inverse_transform(prediction)[0]
            combined_output.snellen_prediction = {"diagnosis": label}
            logger.info(f"Snellen prediction: {label}")

        except Exception as e:
            logger.error(f"Error processing Snellen data: {e}")
            error_message = traceback.format_exc()
            logger.error(error_message)
            raise HTTPException(status_code=500, detail=f"Error processing Snellen data: {e}")

    # --- Dry Eye Assessment Processing ---
    if "dry_eye_data" in form_data:
        try:
            dry_eye_data_str = form_data["dry_eye_data"]
            input_dict = json.loads(dry_eye_data_str)

            # Create a copy to avoid modifying the original input_dict
            input_dict_copy = input_dict.copy()

            column_mapping = {
            "Gender": "Gender",
            "Age": "Age",
            "Sleep_duration": "Sleep duration",
            "Sleep_quality": "Sleep quality",
            "Stress_level": "Stress level",
            "Sleep_disorder": "Sleep disorder",
            "Wake_up_during_night": "Wake up during night",
            "Feel_sleepy_during_day": "Feel sleepy during day",
            "Caffeine_consumption": "Caffeine consumption",
            "Alcohol_consumption": "Alcohol consumption",
            "Smoking": "Smoking",
            "Smart_device_before_bed": "Smart device before bed",
            "Average_screen_time": "Average screen time",
            "Blue_light_filter": "Blue-light filter",
            "Discomfort_Eye_strain": "Discomfort Eye-strain",
            "Redness_in_eye": "Redness in eye",
            "Itchiness_Irritation_in_eye": "Itchiness/Irritation in eye",
            "Systolic": "Systolic",
            "Diastolic": "Diastolic"
        }

            input_renamed = {}
            for k, v in input_dict_copy.items():
                if k in column_mapping:
                    input_renamed[column_mapping[k]] = v
                else:
                    logger.warning(f"Key {k} not found in column_mapping. Skipping.")
            df = pd.DataFrame([input_renamed])

            df["Gender"] = df["Gender"].map({"F": 1, "M": 0})

            yn_cols = [
            "Sleep disorder", "Wake up during night", "Feel sleepy during day",
            "Caffeine consumption", "Alcohol consumption", "Smoking",
            "Smart device before bed", "Blue-light filter",
            "Discomfort Eye-strain", "Redness in eye", "Itchiness/Irritation in eye"
        ]
            for col in yn_cols:
                if col in df.columns: # Check if the column exists before mapping
                    df[col] = df[col].map({"Y": 1, "N": 0})
                else:
                    logger.warning(f"Column {col} not found in DataFrame. Skipping mapping.")


            prediction = dry_eye_model.predict(df)[0]
            result = "Dry Eye" if prediction == 1 else "No Dry Eye"

            combined_output.dry_eye_prediction = DryEyeOutput(dry_eye_probability=result)
            logger.info(f"dry_eye_prediction: {combined_output.dry_eye_prediction}")

        except Exception as e:
            logger.error(f"Error processing dry eye data: {e}")
            error_message = traceback.format_exc()
            logger.error(error_message)
            raise HTTPException(status_code=500, detail=f"Error processing dry eye data: {e}")

    logger.info(f"combined_output: {combined_output}")
    return combined_output

@app.get("/")
async def root():
    """
    Serves the main HTML file.
    """
    return FileResponse("index.html")
