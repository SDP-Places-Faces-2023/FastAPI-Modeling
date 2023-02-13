import io
import json
import torch
from tensorflow import keras
import os
import cv2
import tensorflow as tf
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

import model_classifier
import yolov5s
import joblib
from faceRecMod import create_model

from model_classifier import predict, read_imagefile

mserver = FastAPI()

mserver.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@mserver.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    prediction = model_classifier.predict(image)
    return prediction


@mserver.post("/objectdetection/")
async def get_body(file: bytes = File(...)):
    input_image = Image.open(io.BytesIO(file)).convert("RGB")
    results = yolov5s.model(input_image)
    results_json = json.loads(results.pandas().xyxy[0].to_json(orient="records"))
    return {"result": results_json}


model = tf.keras.models.load_model("face_recognition.h5")

label_encoder = joblib.load("label_encoder.joblib")


@mserver.post("/predict")
async def predict(file: UploadFile):
    # Convert UploadFile object to BytesIO object
    file_bytes = await file.read()
    file = io.BytesIO(file_bytes)

    # Load image from BytesIO object
    image = tf.keras.preprocessing.image.load_img(file, target_size=(100, 100))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.expand_dims(image, axis=0)

    # Make prediction using the model
    result = model.predict(image)
    result = np.argmax(result, axis=1)
    result = label_encoder.inverse_transform(result)
    return {"result": result[0]}
