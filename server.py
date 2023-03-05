import io
import json
import os
from typing import List

# import torch
# from tensorflow import keras
# import os
import cv2
import tensorflow as tf
import numpy as np
# import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

import model_classifier
import yolov5s
import joblib
# from faceRecMod import create_model

from model_classifier import predict, read_imagefile
from recongize import recognizer, labels

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
    result = model.predict(image)[0]
    max_prob_idx = np.argmax(result)
    max_prob = np.max(result)
    predicted_label = label_encoder.inverse_transform([max_prob_idx])

    return {"result": predicted_label[0], "confidence": float(max_prob)}



face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')


@mserver.post("/detect_faces/")
async def detect_faces(file: UploadFile = File(...)):
    contents = await file.read()
    pil_image = Image.open(io.BytesIO(contents)).convert("L")
    image_array = np.array(pil_image, "uint8")
    faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
    if len(faces) == 0:
        print("No Face")
        return "No Face"
    else:
        face_coordinates = faces.tolist()
        print(face_coordinates)
        return face_coordinates[0]

@mserver.post("/recognize_faces/")
async def recognize_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.6, minNeighbors=8)

    results = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        id_, conf = recognizer.predict(roi_gray)
        if 45 <= conf <= 99:
            name = labels[id_]
            results.append({"name": name, "confidence": conf})

    return {"results": results}



@mserver.post("/upload_images/")
async def upload_images(id: str, images: List[UploadFile] = File(...)):
    # Create a folder with the ID
    folder_path = f"./Five_Faces/{id}"
    os.makedirs(folder_path, exist_ok=True)

    # Save the images in the folder
    for image in images:
        file_path = os.path.join(folder_path, image.filename)
        with open(file_path, "wb") as f:
            f.write(image.file.read())

    return {"message": "Images uploaded successfully"}