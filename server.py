import base64
import io
import os
import shutil
from datetime import datetime
from typing import List
from keras_vggface.utils import preprocess_input
import cv2
import tensorflow as tf
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import joblib

mserver = FastAPI()

mserver.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

face_cascade = None
model_vgg2 = None
label_encoder_vgg2 = None

model_path = "face_recognition_vgg2.h5"
label_encoder_path = "label_encoder_vgg2.joblib"
face_cascade_path = "cascades/data/haarcascade_frontalface_alt2.xml"

if os.path.exists(face_cascade_path):
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

if os.path.exists(model_path) and os.path.exists(label_encoder_path):
    model_vgg2 = tf.keras.models.load_model(model_path)
    label_encoder_vgg2 = joblib.load(label_encoder_path)


async def recognize_image(image: Image):
    if model_vgg2 is None or label_encoder_vgg2 is None:
        return {"error": "Required files not found"}
    # Resize input image
    img = image.resize((224, 224))

    # Convert the PIL Image to RGB if it's in grayscale
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Convert the PIL Image to a numpy array and change the channel order to BGR
    img_array = np.array(img)[:, :, ::-1]

    # Preprocess input image
    x = tf.keras.preprocessing.image.img_to_array(img_array)
    x = preprocess_input(x)

    # Make prediction
    pred = model_vgg2.predict(np.array([x]))
    pred_label = label_encoder_vgg2.inverse_transform(np.argmax(pred, axis=1))[0]
    confidence = np.max(pred)

    if confidence > 0.98:
        # Return prediction result
        return {"predicted_face": pred_label}
    elif confidence > 0.75:
        today = datetime.now().strftime("%Y-%m-%d")
        folder_path = os.path.join("unrecognized_faces", today)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Save unrecognized face to file
        file_name = f"closest_face_{pred_label}_confidence_{confidence}.jpg"
        file_path = os.path.join(folder_path, file_name)
        img.save(file_path)

    return {"error": "Unknown", "closest_face": pred_label, "confidence": float(confidence)}


@mserver.post("/detect_recognize/")
async def detect_faces_and_recognize(file: UploadFile = File(...)):
    if face_cascade is None:
        return {"error": "Required face cascade file not found"}
    if model_vgg2 is None or label_encoder_vgg2 is None:
        return {"error": "Required files not found"}
    contents = await file.read()
    pil_image = Image.open(io.BytesIO(contents))
    image_array = np.array(pil_image, "uint8")
    faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

    if len(faces) == 0:
        print("No Face")
        return "No Face"
    else:
        face_coordinates = faces.tolist()
        x, y, w, h = face_coordinates[0]

        # Crop the face from the image
        cropped_face = pil_image.crop((x, y, x + w, y + h))

        # Recognize the cropped face
        recognition_results = await recognize_image(cropped_face)
        if "Unknown" in recognition_results:
            return {"face_coordinates": face_coordinates[0], "recognition_results": "Unknown",
                    "closest_face": recognition_results}
        else:
            return {"face_coordinates": face_coordinates[0], "recognition_results": recognition_results}


async def crop_face(pil_image):
    if face_cascade is None:
        return {"error": "Required face cascade file not found"}
    image_array = np.array(pil_image, "uint8")
    faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

    if len(faces) == 0:
        print("No Face")
        return None
    else:
        face_coordinates = faces.tolist()
        x, y, w, h = face_coordinates[0]

        # Crop the face from the image
        cropped_face = pil_image.crop((x, y, x + w, y + h))
        return cropped_face


@mserver.post("/upload_images/")
async def upload_images(id: str, images: List[UploadFile] = File(...)):
    # Create a folder with the ID
    folder_path = f"./Employee_Images/{id}"
    os.makedirs(folder_path, exist_ok=True)

    # Save the cropped images in the folder
    for image in images:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents))

        # Crop the face from the image
        cropped_face = await crop_face(pil_image)

        if cropped_face is not None:
            # Save the cropped face
            file_path = os.path.join(folder_path, f"cropped_{image.filename}")
            cropped_face.save(file_path)
        else:
            print(f"No face detected in {image.filename}")

    return {"message": "Images uploaded and cropped successfully"}


@mserver.get("/has_images/")
async def has_images(id: str):
    folder_path = f"./Employee_Images/{id}"
    if not os.path.exists(folder_path):
        return {"has_images": False}

    files = os.listdir(folder_path)
    images = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif"))]
    return {"has_images": len(images) > 0}


@mserver.get("/get_images/")
async def get_images(id: str):
    # Build the path to the folder
    folder_path = f"./Employee_Images/{id}"

    # Check if the folder exists
    if not os.path.exists(folder_path):
        # If it doesn't exist, return an error message
        return {"message": f"Folder {id} does not exist"}

    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    # Filter out non-image files
    images = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif"))]

    # Build a dictionary of Base64-encoded image strings for each image file
    encoded_images = {}
    for image_filename in images:
        image_path = os.path.join(folder_path, image_filename)
        with open(image_path, "rb") as f:
            encoded_image = base64.b64encode(f.read()).decode()
        encoded_images[image_filename] = encoded_image

    return encoded_images


@mserver.post("/delete_images/")
async def delete_images(id: str):
    # Build the path to the folder
    folder_path = f"./Employee_Images/{id}"

    # Check if the folder exists
    if os.path.exists(folder_path):
        # If it exists, delete it
        shutil.rmtree(folder_path)
        return {"success": True, "message": f"Folder {id} deleted successfully"}
    else:
        # If it doesn't exist, return an error message
        return {"success": False, "message": f"Folder {id} does not exist"}
