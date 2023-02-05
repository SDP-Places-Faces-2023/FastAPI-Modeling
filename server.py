import io
import json

import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import yolov5s
from model import predict, read_imagefile

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
    prediction = predict(image)
    return prediction


@mserver.post("/objectdetection/")
async def get_body(file: bytes = File(...)):
    input_image = Image.open(io.BytesIO(file)).convert("RGB")
    results = yolov5s.model(input_image)
    results_json = json.loads(results.pandas().xyxy[0].to_json(orient="records"))
    return {"result": results_json}
