import uvicorn
from fastapi import FastAPI, UploadFile, File

from model import predict, read_imagefile

mserver = FastAPI()


@mserver.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    prediction = predict(image)
    return prediction