import cv2
import numpy as np
import pickle
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {}
with open("labels.pickle", 'rb') as f:
    old_labels = pickle.load(f)
    labels = {v: k for k, v in old_labels.items()}

