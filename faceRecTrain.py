import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from faceRecMod import create_model
import joblib

def process_labels(labels):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
    y = keras.utils.to_categorical(integer_encoded)
    return y, label_encoder


def get_data(directory):
    X = []
    y = []
    labels = []
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".jpeg") or filepath.endswith(".jpg"):
                image = tf.keras.preprocessing.image.load_img(filepath, target_size=(100, 100))
                image = tf.keras.preprocessing.image.img_to_array(image)
                X.append(image)
                label = subdir.split(os.sep)[-1]
                labels.append(label)
                print("label", labels)
    y, label_encoder = process_labels(labels)
    return np.array(X), y, labels, label_encoder


def main():
    X, y, labels, label_encoder = get_data('./Five_Faces')

    model = create_model(input_shape=(100, 100, 3), output_shape=len(np.unique(labels)))
    model.fit(X, y, epochs=10)

    model.save('face_recognition.h5')
    joblib.dump(label_encoder, "label_encoder.joblib")

if __name__ == '__main__':
    main()