import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from faceRecMod import create_model
import joblib


def process_labels(labels):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
    y = tf.keras.utils.to_categorical(integer_encoded)
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

    y, label_encoder = process_labels(labels)
    return np.array(X), y, labels, label_encoder

def main():
    X, y, labels, label_encoder = get_data('./Five_Faces')

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2
    )

    model = create_model(input_shape=(100, 100, 3), output_shape=len(np.unique(labels)))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')
    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='face_recognition.h5', verbose=1, save_best_only=True)

    history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=32), epochs=30,
                                  validation_data=(X_val, y_val), callbacks=[early_stop, checkpointer])

    joblib.dump(label_encoder, "label_encoder.joblib")


if __name__ == '__main__':
    main()
