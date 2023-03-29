import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from faceRecVGG2 import create_model
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
                image = tf.keras.preprocessing.image.load_img(filepath, target_size=(224, 224))
                image = tf.keras.preprocessing.image.img_to_array(image)
                X.append(image)
                label = subdir.split(os.sep)[-1]
                labels.append(label)
                print(f"Filepath: {filepath}, Label: {label}")

    y, label_encoder = process_labels(labels)
    return np.array(X), y, label_encoder


def main():
    # Delete the h5 file if it exists
    if os.path.exists("face_recognition_vgg2.h5"):
        os.remove("face_recognition_vgg2.h5")

    X, y, label_encoder = get_data('./Employee_Images')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=0)

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )

    model = create_model(input_shape=(224, 224, 3), num_classes=len(np.unique(label_encoder.classes_)))

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')
    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='face_recognition_vgg2.h5', verbose=1,
                                                      save_best_only=True)

    history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=32), epochs=12,
                                  validation_data=(X_val, y_val), callbacks=[early_stop, checkpointer])

    with open("face_recognition_vgg2_history.txt", "w") as f:
        f.write(str(history.history))

    joblib.dump(label_encoder, "label_encoder_vgg2.joblib")


if __name__ == '__main__':
    main()
