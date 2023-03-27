import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.preprocessing import LabelEncoder
from scratchCnnMod import create_model_scratch
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
    return np.array(X), y, label_encoder


def main():
    X, y, label_encoder = get_data('./Five_Faces')

    # Split the data into training and testing sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )

    fold = 1
    acc_per_fold = []
    loss_per_fold = []

    # Define hyperparameters grid search
    param_grid = {
        'epochs': [30],
        'batch_size': [32],
        'learning_rate': [0.001, 0.0001],
        'dropout_rate': [0.2, 0.3, 0.4]
    }

    # Perform grid search
    for params in ParameterGrid(param_grid):
        print("=======================================")
        print(f"=           Params: {params}          =")
        print("=======================================")
        for train_index, val_index in skf.split(X_train, np.argmax(y_train, axis=1)):
            print("=======================================")
            print("=               Fold ", fold, "              =")
            print("=======================================")
            X_train_fold, X_val = X_train[train_index], X_train[val_index]
            y_train_fold, y_val = y_train[train_index], y_train[val_index]

            model = create_model_scratch(input_shape=(100, 100, 3), output_shape=len(np.unique(label_encoder.classes_)),
                                         learning_rate=params['learning_rate'], dropout_rate=params['dropout_rate'])

            early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')
            checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=f'face_recognition_fold_{fold}.h5', verbose=1,
                                                              save_best_only=True)

            history = model.fit(datagen.flow(X_train_fold, y_train_fold, batch_size=params['batch_size']),
                                epochs=params['epochs'],
                                validation_data=(X_val, y_val), callbacks=[early_stop, checkpointer])

            scores = model.evaluate(X_val, y_val, verbose=0)
            print(f"Accuracy for fold {fold}: {scores[1] * 100}%")
            acc_per_fold.append(scores[1] * 100)
            loss_per_fold.append(scores[0])

            joblib.dump(label_encoder, f"label_encoder_fold_{fold}.joblib")

            fold += 1

        print("=======================================")
        print("=            Cross validation          =")
        print("=======================================")
        print(f"Mean accuracy: {np.mean(acc_per_fold)} (+/- {np.std(acc_per_fold)})")
        print(f"Mean loss: {np.mean(loss_per_fold)} (+/- {np.std(loss_per_fold)})")

        # Reset folds and metrics
        fold = 1
        acc_per_fold = []
        loss_per_fold = []

    # Evaluate on the test set
    scores = model.evaluate(X_test, y_test, verbose=0)
    print(f"Accuracy on test set: {scores[1] * 100}%")


if __name__ == '__main__':
    main()
