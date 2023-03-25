# import tensorflow as tf
# import numpy as np
# from keras_vggface.vggface import VGGFace
# from keras_vggface.utils import preprocess_input
# from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from keras_vggface.vggface import VGGFace


def create_model(input_shape, num_classes):
    # Load pre-trained VGGFace2 model
    base_model = VGGFace(model='resnet50', include_top=False, input_shape=input_shape)

    # Freeze all layers in the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom classification layers on top of the base model
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    # Create final model
    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

    # Compile model with appropriate loss function and optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model