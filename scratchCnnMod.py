import tensorflow as tf


def create_model_scratch(input_shape, output_shape):
    model = tf.keras.models.Sequential()

    # First convolutional block
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    # Second convolutional block
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    # Third convolutional block
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    # Fourth convolutional block
    model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    # Flatten and fully connected layers
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(output_shape, activation='softmax'))

    # Compile the model
    opt = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# In this updated code, we increased the depth of the network by adding more convolutional blocks,
# reduced the dropout rate, changed the optimization algorithm, and added batch normalization layers.
# However, these changes may not be enough to achieve state-of-the-art performance, and further experimentation may be required.


# ReLU is a popular choice for activation functions because it is computationally efficient and has been shown to work well in practice. It has the form f(x) = max(0, x), meaning that it outputs 0 for any negative input values, and outputs the input value directly for any non-negative input values.