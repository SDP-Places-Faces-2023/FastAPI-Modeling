import tensorflow as tf


def create_model_scratch(input_shape, output_shape, learning_rate=0.01, dropout_rate=0.25):
    base_model = tf.keras.models.Sequential()

    # First convolutional block
    base_model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    base_model.add(tf.keras.layers.BatchNormalization())
    base_model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    base_model.add(tf.keras.layers.Dropout(dropout_rate))

    # Second convolutional block
    base_model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    base_model.add(tf.keras.layers.BatchNormalization())
    base_model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    base_model.add(tf.keras.layers.Dropout(dropout_rate))

    # Third convolutional block
    base_model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    base_model.add(tf.keras.layers.BatchNormalization())
    base_model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    base_model.add(tf.keras.layers.Dropout(dropout_rate))

    # Fourth convolutional block
    base_model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu'))
    base_model.add(tf.keras.layers.BatchNormalization())
    base_model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    base_model.add(tf.keras.layers.Dropout(dropout_rate))

    # Flatten and fully connected layers
    base_model.add(tf.keras.layers.Flatten())
    base_model.add(tf.keras.layers.Dense(512, activation='relu'))
    base_model.add(tf.keras.layers.BatchNormalization())
    base_model.add(tf.keras.layers.Dropout(dropout_rate))

    # Freeze the layers of the base model
    base_model.trainable = False

    # Create the final model by adding a dense layer for the number of classes
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs)
    outputs = tf.keras.layers.Dense(output_shape, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
