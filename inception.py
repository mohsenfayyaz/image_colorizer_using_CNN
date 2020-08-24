import tensorflow as tf
from defines import WIDTH, HEIGHT
import numpy as np

def inception_model():
    model = tf.keras.applications.InceptionV3(
        include_top=False,
        weights="imagenet",
        input_shape=(HEIGHT, WIDTH, 3),
    )
    for layer in model.layers:
        layer.trainable = True

    x = model.layers[-3].output
    # x = tf.keras.layers.Conv2D(128, (8, 8), padding="same", activation="relu")(x)
    # x = tf.keras.layers.ReLU(max_value=200, negative_slope=0)(x)
    x = tf.keras.layers.Conv2DTranspose(64, (16, 16), strides=(4, 3), padding="valid")(x)
    x = tf.keras.layers.ReLU(max_value=2, negative_slope=0)(x)
    x = tf.keras.layers.Conv2DTranspose(32, (8, 8), strides=(2, 2), padding="valid")(x)
    x = tf.keras.layers.ReLU(max_value=2, negative_slope=0)(x)
    x = tf.keras.layers.Conv2DTranspose(2, (15, 44), strides=(3, 4), padding="valid")(x)
    x = tf.keras.layers.ReLU(max_value=2, negative_slope=0)(x)
    # x = tf.keras.layers.Conv2DTranspose(2, (7, 7), padding="valid")(x)
    # x = tf.keras.layers.Activation(tf.keras.activations.relu)(x)
    model = tf.keras.Model(inputs=model.inputs, outputs=x)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
                  loss=tf.keras.losses.mean_squared_error,
                  metrics=["accuracy"])
    print(model.summary())
    return model


inception_model()
