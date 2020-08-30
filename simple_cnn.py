import tensorflow as tf
from defines import WIDTH, HEIGHT


def cnn_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(HEIGHT, WIDTH, 3)))
    model.add(tf.keras.layers.Conv2D(16, (4, 4), padding="valid"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.Conv2D(32, (8, 8), padding="valid"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Conv2D(128, (16, 16), padding="valid"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())
    # model.add(tf.keras.layers.Conv2D(128, (16, 16), padding="valid"))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.ReLU())
    # model.add(tf.keras.layers.Conv2D(64, (32, 32), padding="valid", activation="relu"))
    # model.add(tf.keras.layers.Conv2DTranspose(64, (32, 32), padding="valid", activation="relu"))
    # model.add(tf.keras.layers.Conv2DTranspose(32, (16, 16), padding="valid", activation="relu"))
    model.add(tf.keras.layers.Conv2DTranspose(16, (16, 16), padding="valid", activation="relu"))
    model.add(tf.keras.layers.Conv2DTranspose(8, (9, 9), padding="valid", strides=(2, 2), activation="relu"))
    model.add(tf.keras.layers.Conv2DTranspose(2, (4, 4), padding="valid"))
    model.add(tf.keras.layers.ReLU(max_value=200, negative_slope=0))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.mean_squared_error,
                  metrics=["accuracy"])
    print(model.summary())
    return model