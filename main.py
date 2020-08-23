import pathlib

import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from DataGenerator import DataGenerator

# import tensorflow_datasets as tfds

INPUT_DIRECTORY = "../data/output"


def cnn_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(50, 50, 1)))
    model.add(tf.keras.layers.Conv2D(16, (3, 3), padding="same"))
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    model.add(tf.keras.layers.Conv2D(32, (4, 4), padding="same"))
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    model.add(tf.keras.layers.Conv2D(128, (8, 8), padding="same"))
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Conv2D(64, (16, 16), padding="same"))
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    model.add(tf.keras.layers.Conv2D(16, (16, 16), padding="same"))
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    model.add(tf.keras.layers.Conv2D(3, (32, 32), padding="same"))
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mean_squared_error, metrics=["accuracy"])
    return model


class CustomSaver(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if epoch % 2 == 0:  # or save after some epoch, each k-th epoch etc.
            self.model.save("./save/model_" + epoch + ".hd5".format(epoch))


def main():
    train_x, train_y = DataGenerator(INPUT_DIRECTORY).get_train()
    model = cnn_model()
    # image = PIL.Image.fromarray(np.uint8(train_x[0]))
    # image.show()
    print(train_x[0])
    # print(model.summary())
    model.fit(x=train_x, y=train_y, validation_split=0.2, batch_size=10, epochs=30)

    # tf.keras.preprocessing.image_dataset_from_directory(
    #     INPUT_DIRECTORY,
    #     labels="inferred",
    #     label_mode="int",
    #     class_names=None,
    #     color_mode="rgb",
    #     batch_size=32,
    #     image_size=(256, 256),
    #     shuffle=True,
    #     seed=None,
    #     validation_split=None,
    #     subset=None,
    #     interpolation="bilinear",
    #     follow_links=False,
    # )


main()
