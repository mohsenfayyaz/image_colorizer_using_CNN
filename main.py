import pathlib

import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from DataGenerator import DataGenerator
from defines import HEIGHT, WIDTH

# import tensorflow_datasets as tfds

INPUT_DIRECTORY = "../data/w"
OUTPUT_DIRECTORY = "../data/output"


def cnn_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(HEIGHT, WIDTH, 3)))
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
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss=tf.keras.losses.mean_squared_error,
                  metrics=["accuracy"])
    return model


class CustomSaver(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if epoch % 1 == 0:  # or save after some epoch, each k-th epoch etc.
            self.model.save("save\\model_" + str(epoch) + ".h5")


def get_train(use_saved_npy=True):
    train_x = None
    train_y = None
    if use_saved_npy:
        train_x = np.load('train_x.npy')
        train_y = np.load('train_y.npy')
    else:
        train_x, train_y = DataGenerator(INPUT_DIRECTORY, OUTPUT_DIRECTORY).get_train()
        np.save('train_x.npy', train_x)
        np.save('train_y.npy', train_y)
    return train_x, train_y


def get_model(use_saved=True):
    if use_saved:
        return tf.keras.models.load_model('save\\model_2.h5')
    else:
        return cnn_model()


def main():
    train_x, train_y = get_train(use_saved_npy=False)
    model = get_model(use_saved=False)
    image = PIL.Image.fromarray(np.uint8(train_x[10]))
    image.show()
    image = PIL.Image.fromarray(np.uint8(train_y[10]))
    image.show()
    # print(model.summary())
    saver = CustomSaver()
    model.fit(x=train_x, y=train_y, validation_split=0.2, batch_size=5, epochs=30, callbacks=[saver])

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
