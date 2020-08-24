import pathlib

import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from DataGenerator import DataGenerator
from defines import HEIGHT, WIDTH
from utils import show_image_array, array_to_color_mask
# import tensorflow_datasets as tfds

INPUT_DIRECTORY = "../data/input"
OUTPUT_DIRECTORY = "../data/w"
MODEL_FOLDER = "save\\CONV_TRANSPOSE\\"
LOAD_MODEL_OFFSET = None


def cnn_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(HEIGHT, WIDTH, 1)))
    model.add(tf.keras.layers.Conv2D(16, (4, 4), padding="valid"))
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    model.add(tf.keras.layers.Conv2D(32, (8, 8), padding="valid"))
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    model.add(tf.keras.layers.Conv2D(128, (16, 16), padding="valid"))
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Conv2DTranspose(128, (16, 16), padding="valid"))
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    model.add(tf.keras.layers.Conv2DTranspose(32, (8, 8), padding="valid"))
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    model.add(tf.keras.layers.Conv2DTranspose(3, (4, 4), padding="valid"))
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss=tf.keras.losses.mean_squared_error,
                  metrics=["accuracy"])
    return model


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
        return tf.keras.models.load_model(MODEL_FOLDER + 'model_' + str(LOAD_MODEL_OFFSET) + '.h5')
    else:
        return cnn_model()


global_train_x = None


class CustomSaver(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        offset = 0 if LOAD_MODEL_OFFSET is None else LOAD_MODEL_OFFSET + 1
        show_predict(self.model, global_train_x[0], epoch + offset)
        if epoch % 1 == 0:  # or save after some epoch, each k-th epoch etc.
            self.model.save(MODEL_FOLDER + 'model_' + str(epoch + offset) + ".h5")


def show_predict(model, input_img, epoch=0):
    p = model.predict(np.expand_dims(input_img, 0))
    print(p[0])
    print(p[0].shape)
    image = show_image_array(p[0])
    image.save(MODEL_FOLDER + str(epoch) + ".jpg")


def main():
    train_x, train_y = get_train(use_saved_npy=False)
    print(train_x.shape, train_y.shape)
    global global_train_x
    global_train_x = train_x

    model = get_model(use_saved=False if LOAD_MODEL_OFFSET is None else True)
    print(model.summary())
    show_predict(model, train_x[0])
    show_image_array(array_to_color_mask(train_y[0]))

    image = PIL.Image.fromarray(np.uint8(np.squeeze(train_x[10], -1)))
    image.show()
    image = PIL.Image.fromarray(np.uint8(train_y[10]))
    image.show()
    # print(model.summary())
    saver = CustomSaver()
    model.fit(x=train_x, y=train_y, validation_split=0.15, batch_size=5, epochs=30, callbacks=[saver])

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
