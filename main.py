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
from skimage import color
from inception import inception_model, inception_1ch_to_3ch

# import tensorflow_datasets as tfds

INPUT_DIRECTORY = "../data/input"
OUTPUT_DIRECTORY = "../data/output"
# MODEL_FOLDER = "save\\CONV_TRANSPOSE\\"
MODEL_FOLDER = "save\\inception3_trainable2\\"
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
    model.add(tf.keras.layers.Conv2DTranspose(2, (4, 4), padding="valid"))
    model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.mean_squared_error,
                  metrics=["accuracy"])
    print(model.summary())
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


def get_model(use_saved=True, type="inception"):
    if use_saved:
        return tf.keras.models.load_model(MODEL_FOLDER + 'model_' + str(LOAD_MODEL_OFFSET) + '.h5')
    else:
        if type == "inception":
            return inception_model()
        else:
            return cnn_model()


global_train_x = None
global_train_y = None


class CustomSaver(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if epoch % 10 == 0:  # or save after some epoch, each k-th epoch etc.
            offset = 0 if LOAD_MODEL_OFFSET is None else LOAD_MODEL_OFFSET + 1
            show_predict(self.model, global_train_x[285], epoch + offset)
            show_predict(self.model, global_train_x[414], epoch + offset + 0.1)
            self.model.save(MODEL_FOLDER + 'model_' + str(epoch + offset) + ".h5")


def show_predict(model, input_img, epoch=0):
    pred_ch2 = model.predict(np.expand_dims(input_img, 0))[0] * 200 - 100
    print(pred_ch2)
    print(pred_ch2.shape)
    pred_ch3 = np.zeros((HEIGHT, WIDTH, 3)) + 50
    pred_ch3[:, :, 1] = pred_ch2[:, :, 0]
    pred_ch3[:, :, 2] = pred_ch2[:, :, 1]
    # image = show_image_array(global_train_y[64], CIELAB=True)
    # image.save(MODEL_FOLDER + str(epoch) + "base.jpg")
    image = show_image_array(pred_ch3, CIELAB=True)
    image.save(MODEL_FOLDER + str(epoch) + "mask.jpg")

    input_lab = np.squeeze(color.rgb2lab(color.gray2rgb(input_img)))
    pred_ch3_combined_light = pred_ch3
    print(pred_ch3_combined_light.shape, input_lab.shape)
    pred_ch3_combined_light[:, :, 0] = input_lab[:, :, 0]

    image = show_image_array(pred_ch3_combined_light, CIELAB=True)
    image.save(MODEL_FOLDER + str(epoch) + ".jpg")


def fit_on_inception(train_x, train_y):
    model = get_model(use_saved=False if LOAD_MODEL_OFFSET is None else True, type="inception")
    inception_train_x = inception_1ch_to_3ch(train_x)
    global global_train_x
    global_train_x = inception_train_x
    print(inception_train_x.shape)
    show_predict(model, inception_train_x[0])
    saver = CustomSaver()
    model.fit(x=inception_train_x, y=(train_y[:, :, :, 1:] + 100) / 200, validation_split=0.07, batch_size=20, epochs=5000, callbacks=[saver])


def main():
    train_x, train_y = get_train(use_saved_npy=True)
    print(train_x.shape, train_y.shape)
    global global_train_x, global_train_y
    global_train_x = train_x
    global_train_y = train_y
    fit_on_inception(train_x, train_y)
    # show_image_array(train_y[0])
    # show_image_array(train_y[0], CIELAB=True)
    # image = PIL.Image.fromarray(np.uint8(np.squeeze(train_x[10], -1)))
    # image.show()
    # image = PIL.Image.fromarray(np.uint8(train_y[10]))
    # image.show()


main()
