import gc
import pathlib

import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from DataGenerator import DataGenerator, DataGeneratorSimple
from defines import HEIGHT, WIDTH
from test_model import test_model_on_directory, get_model_prediction
from utils import show_image_array, array_to_color_mask
from skimage import color
from inception import inception_model, inception_1ch_to_3ch

# import tensorflow_datasets as tfds

INPUT_DIRECTORY = "../data/input3"
OUTPUT_DIRECTORY = "../data/output3"
# MODEL_FOLDER = "save\\CONV_TRANSPOSE\\"
MODEL_FOLDER = "save\\inception3_trainable_int\\"
LOAD_MODEL_OFFSET = None  # set this to None to start a new model
USE_SAVED_NPY = True
TRAIN_X_NPY = 'train_x.npy'
TRAIN_Y_NPY = 'train_y.npy'
START_IMAGE_INDEX = 4400  # if you have larger RAM you can increase the range
END_IMAGE_INDEX = 10400  # max is 10555 in my dataset
SAVING_MODEL_EPOCH_INTERVAL = 5


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
        # train_x = inception_1ch_to_3ch(np.load(TRAIN_X_NPY)[START_IMAGE_INDEX:END_IMAGE_INDEX, :, :, :])
        train_x = inception_1ch_to_3ch(np.load(TRAIN_X_NPY)[:, :, :, :])
        print("train_x loaded")
        gc.collect()
        # train_y = (np.load(TRAIN_Y_NPY)[START_IMAGE_INDEX:END_IMAGE_INDEX, :, :, 1:] + 100) / 200
        train_y = (np.load(TRAIN_Y_NPY)[:, :, :, 1:] + 100)
        # train_y = np.load(TRAIN_Y_NPY)
        print(train_y.nbytes/1000000)
        gc.collect()
        print("train_y loaded")
    else:
        train_x, train_y = DataGeneratorSimple().get_train(INPUT_DIRECTORY, OUTPUT_DIRECTORY)
        np.save(TRAIN_X_NPY, train_x)
        np.save(TRAIN_Y_NPY, train_y)
    return train_x, train_y


def get_model(use_saved=True, type="inception"):
    if use_saved:
        return tf.keras.models.load_model(MODEL_FOLDER + 'model_' + str(LOAD_MODEL_OFFSET) + '.h5')
    else:
        if type == "inception":
            return inception_model()
        else:
            return cnn_model()


global_train_x = []
global_train_y = []


class CustomSaver(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if epoch % SAVING_MODEL_EPOCH_INTERVAL == 0:  # or save after some epoch, each k-th epoch etc.
            offset = 0 if LOAD_MODEL_OFFSET is None else LOAD_MODEL_OFFSET + 1
            show_predict(self.model, global_train_x[5000], epoch + offset)
            show_predict(self.model, global_train_x[800], epoch + offset + 0.1)
            self.model.save(MODEL_FOLDER + 'model_' + str(epoch + offset) + ".h5")
            test_model_on_directory(self.model, "../data/test/test", "../data/test/")


def show_predict(model, input_img, epoch=0):
    image, mask = get_model_prediction(model, input_img)
    mask.save(MODEL_FOLDER + str(epoch) + "mask.jpg")
    image.save(MODEL_FOLDER + str(epoch) + ".jpg")


def fit_on_inception(train_x, train_y):
    model = get_model(use_saved=False if LOAD_MODEL_OFFSET is None else True, type="inception")

    inception_train_x = train_x
    # del train_x
    global global_train_x_examples
    global_train_x_examples = [inception_train_x[0], inception_train_x[100]]
    print("inception_train_x.shape:", inception_train_x.shape)
    show_predict(model, inception_train_x[0])
    test_model_on_directory(model, "../data/test/test", "../data/test/")

    saver = CustomSaver()
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
    model.fit(x=inception_train_x, y=train_y, validation_split=0.04, batch_size=32, epochs=10000, callbacks=[saver, es])


def main():
    train_x, train_y = get_train(use_saved_npy=USE_SAVED_NPY)
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

    # train_y = (np.load(TRAIN_Y_NPY)[:, :, :, 1:] + 100) / 200
    # np.save("yyy.npy", train_y)

main()
