import pathlib

import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf

# import tensorflow_datasets as tfds

INPUT_DIRECTORY = "../data/output"


def cnn_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(16, (3, 3), padding="same"))
    model.add(tf.keras.activations.relu)
    model.add(tf.keras.layers.Conv2D(32, (4, 4), padding="same"))
    model.add(tf.keras.activations.relu)
    model.add(tf.keras.layers.Conv2D(128, (8, 8), padding="same"))
    model.add(tf.keras.activations.relu)
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Conv2D(64, (16, 16), padding="same"))
    model.add(tf.keras.activations.relu)
    model.add(tf.keras.layers.Conv2D(16, (16, 16), padding="same"))
    model.add(tf.keras.activations.relu)
    model.add(tf.keras.layers.Conv2D(3, (32, 32), padding="same"))
    model.add(tf.keras.activations.relu)
    model.compile(tf.keras.optimizers.Adam, loss=tf.keras.losses.mean_squared_error, metrics=["accuracy"])
    return model


def main():
    data_dir = pathlib.Path(INPUT_DIRECTORY)
    input_images_url = list(data_dir.glob('*'))
    print(len(input_images_url))
    p = PIL.Image.open(str(input_images_url[10]))
    p.show()

    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size

    model = cnn_model()
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=valid_generator,
                        validation_steps=STEP_SIZE_VALID,
                        epochs=10)

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
