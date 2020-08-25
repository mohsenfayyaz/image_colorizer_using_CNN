import pathlib
import PIL
import numpy as np
from skimage import color
from defines import HEIGHT, WIDTH
from inception import inception_1ch_to_3ch
from utils import resize, array_to_image, change_model
import tensorflow as tf
import PIL.Image
import os
from DataGenerator import DataGenerator


def test_model_on_directory(model: tf.keras.Model, inputs_directory, outputs_directory):
    data_dir = pathlib.Path(inputs_directory)
    input_images_url = list(data_dir.glob('*'))
    train_x = DataGenerator().prepare_x(inputs_directory, do_resize=True)
    for x, image_url in zip(train_x, input_images_url):
        x = np.asarray(x)
        # x = np.expand_dims(x, -1)
        x = inception_1ch_to_3ch(x)
        print(x.shape)
        print("colorizing " + str(image_url))
        h, w, ch = x.shape
        # model = change_model(model, new_input_shape=(None, h, w, ch))
        print(model.summary())
        image, mask_image = get_model_prediction(model, x)
        file_name = os.path.splitext(os.path.basename(image_url))[0]
        image.save(outputs_directory + file_name + "_color.jpg")
        mask_image.save(outputs_directory + file_name + "_color_mask.jpg")
    print("done")


def get_model_prediction(model, input_img):
    pred_ch2 = model.predict(np.expand_dims(input_img, 0))[0] * 200 - 100
    pred_ch3 = np.zeros((HEIGHT, WIDTH, 3)) + 50
    pred_ch3[:, :, 1] = pred_ch2[:, :, 0]
    pred_ch3[:, :, 2] = pred_ch2[:, :, 1]
    mask_image = array_to_image(pred_ch3, CIELAB=True)
    input_lab = np.squeeze(color.rgb2lab(color.gray2rgb(input_img)))
    pred_ch3_combined_light = pred_ch3
    pred_ch3_combined_light[:, :, 0] = input_lab[:, :, 0]
    image = array_to_image(pred_ch3_combined_light, CIELAB=True)
    return image, mask_image


def test_main():
    model = tf.keras.models.load_model("save\\inception3_trainable2\\model_1360.h5")
    test_model_on_directory(model, "../data/test/", "../data/test/")


test_main()
