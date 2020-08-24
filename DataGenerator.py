import pathlib
import time

import PIL
import PIL.Image
import numpy as np
from utils import img_to_np_array, to_grayscale, resize, array_to_color_mask, show_image_array
from skimage import color


class DataGenerator:
    def __init__(self, input_directory, output_directory):
        self.input_directory = input_directory
        self.output_directory = output_directory

    def get_train(self, ):
        data_dir = pathlib.Path(self.input_directory)
        input_images_url = list(data_dir.glob('*'))
        data_dir = pathlib.Path(self.output_directory)
        output_images_url = list(data_dir.glob('*'))
        print("#Images:", len(input_images_url))
        t = time.time()
        train_x = np.asarray(
            [img_to_np_array(resize(PIL.Image.open(str(image_url)))) for image_url in input_images_url])
        # train_x = np.array([np.array(PIL.Image.open(image_url)) for image_url in input_images_url])
        print("train_x ready")
        train_y = np.asarray(
            [img_to_np_array(resize(PIL.Image.open(str(image_url)))) for image_url in output_images_url])
        train_y = color.rgb2lab(train_y)
        # train_x /= 255
        # train_y /= 255
        # print("mid train_y")
        # train_y = np.array([np.array(PIL.Image.open(image_url)) for image_url in output_images_url])
        print("train_y ready")
        train_y = np.expand_dims(train_y, -1)
        # for image_url in input_images_url:
        #     current_image = PIL.Image.open(str(image_url))
        #     train_x.append(img_to_np_array(to_grayscale(current_image)))
        #     train_y.append(img_to_np_array(current_image))
        elapsed = time.time() - t
        print(elapsed)
        return train_x, train_y


def test():
    url = "../data/w/20190327_040225.jpg"
    image = np.array(resize(PIL.Image.open(str(url))))
    show_image_array(image)
    i = color.rgb2lab(image)
    show_image_array(i)
    i[:, :, 0] = [70]
    print(i)
    s = color.lab2rgb(i) * 255
    show_image_array(s)
    print(i.shape)

test()