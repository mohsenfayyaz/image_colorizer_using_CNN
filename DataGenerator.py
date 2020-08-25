import pathlib
import time

import PIL
import PIL.Image
import numpy as np
from utils import img_to_np_array, to_grayscale, resize, array_to_color_mask, show_image_array
from skimage import color


class DataGenerator:
    def __init__(self):
        pass

    def prepare_x(self, input_directory, do_resize=True):
        data_dir = pathlib.Path(input_directory)
        input_images_url = list(data_dir.glob('*'))
        if do_resize:
            train_x = np.asarray([np.array(to_grayscale(resize(PIL.Image.open(str(image_url))))) for image_url in input_images_url])
            print(train_x.shape)
            train_x = np.expand_dims(train_x, -1)
        else:
            train_x = [np.array(to_grayscale(PIL.Image.open(str(image_url)))) for image_url in input_images_url]
        print("#Images:", len(input_images_url))
        print("train_x ready")
        return train_x

    def prepare_y(self, output_directory):
        data_dir = pathlib.Path(output_directory)
        output_images_url = list(data_dir.glob('*'))
        train_y = np.asarray([np.array(resize(PIL.Image.open(str(image_url)))) for image_url in output_images_url])
        # for p, u in zip(train_y, output_images_url):
        #     try:
        #         if p.shape[2] != 3:
        #             print(p.shape, u)
        #     except:
        #         print(p.shape, u)
        print(train_y.shape)
        for i, t in enumerate(train_y):
            print(output_images_url[i])
            train_y[i] = color.rgb2lab(train_y[i])
        train_y[:, :, :, 0] = [50]  # ignore the light it can be inferred later
        print("train_y ready")
        return train_y

    def get_train(self, input_directory, output_directory):
        t = time.time()
        train_y = self.prepare_y(output_directory)
        train_x = self.prepare_x(input_directory)
        elapsed = time.time() - t
        print(elapsed)
        return train_x, train_y


def test():
    url = "../data/w/20190327_040225.jpg"
    image = np.array(resize(PIL.Image.open(str(url))))
    show_image_array(image)
    image = np.expand_dims(image, axis=0)
    i = color.rgb2lab(image)
    i = i[0]
    show_image_array(i)
    i[:, :, 0] = [10]
    print(i)
    s = color.lab2rgb(i) * 255
    show_image_array(s)
    print(i.shape)
    train_y = np.load('train_y.npy')
    print(train_y[0])

# test()