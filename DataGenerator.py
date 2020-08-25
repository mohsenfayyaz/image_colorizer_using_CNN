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
        return train_x

    def get_train(self, input_directory, output_directory):
        data_dir = pathlib.Path(output_directory)
        output_images_url = list(data_dir.glob('*'))
        t = time.time()
        train_x = self.prepare_x(input_directory)
        # train_x = np.array([np.array(PIL.Image.open(image_url)) for image_url in input_images_url])
        print("train_x ready")
        train_y = np.asarray(
            [img_to_np_array(resize(PIL.Image.open(str(image_url)))) for image_url in output_images_url])
        train_y = color.rgb2lab(train_y)
        train_y[:, :, :, 0] = [50]  # ignore the light it can be inferred later
        # train_x /= 255
        # train_y /= 255
        # print("mid train_y")
        # train_y = np.array([np.array(PIL.Image.open(image_url)) for image_url in output_images_url])
        print("train_y ready")
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