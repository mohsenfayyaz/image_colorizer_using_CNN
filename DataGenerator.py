import pathlib
import time

import PIL
import PIL.Image
import numpy as np
from utils import img_to_np_array, to_grayscale, resize


class DataGenerator:
    def __init__(self, input_directory):
        self.input_directory = input_directory

    def get_train(self, ):
        data_dir = pathlib.Path(self.input_directory)
        input_images_url = list(data_dir.glob('*'))
        print("#Images:", len(input_images_url))
        t = time.time()
        train_x = np.asarray([img_to_np_array(to_grayscale(resize(PIL.Image.open(str(image_url))))) for image_url in input_images_url])
        print("train_x ready")
        train_y = np.asarray([img_to_np_array(resize(PIL.Image.open(str(image_url)))) for image_url in input_images_url])
        print("train_y ready")
        train_x = np.expand_dims(train_x, -1)
        print(train_x.shape, train_y.shape)
        # for image_url in input_images_url:
        #     current_image = PIL.Image.open(str(image_url))
        #     train_x.append(img_to_np_array(to_grayscale(current_image)))
        #     train_y.append(img_to_np_array(current_image))
        elapsed = time.time() - t
        print(elapsed)
        return train_x, train_y
