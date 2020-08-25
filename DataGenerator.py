import pathlib
import time
import tensorflow as tf
import PIL
import PIL.Image
import numpy as np
from utils import img_to_np_array, to_grayscale, resize, array_to_color_mask, show_image_array
from skimage import color
from defines import HEIGHT, WIDTH


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, list_IDs, labels, batch_size=32, dim=(HEIGHT, WIDTH, 3), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)


class DataGeneratorSimple:
    def __init__(self):
        pass

    def prepare_x(self, input_directory, do_resize=True):
        data_dir = pathlib.Path(input_directory)
        input_images_url = list(data_dir.glob('*'))
        if do_resize:
            train_x = np.asarray(
                [np.array(to_grayscale(resize(PIL.Image.open(str(image_url))))) for image_url in input_images_url])
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
            if i % 100 == 0:
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
