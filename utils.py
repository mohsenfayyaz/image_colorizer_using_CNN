import pathlib
import PIL
from PIL import Image
import numpy as np
from defines import HEIGHT, WIDTH
from skimage import io, color
import tensorflow as tf


def show_image_array(image_array, CIELAB=False):
    try:
        if CIELAB:
            image = PIL.Image.fromarray(np.uint8(color.lab2rgb(image_array) * 255))
        else:
            image = PIL.Image.fromarray(np.uint8(image_array))
    except Exception:
        image = PIL.Image.fromarray(np.uint8(np.squeeze(image_array, -1)))
    # image.show()
    return image


def array_to_image(image_array, CIELAB=False):
    try:
        if CIELAB:
            image = PIL.Image.fromarray(np.uint8(color.lab2rgb(image_array) * 255))
        else:
            image = PIL.Image.fromarray(np.uint8(image_array))
    except Exception:
        image = PIL.Image.fromarray(np.uint8(np.squeeze(image_array, -1)))
    return image


def to_grayscale(image):
    gs_img = remove_transparency(image).convert('L')
    return gs_img
    # img.save('greyscale.png')


def array_to_color_mask(image_array: np.array):
    # color_mask = image_array / np.expand_dims(np.average(image_array + 1, axis=-1), axis=-1) * 200
    lab = color.rgb2lab(image_array)
    print(lab)
    return lab


def img_to_np_array(image):
    return np.array(image)


def resize(image):
    return image.resize((WIDTH, HEIGHT))


def remove_transparency(im, bg_colour=(255, 255, 255)):
    # Only process if image has transparency
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
        # Need to convert to RGBA if LA format due to a bug in PIL
        alpha = im.convert('RGBA').split()[-1]
        # Create a new background image of our matt color.
        # Must be RGBA because paste requires both images have the same format
        bg = Image.new("RGBA", im.size, bg_colour + (255,))
        bg.paste(im, mask=alpha)
        return bg
    else:
        return im


def make_grayscale_folder(src_folder, out_folder):
    data_dir = pathlib.Path(src_folder)
    input_images_url = list(data_dir.glob('*'))
    print("#Images:", len(input_images_url))
    for image_url in input_images_url:
        pil = PIL.Image.open(str(image_url))
        path = pil.filename
        print(path.split("\\"))
        filename = path.split("\\")[-1]
        to_grayscale(pil).save(out_folder + filename)


def change_model(model, new_input_shape=(None, 40, 40, 3)):
    # replace input shape of first layer
    model._layers[0].batch_input_shape = new_input_shape

    # rebuild model architecture by exporting and importing via json
    new_model = tf.keras.models.model_from_json(model.to_json())

    # copy weights from old model to new one
    for layer in new_model.layers:
        try:
            layer.set_weights(model.get_layer(name=layer.name).get_weights())
            print("Loaded layer {}".format(layer.name))
        except:
            print("Could not transfer weights for layer {}".format(layer.name))

    return new_model


# make_grayscale_folder("../data/output2/", "../data/input2/")
