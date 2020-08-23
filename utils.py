import pathlib
import PIL
from PIL import Image
import numpy as np
from defines import HEIGHT, WIDTH


def to_grayscale(image):
    gs_img = remove_transparency(image).convert('L')
    return gs_img
    # img.save('greyscale.png')


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


# make_grayscale_folder("../data/w/", "../data/output/")
