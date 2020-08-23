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
