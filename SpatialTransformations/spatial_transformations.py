#!/usr/bin/env python3
"""
CIS 693 - Spatial Transformations Homework
Author: John Oyster
CSU ID: 2398851
Description:
    List of Functions:
    1. Convert to Grayscale
    2. Negative Transformation

"""

from PIL import Image
import numpy as np


class UnsupportedModeError(Exception):
    pass


def convert_to_grayscale(image):
    """

    :param image:
    :return:
    """
    if 'RGB' in image.mode:
        image_data = np.asarray(image)
        image_data.dot([0.2989, 0.5870, 0.1140])
    else:
        raise UnsupportedModeError

    new_image = Image.fromarray(np.uint8(image_data))
    return new_image


def negative_transform(image):
    """
    s = L - 1 - r
    :return:  inverged image file
    :rtype:     Image
    """
    if 'L' or 'RGB' in image.mode:
        image_data = np.asarray(image)
        image_data = 255 - image_data
    else:
        raise UnsupportedModeError

    new_image = Image.fromarray(np.uint8(image_data))
    return new_image


if __name__ == '__main__':
    image_file = Image.open("./Cleveland.jpg")
    #image_file.show()
    gray_image = convert_to_grayscale(image_file)
    gray_image.show()
    image_file = image_file.convert('LA')
    image_file.show()
    negative_image = negative_transform(image_file)
    #negative_image.show()
