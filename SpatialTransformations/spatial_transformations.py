#!/usr/bin/env python3
"""
CIS 693 - Spatial Transformations Homework
Author: John Oyster
CSU ID: 2398851
Date:   May 22, 2020
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
    """Convert RGB Image to grayscale using RGB wieghts with dot product.

    :param image:   Original Image
    :type:          PIL.Image
    :return:        Grayscale Image
    :rtype:         PIL.Image
    """
    rgb_weights = [0.2989, 0.5870, 0.1140]
    if 'RGB' in image.mode:
        image_data = np.asarray(image)
        gray_image = np.dot(image_data[..., :3], rgb_weights)
    else:
        raise UnsupportedModeError

    new_image = Image.fromarray(np.uint8(gray_image))
    return new_image


def negative_transform(image):
    """Grayscale negative image transformation.
    :param image:   Original Image
    :type:          PIL.Image
    :return:        inverged image file
    :rtype:         PIL.Image
    """
    L = 2 ** image.bits                     # Input gray level
    if 'L' or 'RGB' in image.mode:
        image_data = np.asarray(image)
        image_data = (L - 1) - image_data   # s = L - 1 - r
    else:
        raise UnsupportedModeError

    new_image = Image.fromarray(np.uint8(image_data))
    return new_image


if __name__ == '__main__':
    image_file = Image.open("./Cleveland.jpg")
    image_file.show()
    gray_image = convert_to_grayscale(image_file)
    gray_image.show()
    negative_image = negative_transform(image_file)
    negative_image.show()
