#!/usr/bin/env python3
"""CIS 693 - Spatial Transformations Homework.

Author: John Oyster
Date:   May 22, 2020
Description:
    List of Functions:
    1. Convert to Grayscale
    2. Negative Transformation
    3. Logarithmic Transformation
    4. Gamma Transformation

"""
import cv2
import numpy as np


def convert_to_grayscale(image):
    """Convert RGB Image to grayscale using RGB weights with dot product.

    :param image:   Original Image
    :type:          numpy.ndarray
    :return:        Grayscale Image
    :rtype:         numpy.ndarray
    """
    rgb_weights = [0.2989, 0.5870, 0.1140]
    new_image = np.dot(image[..., :3], rgb_weights)
    new_image = new_image.astype(np.uint8)

    return new_image


def negative_transform(image):
    """Grayscale negative image transformation.

    :param image:   Original Image
    :type:          numpy.ndarray
    :return:        inverted image file
    :rtype:         numpay.ndarray
    """
    L = 2 ** int(round(np.log2(image.max())))       # Input gray level
    new_image = (L - 1) - image                     # s = L - 1 - r
    new_image = new_image.astype(np.uint8)

    return new_image


def log_transform(image):
    """Logarithmic Transformation of grayscale image.

    :param image:   Original Image
    :type:          numpy.ndarray
    :return:        log transformed image file
    :rtype:         numpy.ndarray
    """
    # Selecting c to help normalize r
    c = np.iinfo(image.dtype).max / np.log(1 + np.max(image))
    new_image = c * np.log(1 + image)
    new_image = new_image.astype(np.uint8)

    return new_image


def gamma_transformation(image, gamma=1.0):
    """Power-Law (Gamma) Transformation of grayscale image.

    :param image:   Original Image
    :type:          numpy.ndarray
    :param gamma:   Gamma value to apply
    :type:          float
    :return:        gamma transformed image file
    :rtype:         numpy.ndarray
    """
    c = 255                     # TODO(John): Figure out a calculation for c
    norm_image = image / np.max(image)
    new_image = c * np.power(norm_image, gamma)
    new_image = new_image.astype(np.uint8)

    return new_image


if __name__ == '__main__':
    image_file = cv2.imread("./Cleveland.jpg")
    cv2.imshow('Original Image', image_file)
    cv2.waitKey(0)
    gray_image = convert_to_grayscale(image_file)
    cv2.imshow('Grayscale Image', gray_image)
    cv2.waitKey(0)
    negative_image = negative_transform(gray_image)
    cv2.imshow('Inverted Image', negative_image)
    cv2.waitKey(0)
    log_image = log_transform(gray_image)
    cv2.imshow('Logarithmic Transformed Image', log_image)
    cv2.waitKey(0)
    gamma_image = gamma_transformation(gray_image, 2.0)
    cv2.imshow('Gamma Correction', gamma_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
