#!/usr/bin/env python3
"""CIS 693 - Term Project.

Author: John Oyster
Date:   June 24, 2020
Description:

    This project will look at an implementation of LSB Steganography

    DISCLAIMER: Comment text is taken from course handouts and is copyright
        2020, Dr. Almabrok Essa, Cleveland State University,
    Objectives:



"""
#  Copyright (c) 2020. John Oyster in agreement with Cleveland State University.
import numpy as np
import cv2
from matplotlib import pyplot as plt


def get_image(filename, grayscale=False):
    """Load an image into a ndarray using OpenCV.

    :param filename:        File path and name to load
    :type:                  str
    :param grayscale:       Set true to return image in grayscale
    :type:                  bool
    :return:                Grayscale Image
    :rtype:                 numpy.ndarray
    """
    # Read in image
    image = cv2.imread(filename)

    # Convert to grayscale if needed
    if grayscale and len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    return image


def compute_max_message_length(image):
    size_x, size_y = image.shape
    return max_length


def display_side_by_side(image1, image2, grayscale=False):
    """Show two images side by side.

    :param image1:      Image 1
    :type:              numpy.ndarray
    :param image2:      Image 2
    :type:              numpy.ndarray
    """
    # Normalize color to match
    if grayscale and len(image1.shape) > 2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB)
    if grayscale and len(image2.shape) > 2:
        image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2RGB)

    # Create subplots
    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(image1)
    fig.add_subplot(1, 2, 2)
    plt.imshow(image2)
    plt.show(block=True)


def main():
    """Execute this routine if this file is called directly.

    This function is used to test the parameters of the SIFT method
    and make sure that it works.

    :return:        Errno = 0 if good
    :rtype:         int
    """
    clean_image = get_image("./Data/Cleveland.jpg")

    # Display results
    display_side_by_side(clean_image, clean_image)


if __name__ == '__main__':
    main()

    cv2.destroyAllWindows()
