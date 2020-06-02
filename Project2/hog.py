#!/usr/bin/env python3
"""CIS 693 - Project 2.

Author: John Oyster
Date:   June 6, 2020
Description:
    DISCLAIMER: Comment text is taken from course handouts and is copyright
        2020, Dr. Almabrok Essa, Cleveland State University,
    Objectives:
        1. Write a program to implement the Histogram of Orientated Gradients
           (HOG) Algorithm for pedestrian detection. The dataset used to
           evaluate the descriptor is “NICTA Pedestrian Dataset,” where it
           contains both the training set and the testing set. The training
           set contains 1000 positives samples (images contain pedestrians)
           and 2000 negatives samples (images do not contain pedestrians).
           The testing set includes 500 positive samples and 500 negative
           samples. Resize all images to 64 × 128 and use the following set
           of parameters:
            - Cell size [8 8]
            - Block size [16 16]
            - Gradient operators: G x = [-1 0 1] and G y = [-1 0 1] T
            - Number of orientation bins = 9
    Assumptions:
        1. Unless this statement is remove, 8-bit pixel values

"""
#  Copyright (c) 2020. John Oyster in agreement with Cleveland State University.
from enum import Enum
import os.path
import cv2
import numpy as np


# ----------------------------------------------------------------------------
# Define Test Images
def test_images():
    """Generator to return test images.

    :return:        Test Image Filename
    :rtype:         str
    """
    test_directory = './NICTA/TestSet/PositiveSamples'
    test_set = [
        'item_00000000.pnm'
    ]
    for image, _ in enumerate(test_set):
        yield os.path.join(test_directory, test_set[image])


def gamma_correction(image, gamma=1.0):
    """Power-Law (Gamma) Transformation of grayscale image.

    :param image:   Original Image
    :type:          numpy.ndarray
    :param gamma:   Gamma value to apply
    :type:          float
    :return:        gamma transformed image file
    :rtype:         numpy.ndarray
    """
    bits_per_pixel = np.iinfo(image.dtype).max
    norm_image = image / np.max(image)
    new_image = bits_per_pixel * np.power(norm_image, gamma)
    new_image = new_image.astype(np.uint8)

    return new_image


def compute_gradients(image):
    """
    - Cellsize[8 8]
    - Blocksize[16 16]
    - Gradient operators: Gx = [-1 0 1] and Gy = [-1 0 1]T
    - Number of orientation bins = 9

    :param image:
    :return:
    """
    kernel_prewitt = np.array([[1, 1, 1],
                               [0, 0, 0],
                               [-1, -1, -1]])
    gradient_x = cv2.filter2D(image, -1, kernel_prewitt)
    gradient_y = cv2.filter2D(image, -1, np.transpose(kernel_prewitt))

    magnitude = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
    angle = np.arctan2(gradient_y, gradient_x) * (180 / np.pi)
    return gradient_x, gradient_y, magnitude, angle


def compute_weighted_vote(image, gradient, block_size=(16, 16), cell_size=(8, 8)):

    image_size_x, image_size_y = image.shape
    print( (image_size_x/block_size[0] *1.5)+1 * (image_size_y/block_size[1] *1.5)+1)
    cell_count = image_size_x/cell_size[0] * image_size_y/cell_size[1]




if __name__ == '__main__':
    for my_image in test_images():
        # Step 1 - Input Image
        # Load in the test image, resize to 64x128, and convert to grayscale
        print("[INFO] Loading test image {}".format(my_image))
        test_image = cv2.imread(my_image)
        test_image = cv2.resize(test_image, (64, 128))
        test_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)

        # Step 2 - Normalize gamma and color
        gamma_value = 1.0
        test_image = gamma_correction(test_image, gamma_value)

        # Step 3 - Compute gradients
        test_gradient = compute_gradients(test_image)

        # Step 4 - Weighted vote into spatial and orientation cells
        compute_weighted_vote(test_image, test_gradient)


        # Step 5 - Contrast normalize over overlapping spatial blocks
        # Step 6 - Collect HOG's over detection window
        # Step 7 - Linear SVM

    cv2.destroyAllWindows()
