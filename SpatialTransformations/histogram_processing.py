#!/usr/bin/env python3
"""CIS 693 - Histogram Processing Homework.

Author: John Oyster
Date:   May 22, 2020
Description:
    List of Functions:
    1. Histogram Processing

"""
import cv2
import numpy as np
from matplotlib import pyplot as plt


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


def process_histogram(image, show_plot=True):
    """Generate a normalized histogram of a given grayscale image.

    p(r_k) = n_k/M*N  for given NxM image

    :param image:       Original Image
    :type:              nmpy.ndarray
    :param show_plot:   True will show Matplotlib plot
    :type:              bool
    :return:            hist, bins, pdf
    """
    dim_M, dim_N = image.shape
    image_max_pixel_size = np.iinfo(image.dtype).max
    image_bins, counts = np.unique(image, return_counts=True)
    hist = np.zeros(image_max_pixel_size, dtype=np.uint8)
    pdf = np.zeros(image_max_pixel_size, dtype=np.double)
    bins = list(range(image_max_pixel_size))
    for image_bin in bins:
        if image_bin in image_bins:
            hist[image_bin] = counts[np.where(image_bin == image_bins)]
            pdf[image_bin] = hist[image_bin] / (dim_M*dim_N)
    if show_plot:
        _show_histogram(hist, bins)
    return hist, bins, pdf


def _show_histogram(hist, bins):
    """Show Matplotlib histogram of image.

    :param hist:        Histogram count values
    :type:              numpy.ndarray
    :param bins:        Bins
    :type:              numpy.ndarray
    :return:            None
    """
    plt.hist(hist, bins)
    plt.title("Intensity Histogram")
    plt.show()


if __name__ == '__main__':
    image_file = cv2.imread("./Cleveland.jpg")
    gray_image = convert_to_grayscale(image_file)
    cv2.imshow('Grayscale Image', gray_image)
    cv2.waitKey(0)
    process_histogram(gray_image)

    cv2.destroyAllWindows()
