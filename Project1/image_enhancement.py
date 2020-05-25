#!/usr/bin/env python3
"""CIS 693 - Project 1.

Author: John Oyster
Date:   May 29, 2020
Description:
    DISCLAIMER: Comment text is taken from course handouts and is copyright
        2020, Dr. Almabrok Essa, Cleveland State University,
    Objectives:
        1. Write  a Python program  to  implement the  nonlinear  approach
          for  adaptive  and  integrated neighborhood  image enhancement
          algorithm  for  enhancement  of  images  captured  in  low  and
          non-uniform lighting environments.
        2.Test  and  evaluate  the  algorithm  on  sample  color  images
          of  different  types  (low  lighting, uniform darkness, non-uniform
          lighting and extremely dark images). SeeData1 enclosed.
        3.Show  a  quantitative  evaluation  (graphical  representation  of
          statistical  characteristics  of images  before  and  after
          enhancement)  of  the  performance  of  the  algorithm  on
          several  test images.
    Assumptions:
        1. Unless this statement is remove, 8-bit pixel values

"""
import cv2
import numpy as np
import os.path


def test_images():
    """Generator to return test images.

    :return:        Test Image Filename
    :rtype:         str
    """
    test_directory = './Data/'
    test_set = [
        'Image-1.jpg'
        # 'Image-2.bmp',
        # 'Image-3.jpg',
        # 'Image-4.jpg',
        # 'Image-5.jpg',
        # 'Image-7.bmp'
    ]
    yield os.path.join(test_directory, test_set.pop())


# ----------------------------------------------------------------------------
# Three main independent processes:
#
#   1. Adaptive luminance enhancement
#       - Treatment of luminance information
#       - Dynamic range compression
#   2. Adaptive contrast enhancement
#       - Preservation of details
#       - Approximation of the tonality with the original image
#   3. Color restoration
#       - Convert the intensity images back to color images
# ----------------------------------------------------------------------------


def _convert_rgb_to_grayscale(image, normalize=False):
    """Convert input color image to the intensity (gray-scale) image.

    Method in standard NTSC (National Television Standards Committee)
    Approach:
        - Find the parameter z associated with the nonlinear function.
        - Apply the nonlinear transfer function elevate the intensity
          values of low-intensity.
        - Compute the enhanced intensity.

    :param image:       Image array
    :type:              numpy.ndarray
    :param normalize:   Decide if normalize image [0 1]
    :type:              bool
    :return:            Grayscale Image
    :rtype:             numpy.ndarray
    """

    bits_per_pixel = np.iinfo(image.dtype).max
    rgb_weights = np.array([76.245, 19.685, 29.071]) / bits_per_pixel
    new_image = np.dot(image[..., :3], rgb_weights)
    new_image = new_image.astype(np.uint8)

    if normalize:
        new_image = new_image / bits_per_pixel

    return new_image


def adaptive_luminance_enhancement(image, normalize=True):
    """

    :param image:       Input grayscale image
    :type:              numpy.ndarray
    :param normalize:   Decide if normalize image [0 1]
    :type:              bool
    :return:
    """
    # Define image properties
    print(image)
    bits_per_pixel = np.iinfo(image.dtype).max

    # Define the z parameter for the normalized image enhancement equation.
    def _z_func(L):
        if L <= 50:
            z = 0
        elif (50 < L) & (L <= 105):
            z = (L - 50)/100
        else:    # (L > 105)
            z = 1
        return z
    z_func = np.vectorize(_z_func, otypes=[np.float64])
    z_param = z_func(image)

    # Normalize the input image if needed
    image = image.astype(np.float64)
    new_image = (image / bits_per_pixel) if normalize else image

    # Apply image enhancement equation I'_n
    enhance_image = 0.5 * (np.power(new_image, 0.75 * z_param + 0.25) +
                          (1 - new_image) * 0.4 * (1 - z_param) +
                          np.power(new_image, 2 - z_param))

    print(enhance_image)
    return


def adaptive_contrast_enhancement():
    pass


def color_restoration():
    pass


if __name__ == '__main__':
    for my_image in test_images():
        print("Enhancing Image {}".format(my_image))
        original_image = cv2.imread(my_image)

        # Step 1 - Convert to grayscale
        gray_image = _convert_rgb_to_grayscale(original_image)

        # Step 2 - Adaptive Luminance Enhancement
        test_arr = np.array([[5, 25, 65],
                             [100, 150, 255]])
        #luminance_intensity_image = adaptive_luminance_enhancement(gray_image)
        luminance_intensity_image = adaptive_luminance_enhancement(test_arr)

        # Step n - Display results
        # import pdb;pdb.set_trace()
        gray_image_3 = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)  # Trick to stack images
        output_stack = np.hstack((original_image, gray_image_3))
        cv2.imshow('Filter results', output_stack)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
