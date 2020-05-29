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
from enum import Enum
import numpy as np
import os.path
from scipy.ndimage import gaussian_filter


# ----------------------------------------------------------------------------
# Define Test Images
def test_images():
    """Generator to return test images.

    :return:        Test Image Filename
    :rtype:         str
    """
    test_directory = './Data/'
    test_set = [
        'Image-1.jpg',
        'Image-2.bmp',
        'Image-3.jpg',
        'Image-4.jpg',
        'Image-5.jpg',
        'Image-7.bmp'
    ]
    for image in range(len(test_set)):
        yield os.path.join(test_directory, test_set[image])


# ----------------------------------------------------------------------------
# Three main independent processes:
#   1. Adaptive luminance enhancement
#       - Treatment of luminance information
#       - Dynamic range compression
#   2. Adaptive contrast enhancement
#       - Preservation of details
#       - Approximation of the tonality with the original image
#   3. Color restoration
#       - Convert the intensity images back to color images
# ----------------------------------------------------------------------------
def adaptive_luminance_enhancement(image, normalize=True):
    """Perform the non-linear luminance enhancement formula.

    :param image:       Input grayscale image
    :type:              numpy.ndarray
    :param normalize:   Decide if normalize image [0 1]
    :type:              bool
    :param output_RGB:  Decide to repeat luminance values three times
    :type:              bool
    :return:            Luminance Values of image
    :rtype:             numpy.ndarray
    """
    # Define image properties
    bits_per_pixel = np.iinfo(image.dtype).max

    # Define the z parameter for the normalized image enhancement equation.
    def _z_func(L):
        if L <= 50:
            z = 0
        elif (50 < L) and (L <= 150):
            z = (L - 50) / 100
        else:  # (L > 150)
            z = 1
        return z

    z_func = np.vectorize(_z_func, otypes=[np.float64])
    z_param = z_func(image)

    # Normalize the input image if needed
    image = image.astype(np.float64)
    new_image = (image / bits_per_pixel) if normalize else image

    # Apply image enhancement equation I'_n
    # TODO(John): Checked up to this point
    luminance_values = 0.5 * (np.power(new_image, 0.75 * z_param + 0.25) +
                              (1 - new_image) * 0.4 * (1 - z_param) +
                              np.power(new_image, 2 - z_param))
    return luminance_values


def adaptive_contrast_enhancement(image, image_norm, sigma):
    bits_per_pixel = np.iinfo(image.dtype).max
    # Get the Gaussian kernel convolved image
    # Iconv(x,y) = I(x, y) * G(x, y)
    kernel = kernel_gaussian(5, sigma)

    # Make the gray scale image have three channels
    convolved_image = gaussian_filter(image, sigma=sigma)
    image = image.astype(np.float64)
    #image = np.stack((image,)*3, axis=-1)
    #image = np.repeat(image[..., np.newaxis], 3, axis=2)
    #image_norm = np.stack((image_norm,)*3, axis=-1)
    #image_norm = np.repeat(image_norm[..., np.newaxis], 3, axis=2)
    #convolved_image = cv2.filter2D(image, -1, kernel)

    # Calculate P parameter
    def _p_func(sigma):
        if sigma <= 3:
            p = 3
        elif (sigma > 3) and (sigma < 10):
            p = (27 - 2 * sigma) / 7
        else:  # sigma >= 10
            p = 1
        return p

    p_func = np.vectorize(_p_func, otypes=[np.float64])
    # Find E(x, y) = r(x, y)^P = (Iconv(x, y)/I(x, y))^P
    E_param = np.power(convolved_image / image, p_func(sigma))

    # Find the center-surround contrast enhancement
    # S(x,y) = 255*Inorm(x, y)^E(x, y)
    S_param = bits_per_pixel * np.power(image_norm, E_param)
    S_param = S_param.astype(np.uint8)
    return S_param


def color_restoration(image, pixel_intensities, grayscale_image, hue_adjust=1):
    image = image.astype(np.float64)
    grayscale_image = grayscale_image.astype(np.float64)
    pixel_intensities = pixel_intensities.astype(np.float64)
    pixel_intensities = np.stack((pixel_intensities,) * 3, axis=-1)
    grayscale_image = np.stack((grayscale_image,)*3, axis=-1)
    #grayscale_image = np.repeat(grayscale_image[..., np.newaxis], 3, axis=2)
    color_enhanced_image = pixel_intensities * (image / grayscale_image) * hue_adjust
    color_enhanced_image = color_enhanced_image.astype(np.uint8)
    return color_enhanced_image


# ----------------------------------------------------------------------------
# Support Functions
class Color(Enum):
    """Enumeration for RGB Channels."""
    RED = 0
    GREEN = 1
    BLUE = 2


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


def kernel_gaussian(size=5, sigma=1, use_opencv=True):
    """Gaussian distribution kernel function.

    :param size:        Kernel mask size
    :type:              int
    :param sigma:       Standard deviation of image
    :type:              float
    :return:            Kernel mask
    :rtype:             numpy.ndarray
    """
    if use_opencv:
        side_x = cv2.getGaussianKernel(size, sigma)
        side_y = cv2.getGaussianKernel(size, sigma)
        kernel = side_x.dot(side_y.T)
    else:
        # G(x, y) = (1/2*pi*std^2) exp( -x^2 + y^2/ 2*std^2 )
        # G_i(x, y) = K exp(-(x^2+y^2)/c_i^2
        kernel_side = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
        x_values, y_values = np.meshgrid(kernel_side, kernel_side)
        K = 1 / (2 * np.pi * sigma ** 2)
        c_i = np.sqrt(2) * sigma
        kernel = K * np.exp(-0.5 * (np.square(x_values) + np.square(y_values)) / np.square(sigma))
        kernel = kernel / np.sum(kernel)
    return kernel


def image_convolution(image, kernel):
    (N, M) = image.shape[:2]
    (kernel_N, kernel_M) = kernel.shape[:2]

    # Add a pad to image border to support kernel mask
    pad = (kernel_N - 1) // 2
    pad_image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    convolved_image = np.zeros((N, M), dtype="float64")
    
    # Slide mask over image
    for y in np.arange(pad, N + pad):
        for x in np.arange(pad, M + pad):
            # Identify region of interest
            roi = pad_image[y - pad:y + pad + 1, x - pad:x + pad + 1]
            convolved_area = np.dot(roi, kernel).sum()
            convolved_image[y - pad, x - pad] = convolved_area
    return convolved_image


if __name__ == '__main__':
    for my_image in test_images():
        print("[INFO] Enhancing Image {}".format(my_image))
        original_image = cv2.imread(my_image)

        # Step 1 - Find Image Properties
        image_std = np.std(original_image)
        print("[INFO] The image global standard deviation is:  {}".format(image_std))

        # Step 2 - Convert to grayscale
        gray_image = _convert_rgb_to_grayscale(original_image)

        # import pdb;pdb.set_trace()
        # Step 3 - Adaptive Luminance Enhancement
        luminance_intensity_image = adaptive_luminance_enhancement(gray_image)

        # Step 4 - Adaptive Contrast Enhancement
        contrasted_image = adaptive_contrast_enhancement(gray_image, luminance_intensity_image, image_std)

        # Step 5 - Color Restoration
        enhanced_image = color_restoration(original_image, contrasted_image, gray_image)
        #contrasted_image = np.stack((contrasted_image,) * 3, axis=-1)
        #enhanced_image = contrasted_image

        # Step 6 - Display results
        resize_images = True
        if resize_images:
            original_image = cv2.resize(original_image, (500, 500))
            enhanced_image = cv2.resize(enhanced_image, (500, 500))
        output_stack = np.hstack((original_image, enhanced_image))
        cv2.imshow('Filter results', output_stack)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
