#!/usr/bin/env python3
"""CIS 693 - Project 3.

Author: John Oyster
Date:   June 12, 2020
Description:
    DISCLAIMER: Comment text is taken from course handouts and is copyright
        2020, Dr. Almabrok Essa, Cleveland State University,
    Objectives:
        The objective of this project is to generate a face recognition system.
        The dataset used is AT&T data (ORL). It contains 40 distinct subjects
        that have a total of 400 face images corresponding to 10  different
        images per subject. The images are taken at different times with
        different specifications, including slightly varying illumination,
        different facial expressions such as open and closed eyes, smiling and
        non-smiling, and facial details like wearing glasses.
        1. Write a program to implement the Local Binary Pattern (LBP) Algorithm
           for face recognition.
           - Divide the image into small blocks 16×16.
           - Extract the information (histogram) of each block separately using LBP.
           - Concatenate these local histograms to form a final feature vector.
        2. Write a program to train and test the linear Support Vector Machine (SVM)
           classifier for face recognition using the extracted features from part 1.
           a) Train the SVM classifier with LBP features of 70% of the dataset
              randomly selected.
           b) Classify the LBP features of the rest 30% of the dataset using the
              trained SVM model.
           c) Repeat the process of (a) and (b) 10 times and compute the average
              recognition accuracy.
        3. Repeat the experiment in Part 2 for training the SVM classifier with
           different set of kernel functions (e. g. rbf, polynomial, etc.).
        4. Repeat the experiment in Part 2 using any other different classifier.
    Assumptions:
        1. Unless this statement is remove, 8-bit pixel values

"""
#  Copyright (c) 2020. John Oyster in agreement with Cleveland State University.
import warnings
import cv2
import numpy as np
from scipy.io import loadmat


def process_mat_file(mat_file="./Data/ORL_64x64.mat"):
    """Process mat format into Python dictionary.

    :param mat_file:        Pre-7.3 MATLAB mat file path
    :type:                  str
    :return:                Dictionary of faces
    :rtype:                 dict
    """
    # Since matrix is pre-MATLAB 7.3 format, use 'loadmat'
    # ASSUMPTION: Expecting:
    #   'gnd' key -->  person identifier
    #   'fea' key -->  person 'features' or image
    mat = loadmat(mat_file)

    # Enumerate over image data
    image_dictionary = {}
    for count, person_id in enumerate(mat['gnd']):
        person_id = int(person_id)

        # Acquire the current image array from the 'features' array
        # Numpy has a quirk that reshaping (64, 64) will rotate the data 90 degrees
        # so make sure to transpose the data.
        image = np.transpose(mat['fea'][count].reshape((64, 64)))

        # Create new dictionary to store data
        # Format:  dict( person_id : list( image_0, image_1, etc..))
        if person_id in image_dictionary:
            image_dictionary[person_id].append(image)
        else:
            image_dictionary[person_id] = [image]

    return image_dictionary


def pad_image(image, pad_width=1, pad_type=cv2.BORDER_CONSTANT):
    """Add pad values around an image.

    :param image:           Input image to pad
    :tyoe:                  numpy.ndarray
    :param pad_width:       Width of pad
    :type:                  numpy.ndarray
    :param pad_type:        Set type of pad (cv2 Enum)
    :type:                  int
    :return:                Padded image
    :rtype:                 numpy.ndarray
    """
    #print("[DEBUG] Input image to pad function is {}".format(image.shape))
    pad_value = [0, 0, 0]
    padded_image = cv2.copyMakeBorder(image, pad_width, pad_width, pad_width,
                                      pad_width, pad_type, value=pad_value)
    return padded_image


def threshold_pixel(image, center, x, y):
    """Calculate if Pixel(x, y) intensity is greater than center intensity of neighborhood.

    :param image:       Input image to pull from
    :type:              numpy.ndarray
    :param center:      Value of center pixel in 3x3 neighborhood
    :type:              int
    :param x:           x coordinate to calculate in 'image'
    :type:              int
    :param y:           y coordinate to calculate in 'image'
    :type:              int
    :return:            Thresholded pixel value w.r.t center
    ;rtype:             int
    """
    value = 1 if float(image[x][y]) - center >= 0 else 0
    return value


def calculate_lbp_pixel(image, x, y):
    """
     32 |  64 | 128
    ----+-----+-----
     16 |   0 |   1
    ----+-----+-----
     8  |   4 |   2
    """
    center = image[x][y]
    binary_code = np.empty(8)
    binary_code[0] = threshold_pixel(image, center, x, y + 1)       # Right
    binary_code[1] = threshold_pixel(image, center, x + 1, y + 1)   # Bottom Right
    binary_code[2] = threshold_pixel(image, center, x + 1, y)       # Bottom
    binary_code[3] = threshold_pixel(image, center, x + 1, y - 1)   # Bottom Left
    binary_code[4] = threshold_pixel(image, center, x, y - 1)       # Left
    binary_code[5] = threshold_pixel(image, center, x - 1, y - 1)   # Top Left
    binary_code[6] = threshold_pixel(image, center, x - 1, y)       # Top
    binary_code[7] = threshold_pixel(image, center, x - 1, y + 1)   # Top Right

    weights = np.array([1, 2, 4, 8, 16, 32, 64, 128])
    lbp_value = np.dot(binary_code, weights).astype(np.uint8)

    return lbp_value


def calculate_lbp_region(image):
    width, height = image.shape

    #print("[DEBUG] Input image to region is {}".format(image.shape))
    # Step 1: Pad the input images to allow LBP thresholding operator space
    #         to work on the edge pixel cases.
    padded_image = pad_image(image)

    # Step 2:
    region_descriptor = np.zeros((width, height), np.uint8)
    for row in range(0, height):
        for col in range(0, width):
            region_descriptor[row, col] = calculate_lbp_pixel(padded_image, row+1, col+1)
    hist_lbp = cv2.calcHist([region_descriptor], [0], None, [256], [0, 256])
    return hist_lbp.ravel()


def calculate_lbp(image, region_size=(16, 16)):
    # Get image data
    width, height = image.shape
    rx, ry = region_size

    # Segment input image into sub sections based on 'region_size'
    # Test 'region_size' just in case
    if (height % rx != 0) or (width % ry != 0):
        warnings.warn("In calculate_lbp: input image dimensions are not divisible by region_size")
    num_regions_x = width // rx
    num_regions_y = height // ry
    #print("[DEBUG] Image contans x={} y={} regions".format(num_regions_x, num_regions_y))
    # TODO(John): Need to figure out the initialization size
    lbp_descriptor = np.empty((num_regions_x, num_regions_y, 256))
    for row in range(num_regions_y):
        for col in range(num_regions_x):
            lbp_descriptor[row, col] = calculate_lbp_region(image[row*ry:row*ry+ry, col*rx:col*rx+rx])

    return lbp_descriptor.ravel()


if __name__ == '__main__':
    # Load the data set
    # Expecting 40 people with 10 images each
    data_set = process_mat_file()

    for person, images in data_set.items():
        print("[INFO] Processing Person {}".format(person))

        # Cycles through each sample image for each person
        for image_sample in images:
            calculate_lbp(image_sample)
            # Calculate the LBP histogram descriptor for each image

    cv2.destroyAllWindows()
