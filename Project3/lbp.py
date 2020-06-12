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
    Assumptions:
        1. Unless this statement is remove, 8-bit pixel values
"""
#  Copyright (c) 2020. John Oyster in agreement with Cleveland State University.
import warnings
import cv2
import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt


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
    value = 0
    try:
        value = 1 if float(image[x][y]) - center >= 0 else 0
    except IndexError:
        # Padding is messed up for non-project cases
        pass
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
    """

    :param image:
    :return:
    """
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
    """

    :param image:
    :param region_size:
    :return:
    """
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


def display_lbp(plot_data):
    figure = plt.figure()
    for item in range(len(plot_data)):
        current_dict = plot_data[item]
        current_img = current_dict["img"]
        current_xlabel = current_dict["xlabel"]
        current_ylabel = current_dict["ylabel"]
        current_xtick = current_dict["xtick"]
        current_ytick = current_dict["ytick"]
        current_title = current_dict["title"]
        current_type = current_dict["type"]
        current_plot = figure.add_subplot(1, len(plot_data), item + 1)
        if current_type == "gray":
            current_plot.imshow(current_img, cmap=plt.get_cmap('gray'))
            current_plot.set_title(current_title)
            current_plot.set_xticks(current_xtick)
            current_plot.set_yticks(current_ytick)
            current_plot.set_xlabel(current_xlabel)
            current_plot.set_ylabel(current_ylabel)
        elif current_type == "histogram":
            current_plot.plot(current_img, color="black")
            current_plot.set_xlim((0, 260))
            current_plot.set_title(current_title)
            current_plot.set_xlabel(current_xlabel)
            current_plot.set_ylabel(current_ylabel)
            ytick_list = [int(i) for i in current_plot.get_yticks()]
            current_plot.set_yticklabels(ytick_list, rotation=0)

    plt.show()


def main():
    """Execute this routine if this file is called directly.

    This function is used to test the parameters of the LBP method
    and make sure that it works.

    :return:        Errno = 0 if good
    :rtype:         int
    """
    # Load the data set
    # Expecting 40 people with 10 images each
    data_set = process_mat_file()

    num_of_samples = 1
    for person, images in data_set.items():
        print("[INFO] Processing Person {}".format(person))

        # Cycles through each sample image for each person
        for image_sample in images:
            # Get image properties
            height, width = image_sample.shape

            # Calculate the LBP descriptor
            image_lbp = calculate_lbp(image_sample)

            if num_of_samples > 0:
                # Calculate the LBP histogram descriptor for each image
                image_lbp = np.zeros((height, width, 3), np.uint8)
                for row in range(height):
                    for col in range(width):
                        image_lbp[row, col] = calculate_lbp_pixel(image_sample, row, col)
                hist_lbp = cv2.calcHist([image_lbp], [0], None, [256], [0, 256])
                output_list = []
                output_list.append({
                    "img": image_sample,
                    "xlabel": "",
                    "ylabel": "",
                    "xtick": [],
                    "ytick": [],
                    "title": "Sample Image",
                    "type": "gray"
                })
                output_list.append({
                    "img": image_lbp,
                    "xlabel": "",
                    "ylabel": "",
                    "xtick": [],
                    "ytick": [],
                    "title": "LBP Image",
                    "type": "gray"
                })
                output_list.append({
                    "img": hist_lbp,
                    "xlabel": "Bins",
                    "ylabel": "Number of pixels",
                    "xtick": None,
                    "ytick": None,
                    "title": "Histogram (LBP)",
                    "type": "histogram"
                })

                display_lbp(output_list)

                num_of_samples -= 1

    return 0


if __name__ == '__main__':
    # Enter main program
    main()

    # Clean-up
    cv2.destroyAllWindows()
