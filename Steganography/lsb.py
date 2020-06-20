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
    size_x, size_y = image.shape[:2]
    max_length = size_x * size_y / 4
    return max_length


def encode_message(image, message):
    size_x, size_y = image.shape[:2]
    if compute_max_message_length(image) < len(message):
        return -1
    bin_message = [format(ord(ch), '#010b') for ch in list(message)]
    steg_message = []
    for ch in bin_message:
        steg_message.append(ch[2:4])
        steg_message.append(ch[4:6])
        steg_message.append(ch[6:8])
        steg_message.append(ch[8:10])
    #steg_message = np.array(steg_message)
    index = 0
    steg_length = len(steg_message)
    steg_image = np.empty((size_x, size_y))
    for row in range(size_x):
        for col in range(size_y):
            #if image[row, col] > 250:
            #    continue
            if index >= steg_length or steg_message[index] == '':
                break
            image[row, col] = image[row, col] + int(steg_message[index], 2)
            index += 1

    return image


def decode_message(image):
    size_x, size_y = image.shape[:2]

    message = []
    recovered_message = []
    for row in range(size_x):
        for col in range(size_y):
            #if image[row, col] > 254:
            #    continue
            data = image[row, col] & 0b11
            message.append(format(data, '#04b')[-2:])
    for ch in range(int(len(message) / 4)):
        recovered_message.append(message[ch*4:ch*4+4])
    ascii_message = []
    for ch in recovered_message:
        character = chr(int(''.join(ch), 2))
        #import pdb; pdb.set_trace()
        ascii_message.append(character)
    return ascii_message


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
    clean_image = get_image("./Data/Cleveland.jpg", grayscale=True)

    steg_image = encode_message(clean_image, "Hello, there fine world!")
    # Display results
    display_side_by_side(clean_image, clean_image)

    recovered_msg = decode_message(steg_image)

    print(recovered_msg)


if __name__ == '__main__':
    main()

    cv2.destroyAllWindows()
