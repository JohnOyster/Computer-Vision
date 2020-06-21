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


def get_image(filename):
    """Load an image into a ndarray using OpenCV.

    :param filename:        File path and name to load
    :type:                  str
    :return:                RGB Image
    :rtype:                 numpy.ndarray
    """
    # Read in image
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def compute_max_message_length(image):

    bits_per_pixel = np.iinfo(image.dtype).bits
    max_length = (image.size / bits_per_pixel) * 3
    return max_length


def convert_message_to_bits(message):
    bit_message = [format(ord(ch), '#010b')[2:] for ch in list(message)]
    return "".join(bit_message)


def hide_bit(pixel, message_bit):
    # Take only the first 7 bits and add 1 iff message_bit == 1
    lsb_mask = pixel & ~1
    return np.uint8(lsb_mask | int(message_bit))


def encode_message(image, message):
    # Get image properties
    width, height = image.shape[:2]

    # Add a length value to the message to get it working :)
    steg_message = "{}|{}".format(len(message), message)

    # Convert the ASCII string into a list of binary encoded strings
    # for easier manipulation
    binary_message = convert_message_to_bits(steg_message)

    # Add padding to make sure message is multiple of three
    steg_padding = '0' * (3 - (len(binary_message) % 3))
    binary_message += steg_padding
    binary_message_length = len(binary_message)
    if binary_message_length >= compute_max_message_length(image):
        print("[DEBUG]  Message too large!")
    steg_image = np.copy(image)
    index = 0
    for row in range(height):
        for col in range(width):
            # Last sanity check
            if index + 3 <= binary_message_length:

                # Get the current pixel
                r_pixel, g_pixel, b_pixel = image[col, row]

                # Hide a message bit in each RGB channel
                r_pixel = hide_bit(r_pixel, binary_message[index])
                g_pixel = hide_bit(g_pixel, binary_message[index + 1])
                b_pixel = hide_bit(b_pixel, binary_message[index + 2])

                # Save the encoded pixel
                steg_image[col, row] = (r_pixel, g_pixel, b_pixel)

                # Increment message index
                index += 3
            else:
                return steg_image


def decode_message(image):
    # Get image properties
    width, height = image.shape[:2]
    bits_per_pixel = np.iinfo(image.dtype).bits

    recovered_message = []
    expected_message_length = None
    recovered_char = 0
    count = 0
    length_of_message_length = 0
    for row in range(height):
        for col in range(width):

            pixel = image[col, row]

            for color in pixel:
                recovered_char += (color & 1) << (bits_per_pixel - 1 - count)
                count += 1
                if count == bits_per_pixel:
                    recovered_message.append(chr(recovered_char))
                    recovered_char = 0
                    count = 0
                    if recovered_message[-1] == "|" and expected_message_length is None:
                        expected_message_length = int("".join(recovered_message[:-1]))
                        length_of_message_length = len(str(expected_message_length)) + 1

            if len(recovered_message) - length_of_message_length == expected_message_length:
                return "".join(recovered_message)[length_of_message_length:]


def display_side_by_side(image1, image2):
    """Show two images side by side.

    :param image1:      Image 1
    :type:              numpy.ndarray
    :param image2:      Image 2
    :type:              numpy.ndarray
    """

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

    steg_image = encode_message(clean_image, "Hello, there fine world!")
    # Display results
    display_side_by_side(clean_image, steg_image)

    recovered_msg = decode_message(steg_image)

    print(recovered_msg)


if __name__ == '__main__':
    main()

    cv2.destroyAllWindows()
