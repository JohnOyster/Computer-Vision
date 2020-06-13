#!/usr/bin/env python3
"""CIS 693 - Project 4.

Author: John Oyster
Date:   June 19, 2020
Description:

    The objective of this project is to implement the image registration technique.
    Given two images image1 and image2, the aim is to register image2 with respect
    to image1.  The registration problem has to be solved by matching SIFT features
    extracted from the images.

    Inputs: Images image1 and image2 are given.
    Output: image2 transformed with respect to image1 such that the difference
            between two images is minimum.

    DISCLAIMER: Comment text is taken from course handouts and is copyright
        2020, Dr. Almabrok Essa, Cleveland State University,
    Objectives:

    1. Extract SIFT features from images using a function from OpenCV library
       or any other library.
    2. Match features using naive nearest neighbor approach (function to be
       implemented) and the second version of nearest neighbor approach
       - cv2.BFMatcher().
    3. Compute the transformation matrix(affine transformation).
    4. Transform image2 such that it aligns with image1.
    5. Compute registration error for both feature matching methods

    Analysis question:
        Which feature matching algorithm works better and why?
"""
#  Copyright (c) 2020. John Oyster in agreement with Cleveland State University.
import cv2


def main():

    return 0


if __name__ == '__main__':
    main()

    cv2.destroyAllWindows()