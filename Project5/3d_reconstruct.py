#!/usr/bin/env python3
"""CIS 693 - Project 5.

Author: John Oyster
Date:   June 26, 2020
Description:

    The objective of this project is to perform 3D indoor scene reconstruction.

    1. Write a program to create a 3D point cloud model based on the pinhole
       camera model. Use one RGB-D image (image1 rgb with image1 depth) that
       is captured using a Kinect camera, see included Data5.

       The essential parameters of RGB-D Microsoft Kinect Camera are as following:
       - Focal length:
            f_x = 521
            f_y = 521
       - Camera center:
            c_x = 325
            c_y = 250

    2. Repeat part 1 using two RGB-D images (image1 rgb with image1 depth and
       image2 rgb with image2 depth) to estimate the camera position assuming
       the camera position of image1 is at the origin (0,0,0).
       - To estimate the camera transformation matrix (T), apply Iterative
         Closed Point (ICP) method using point-to-plane error metric (or you
         may use any built-in function).
       - Once the T is estimated, transform point cloud of image 2 to image 1
         camera coordinates (you use any built-in function).
       - At the end, merge the two 3D point clouds.
"""
#  Copyright (c) 2020. John Oyster in agreement with Cleveland State University.
import numpy as np
import cv2
from matplotlib import pyplot as plt


def main():
    """Execute this routine if this file is called directly.

    This function is used to test the parameters of the SIFT method
    and make sure that it works.

    :return:        Errno = 0 if good
    :rtype:         int
    """

    return 0


if __name__ == '__main__':
    main()

    cv2.destroyAllWindows()
