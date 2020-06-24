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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt




def point_cloud_pinhole(rgb_image, depth_image):
    # Get image properties
    width, height = rgb_image.shape[:2]
    depth_resolution = np.iinfo(depth_image.dtype).max

    # Define camera properties
    # Focal length:
    f_x = 521.0
    f_y = 521.0

    # Camera center:
    c_x = 325.0
    c_y = 250.0

    # Check images
    if rgb_image.shape[:2] != depth_image.shape[:2]:
        print("[DEBUG] Images need to have same shape")

    # Need to estimate the Image Transformation Matrix Ti
    # Where,
    #            Ti = [[R_i, t_i],
    #                  [0,   1]]
    #            R_i is from SO(3) rotation matrix
    #            t_ is from R^3 translation vector
    #
    #            p = pi(P) = (f_x (X/Z) + c_x,  f_y (y/Z) + c_y)^T
    cloud_list = []
    for row in range(width):
        for col in range(height):
            # TODO(John): Need to figure out the Z scaling factor
            depth_val = np.float16(depth_image[row, col]) / 1000
            if depth_val == 0:
                continue
            projected_x = depth_val * (row - c_x) / f_x
            projected_y = depth_val * (col - c_y) / f_y
            r_pixel, g_pixel, b_pixel = rgb_image[row, col]
            datum = (projected_x, projected_y, depth_val, r_pixel, g_pixel, b_pixel)
            cloud_list.append(datum)

    return np.array(cloud_list)


def plot_point_cloud(cloud_list):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(cloud_list[:, 0].min(), cloud_list[:, 0].max())
    ax.set_ylim(cloud_list[:, 1].min(), cloud_list[:, 1].max())
    ax.set_zlim(cloud_list[:, 2].min(), cloud_list[:, 2].max())
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    ax.set_zlabel('Depth')
    ax.scatter(cloud_list[:, 0], cloud_list[:, 1], cloud_list[:, 2])
    plt.show()


def main():
    """Execute this routine if this file is called directly.

    This function is used to test the parameters of the SIFT method
    and make sure that it works.

    :return:        Errno = 0 if good
    :rtype:         int
    """
    # Get Images
    image1_rgb = cv2.imread("./Data/rgb/image1.png")
    image1_rgb = cv2.cvtColor(image1_rgb, cv2.COLOR_BGR2RGB)
    image1_depth = cv2.imread("./Data/depth/image1.png")
    image1_depth = cv2.cvtColor(image1_depth, cv2.COLOR_BGR2GRAY)
    image2_rgb = cv2.imread("./Data/rgb/image2.png")
    image2_rgb = cv2.cvtColor(image2_rgb, cv2.COLOR_BGR2RGB)
    image2_depth = cv2.imread("./Data/depth/image2.png")
    image2_depth = cv2.cvtColor(image2_depth, cv2.COLOR_BGR2GRAY)

    # -----------------------------------------------------------------
    # Part 1 - Pinhole Camera Model
    pin_cloud = point_cloud_pinhole(image1_rgb, image1_depth)
    print(pin_cloud)
    plot_point_cloud(pin_cloud)



    return 0


if __name__ == '__main__':
    main()

    cv2.destroyAllWindows()
