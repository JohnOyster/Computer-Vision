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
import numpy as np
import cv2
from matplotlib import pyplot as plt


def get_image(filename):
    """Load an image into a ndarray using OpenCV.

    :param filename:        File path and name to load
    :type:                  str
    :return:                Grayscale Image
    :rtype:                 numpy.ndarray
    """
    # Read in image
    image = cv2.imread(filename)

    # Convert to grayscale if needed
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    return image


def extract_sift_features(image, key_point_limit=400):

    # Get the SIFT Features using OpenCV
    sift = cv2.xfeatures2d.SIFT_create()

    # Extract key points and SIFT descriptors
    # cv2.KeyPoint
    # 'angle',
    # 'class_id',
    # 'convert',
    # 'octave',
    # 'overlap',
    # 'pt',
    # 'response',
    # 'size'
    key_points = sift.detect(image)

    # Sort the key points based on response (the higher the more 'cornerness' it has)
    key_points = sorted(key_points, key=lambda x: x.response, reverse=True)

    # Compute the SIFT descriptors up to <key_point_limit>
    if key_point_limit < len(key_points):
        key_points = key_points[:key_point_limit]
    key_points, descriptors = sift.compute(image, key_points)
    # key_points, descriptors = sift.detectAndCompute(image, None)

    return key_points, descriptors


def match_sift_descriptors(image1_desc, image2_desc, max_distance=3000.0):
    matches = []
    for query_index, desc1 in enumerate(image1_desc):
        for train_index, desc2 in enumerate(image2_desc):
            distance = np.square(desc1 - desc2).sum()
            if distance < max_distance:
                # Matches contain:
                # - Distance (distance)
                # - Image Index (imgIdx)
                # - Query Index (queryIdx)
                # - Train Index (trainIdx)
                dmatch = cv2.DMatch(_distance=distance,
                                    _queryIdx=query_index,
                                    _trainIdx=train_index,
                                    _imgIdx=0)
                matches.append(dmatch)
    return matches


def qualify_matches(matches, good_percentage=1.0):
    # Sort matches
    matches = sorted(matches, key=lambda x: x.distance, reverse=False)

    # Return only good matches
    good_match_count = int(len(matches) * good_percentage)
    return matches[:good_match_count]


def compute_transformation_matrix(image1_points, image2_points, matches):
    # Extract location of good matches
    points_1 = np.zeros((len(matches), 2), dtype=np.float32)
    points_2 = np.zeros((len(matches), 2), dtype=np.float32)

    # Matches already sorted so first 3 should be best
    for index, match in enumerate(matches):
        points_1[index, :] = image1_points[match.queryIdx].pt
        points_2[index, :] = image2_points[match.trainIdx].pt

    # Affine transform matrix expects only three points
    matrix = cv2.getAffineTransform(points_1[:3], points_2[:3])
    return matrix


def align_image(image, affine_matrix):
    # TODO(John): Need to verify size stuff here.
    size_x, size_y = image.shape
    return cv2.warpAffine(image, affine_matrix, (size_y, size_x))


def find_registration_error(image1, image2):
    # Convert to grayscale if needed
    if len(image1.shape) > 2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    if len(image2.shape) > 2:
        image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

    # Find error using sum of squared differences
    return np.square(image1 - image2).sum()


def main():
    # Define maximum number of features to calculate
    max_features = 400

    # Read in test images
    image1 = get_image("./Data/image1.png")
    image2 = get_image("./Data/image2.png")
    image2_color = cv2.imread("./Data/image2.png")

    # Initiate SIFT detector
    orb = cv2.ORB(max_features)

    # Compute the SIFT features for the test images
    image1_key_points, image1_descriptors = extract_sift_features(image1)
    image2_key_points, image2_descriptors = extract_sift_features(image2)

    # Using Naive Nearest Neighbors Approach
    nnn_matches = match_sift_descriptors(image1_descriptors, image2_descriptors)

    # create BFMatcher object
    #bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    # Match descriptors.
    bf_matches = bf.match(image1_descriptors, image2_descriptors)

    # Sort them in the order of their distance.
    nnn_matches = qualify_matches(nnn_matches)
    bf_matches = qualify_matches(bf_matches)

    # Draw first 20 matches.
    image_matched_nnn = cv2.drawMatches(image1, image1_key_points, image2, image2_key_points, nnn_matches[:20], None, flags=2)
    image_matched_bf = cv2.drawMatches(image1, image1_key_points, image2, image2_key_points, bf_matches[:20], None, flags=2)
    plt.imshow(image_matched_nnn)
    plt.show()
    plt.imshow(image_matched_bf)
    plt.show()

    # Compute the Affine transformation matrix
    bf_matix = compute_transformation_matrix(image1_key_points, image2_key_points, bf_matches)

    # Transform image2 so that it aligns with image1
    transformed_image = align_image(image2, bf_matix)
    image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB)
    transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_GRAY2RGB)

    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(image1)
    fig.add_subplot(1, 2, 2)
    plt.imshow(transformed_image)
    plt.show(block=True)

    # Compute the registration error
    reg_error = find_registration_error(image1, transformed_image)
    print(reg_error)

    return 0


if __name__ == '__main__':
    main()

    cv2.destroyAllWindows()