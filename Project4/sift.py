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
    """Extract features from an image using OpenCV.

    :param image:               Input grayscale image
    :type:                      numpy.ndarray
    :param key_point_limit:     Limit number of features
    :type:                      int
    :return:                    List of key points and descriptors
    :type:                      tuple
    """
    # Get the SIFT Features using OpenCV
    # pylint: disable=no-member
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
    """Match image SIFT descriptors using Euclidean distance.

    :param image1_desc:         SIFT descriptors from image 1
    :type:                      list
    :param image2_desc:         SIFT descriptors from image 2
    :type:                      list
    :param max_distance:        Maximum distance to consider between matches
    :type:                      float
    :return:                    List of OpenCV DMatch objects
    :rtype:                     list
    """
    matches = []
    # Need to maintain descriptor index
    # Image one will be the query source and image 2 will be the
    #   train destination
    for query_index, desc1 in enumerate(image1_desc):
        for train_index, desc2 in enumerate(image2_desc):
            # Use SSD for distance
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
    """Sort matches so more prominent ones are first.

    :param matches:         List of DMatch objects
    :type:                  list
    :param good_percentage: Percent of matches to accept
    :type:                  float
    :return:                List of DMatch objects
    :rtpe:                  list
    """
    # Sort matches
    matches = sorted(matches, key=lambda x: x.distance, reverse=False)

    # Return only good matches
    if good_percentage > 1.0 or good_percentage < 0:
        good_percentage = 1.0
    good_match_count = int(len(matches) * good_percentage)
    return matches[:good_match_count]


def compute_transformation_matrix(image1_points, image2_points, matches):
    """Find the Affine transformation 3x3 matrix.

    :param image1_points:       SIFT key points from image 1
    :type:                      list
    :param image2_points:       SIFT key points from image 2
    :type:                      list
    :param matches:             List of DMatch objects
    :type:                      list
    :return:                    3x3 Affine matrix
    :rtype:                     numpy.ndarray
    """
    # Extract location of good matches
    # OpenCV needs float32 for this and not float64
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
    """Transform image using Affine transformation matrix.

    :param image:       Image to manipulate
    :type:              numpy.ndarray
    :param affine_matrix: 3x3 transformation matrix
    :type:              numpy.ndarray
    :returns:           Transformed image
    :rtype:             numpy.ndarray
    """
    size_x, size_y = image.shape
    return cv2.warpAffine(image, affine_matrix, (size_y, size_x))


def display_side_by_side(image1, image2):
    """Show two images side by side.

    :param image1:      Image 1
    :type:              numpy.ndarray
    :param image2:      Image 2
    :type:              numpy.ndarray
    """
    # Normalize color to match
    image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB)
    image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2RGB)

    # Create subplots
    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(image1)
    fig.add_subplot(1, 2, 2)
    plt.imshow(image2)
    plt.show(block=True)


def find_registration_error(image1, image2):
    """Calculate the image registration error.

    :param image1:
    :param image2:
    :return:
    """
    # Convert to grayscale if needed
    if len(image1.shape) > 2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    if len(image2.shape) > 2:
        image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)

    size_x = max(image1.shape[0], image2.shape[0])
    size_y = max(image1.shape[1], image2.shape[1])

    # Find error using sum of squared differences
    return np.square(image1 - image2).sum() / (size_x * size_y)


def main():
    """Execute this routine if this file is called directly.

    This function is used to test the parameters of the SIFT method
    and make sure that it works.

    :return:        Errno = 0 if good
    :rtype:         int
    """
    # Define maximum number of features to calculate
    max_features = 400

    # Read in test images
    image1 = get_image("./Data/image1.png")
    image2 = get_image("./Data/image2.png")

    # Initiate SIFT detector
    # pylint: disable=unused-variable
    orb = cv2.ORB(max_features)

    # Compute the SIFT features for the test images
    image1_key_points, image1_descriptors = extract_sift_features(image1)
    image2_key_points, image2_descriptors = extract_sift_features(image2)

    # Using Naive Nearest Neighbors Approach
    nnn_matches = match_sift_descriptors(image1_descriptors, image2_descriptors)

    # create BFMatcher object
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    # Match descriptors.
    bf_matches = bf.match(image1_descriptors, image2_descriptors)

    # Sort them in the order of their distance.
    nnn_matches = qualify_matches(nnn_matches)
    bf_matches = qualify_matches(bf_matches)

    # Draw first 20 matches.
    image_matched_nnn = cv2.drawMatches(image1, image1_key_points,
                                        image2, image2_key_points,
                                        nnn_matches[:20], None, flags=2)
    image_matched_bf = cv2.drawMatches(image1, image1_key_points,
                                       image2, image2_key_points,
                                       bf_matches[:20], None, flags=2)
    plt.imshow(image_matched_nnn)
    plt.show()
    plt.imshow(image_matched_bf)
    plt.show()

    # Compute the Affine transformation matrix
    bf_matix = compute_transformation_matrix(image1_key_points, image2_key_points, bf_matches)
    nnn_matrix = compute_transformation_matrix(image1_key_points, image2_key_points, nnn_matches)

    # Transform image2 so that it aligns with image1
    bf_transformed_image = align_image(image2, bf_matix)
    nnn_transformed_image = align_image(image2, nnn_matrix)

    # Show transformed image
    display_side_by_side(image1, bf_transformed_image)
    display_side_by_side(image1, nnn_transformed_image)

    # Compute the registration error
    # Find the registration error between image 1 and image 2
    orig_reg_error = find_registration_error(image1, image2)
    bf_reg_error = find_registration_error(image1, bf_transformed_image)
    nnn_reg_error = find_registration_error(image1, nnn_transformed_image)
    print("[INFO] Original registration error between image 1 and image 2:  {}".format(orig_reg_error))
    print("[INFO] BFMatch registration error between image 1 and image 2:  {}".format(bf_reg_error))
    print("[INFO] NNN registration error between image 1 and image 2:  {}".format(nnn_reg_error))

    objects = ('Original', 'BFMatch', 'NNN')
    y_pos = np.arange(len(objects))
    performance = [orig_reg_error, bf_reg_error, nnn_reg_error]

    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Error')
    plt.title('Registration Error between images')

    plt.show()

    return 0


if __name__ == '__main__':
    main()

    cv2.destroyAllWindows()
