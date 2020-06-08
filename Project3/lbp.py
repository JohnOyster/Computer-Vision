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



if __name__ == '__main__':
    dataset = process_mat_file()
    print(dataset[1][9])
    print(type(dataset))
    cv2.destroyAllWindows()
