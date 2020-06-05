#!/usr/bin/env python3
"""CIS 693 - Project 2.

Author: John Oyster
Date:   June 6, 2020
Description:
    DISCLAIMER: Comment text is taken from course handouts and is copyright
        2020, Dr. Almabrok Essa, Cleveland State University,
    Objectives:
        2. Write  a  program  to  train  and  test  the  linear  Support  Vector
        Machine  (SVM)  classifier  for pedestrian detection using the extracted
         features from part 1.
            a) Train  the  SVM  classifier  with HOGfeatures of  the  training
                set  (use  built-in function/library (e.g. from sklearn.svm import SVC)).
            b) Classify the HOGfeatures
                of the testing images (both positive and negatives samples)using
                the trained SVM model (use built-in function/library).
            c) Compute the accuracy, false positive rate, and the miss rate.
        3. Repeat  the  experiment  in  part  2  for  training  the  SVM  classifier
            with  different  set  of  kernel functions (e. g. rbf, polynomial, etc.).
    Assumptions:
        1. Unless this statement is remove, 8-bit pixel values

"""
#  Copyright (c) 2020. John Oyster in agreement with Cleveland State University.
import os.path
from os import listdir
from os.path import isfile, join

import numpy as np
import cv2
from sklearn.svm import SVC
from Project2 import hog


def get_good_train_set(directory="./NICTA/TrainSet/PositiveSamples"):
    test_files = [join(directory, image) for image in listdir(directory) if isfile(join(directory, image))]
    return test_files


def get_bad_train_set(directory="./NICTA/TrainSet/NegativeSamples"):
    test_files = [join(directory, image) for image in listdir(directory) if isfile(join(directory, image))]
    return test_files


def get_good_test_set(directory="./NICTA/TestSet/PositiveSamples"):
    test_files = [join(directory, image) for image in listdir(directory) if isfile(join(directory, image))]
    return test_files

def get_bad_test_set(directory="./NICTA/TestSet/NegativeSamples"):
    test_files = [join(directory, image) for image in listdir(directory) if isfile(join(directory, image))]
    return test_files


if __name__ == '__main__':
    gamma_value = 1.0
    good_set = get_good_train_set()
    image_count = len(good_set)
    good_set_hog = np.empty((image_count, 3780))
    image_index = 0
    for image in good_set:
        test_image = cv2.imread(image)
        test_image = cv2.resize(test_image, (64, 128))
        test_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)
        test_image = hog.gamma_correction(test_image, gamma_value)
        test_gradient = hog.compute_gradients(test_image)
        cell_histograms, _ = hog.compute_weighted_vote(test_gradient)
        hog_blocks, _ = hog.normalize_blocks(cell_histograms)
        good_set_hog[image_index] = hog_blocks.ravel()
        image_index += 1
    good_set_tag = np.ones(image_count)

    bad_set = get_bad_train_set()
    image_count = len(bad_set)
    bad_set_hog = np.empty((image_count, 3780))
    image_index = 0
    for image in bad_set:
        test_image = cv2.imread(image)
        test_image = cv2.resize(test_image, (64, 128))
        test_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)
        test_image = hog.gamma_correction(test_image, gamma_value)
        test_gradient = hog.compute_gradients(test_image)
        cell_histograms, _ = hog.compute_weighted_vote(test_gradient)
        hog_blocks, _ = hog.normalize_blocks(cell_histograms)
        bad_set_hog[image_index] = hog_blocks.ravel()
        image_index += 1
    bad_set_tag = np.zeros(image_count)

    train_data = np.concatenate((good_set_hog, bad_set_hog))
    tag_data = np.concatenate((good_set_tag, bad_set_tag))
    clf = SVC(kernel='linear', C=1.0)
    clf.fit(train_data, tag_data)

    good_test_set = get_good_test_set()
    image_count = len(good_test_set)
    good_test_set_hog = np.empty((image_count, 3780))
    image_index = 0
    for image in good_test_set:
        test_image = cv2.imread(image)
        test_image = cv2.resize(test_image, (64, 128))
        test_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)
        test_image = hog.gamma_correction(test_image, gamma_value)
        test_gradient = hog.compute_gradients(test_image)
        cell_histograms, _ = hog.compute_weighted_vote(test_gradient)
        hog_blocks, _ = hog.normalize_blocks(cell_histograms)
        good_test_set_hog[image_index] = hog_blocks.ravel()
        image_index += 1


    bad_test_set = get_bad_test_set()
    image_count = len(bad_test_set)
    bad_test_set_hog = np.empty((image_count, 3780))
    image_index = 0
    for image in bad_test_set:
        test_image = cv2.imread(image)
        test_image = cv2.resize(test_image, (64, 128))
        test_image = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)
        test_image = hog.gamma_correction(test_image, gamma_value)
        test_gradient = hog.compute_gradients(test_image)
        cell_histograms, _ = hog.compute_weighted_vote(test_gradient)
        hog_blocks, _ = hog.normalize_blocks(cell_histograms)
        bad_test_set_hog[image_index] = hog_blocks.ravel()
        image_index += 1


    good_test_results = np.empty(len(good_test_set))
    test_index = 0
    for hog in good_test_set_hog:
        good_test_results[test_index] = clf.predict([hog])
        test_index += 1

    bad_test_results = np.empty(len(bad_test_set))
    test_index = 0
    for hog in bad_test_set_hog:
        bad_test_results[test_index] = clf.predict([hog])
        test_index += 1


    print(np.sum(good_test_results))
    print(np.sum(bad_test_results))


    cv2.destroyAllWindows()
