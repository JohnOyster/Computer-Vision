#  Copyright (c) 2020. John Oyster in agreement with Cleveland State University.

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
"""
#  Copyright (c) 2020. John Oyster in agreement with Cleveland State University.
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from Project3 import lbp


def generate_lbp_descriptors(data):
    """

    :param data:
    :return:
    """
    entry_count = sum([len(data[x]) for x in data if isinstance(data[x], list)])
    for person_id, feature_set in data.items():
        image_count = len(feature_set)
        random.shuffle(feature_set)
        descriptors = np.array([lbp.calculate_lbp(image) for image in feature_set])
        tags = np.full(image_count, person_id)
        yield descriptors, tags


if __name__ == '__main__':
    number_of_trials = 10
    mat_data = lbp.process_mat_file()

    # Set Classifier parameters
    C = 1.0  # SVM regularization parameter
    cv = 2  # Cross-validation sets

    linear_score = np.empty(number_of_trials)
    linear_xref = np.empty((number_of_trials, cv))
    rbf_score = np.empty(number_of_trials)
    rbf_xref = np.empty((number_of_trials, cv))
    poly_score = np.empty(number_of_trials)
    poly_xref = np.empty((number_of_trials, cv))
    knn_score = np.empty(number_of_trials)
    knn_xref = np.empty((number_of_trials, cv))

    for trial in range(number_of_trials):
        print("[INFO] Processing Trial {}".format(trial+1))
        train_set, valid_set, test_set = [], [], []
        train_tag, valid_tag, test_tag = [], [], []
        for X, y in generate_lbp_descriptors(mat_data):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)
            X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.15, shuffle=True)
            train_set.append(X_train)
            train_tag.append(y_train)
            valid_set.append(X_valid)
            valid_tag.append(y_valid)
            test_set.append(X_test)
            test_tag.append(y_test)
        train_set = np.array(train_set).reshape(-1, 4096)
        train_tag = np.array(train_tag).ravel()
        valid_set = np.array(valid_set).reshape(-1, 4096)
        valid_tag = np.array(valid_tag).ravel()
        test_set = np.array(test_set).reshape(-1, 4096)
        test_tag = np.array(test_tag).ravel()

        linear_svc = SVC(kernel='linear', C=C).fit(train_set, train_tag)
        rbf_svc = SVC(kernel='rbf', C=C).fit(train_set, train_tag)
        poly_svc = SVC(kernel='poly', C=C).fit(train_set, train_tag)
        knn = KNeighborsClassifier(n_neighbors=5).fit(train_set, train_tag)
        linear_xref[trial] = cross_val_score(linear_svc, valid_set, valid_tag, cv=cv)
        rbf_xref[trial] = cross_val_score(rbf_svc, valid_set, valid_tag, cv=cv)
        poly_xref[trial] = cross_val_score(poly_svc, valid_set, valid_tag, cv=cv)
        knn_xref[trial] = cross_val_score(knn, valid_set, valid_tag, cv=cv)
        linear_score[trial] = linear_svc.score(test_set, test_tag)
        rbf_score[trial] = rbf_svc.score(test_set, test_tag)
        poly_score[trial] = poly_svc.score(test_set, test_tag)
        knn_score[trial] = knn.score(test_set, test_tag)

    print(40*'=')
    print("[INFO] Linear Xref Accuracy = {}".format(linear_xref.ravel().mean()))
    print("[INFO] Linear Xref Std. Deviation = {}".format(linear_xref.ravel().std()))
    print("[INFO] RBF Xref Accuracy = {}".format(rbf_xref.ravel().mean()))
    print("[INFO] RBF Xref Std. Deviation = {}".format(rbf_xref.ravel().std()))
    print("[INFO] Poly Xref Accuracy = {}".format(poly_xref.ravel().mean()))
    print("[INFO] Poly SXref td. Deviation = {}".format(poly_xref.ravel().std()))
    print("[INFO] kNN Xref Accuracy = {}".format(knn_xref.ravel().mean()))
    print("[INFO] kNN Xref Std. Deviation = {}".format(knn_xref.ravel().std()))
    print(40*'=')
    print("[INFO] Linear Accuracy = {}".format(linear_score.mean()))
    print("[INFO] Linear Std. Deviation = {}".format(linear_score.std()))
    print("[INFO] RBF Accuracy = {}".format(rbf_score.mean()))
    print("[INFO] RBF Std. Deviation = {}".format(rbf_score.std()))
    print("[INFO] Poly Accuracy = {}".format(poly_score.mean()))
    print("[INFO] Poly Std. Deviation = {}".format(poly_score.std()))
    print("[INFO] kNN Accuracy = {}".format(knn_score.mean()))
    print("[INFO] kNN Std. Deviation = {}".format(knn_score.std()))




