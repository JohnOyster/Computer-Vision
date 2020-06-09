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