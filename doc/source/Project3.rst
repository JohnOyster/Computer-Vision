Welcome to CIS 693 -- Project 3 documentation!
==============================================
Abstract
--------
This project looks at the Local Binary Pattern (LBP) method for extracting
features out of a an image. This method will be applied to the facial
recognition problem. Using the AT&T 64x64 facial data set, this program
will define the LBP descriptor for each image using a 16x16 block size.
This is done using Python 3.6, Numpy, and the Scikit-Learn libraries to
generate the LBP descriptor and classify it against different classifiers.

Introduction
------------


Local Binary Pattern (LBP) Functions
------------------------------------
.. automodule:: Project3.lbp
   :members:

Scikit-Learn Classifier Program
-------------------------------
.. automodule:: Project3.train_svm
   :members:

Conclusion
----------
Cool beans