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
A Local Binary Pattern (LBP) is a method of taking a pixel of interest and
examining its surrounding pixels to determine the texture of the image.
Initial applications of this technique could be used to determine a surface's
composition like brick or cloth; however, another application of this method
can be used to extrapolate more complex textures. Once such application of
this technique is in the use of facial recognition.

In facial recognition, this technique can be used to determine the positioning,
skew, and unique factors of a face. These and more aspects can be abstracted
into a defined texture for a face.  An issue that comes with facial recognition
is the event in which an image is presented with different points of view or
different lighting. In these situations, a face's texture will change slightly.
The application of LBP can be used to normalize values over chunks or regions
of an image in order to average out these types of dynamic shifts in a target.

This program looks to apply the LBP method towards facial recognition. A
data set of different people from different lighting and perspectives are
used to define a classifier program. This program will then use the LBP
descriptors to create a prediction of which image belongs to which subject.

Local Binary Pattern (LBP) Functions
------------------------------------
.. automodule:: Project3.lbp
   :members:

Scikit-Learn Classifier Program
-------------------------------
.. automodule:: Project3.train_svm
   :members:

Classifier Output
-----------------
.. program-output:: python3 ../../Project3/train_svm.py

Conclusion
----------
In this program, it was seen that different classifiers can be used to
predict which image belongs to which test subject.  In this particular
application, the Linear SVM classifier was most consistent in
predicting which image belongs to which test subject.  An interesting
observation is that the Poly kernel was the next the second most accurate.
This can be attributed to the non-linearity of facial movement and lighting.

Besides the SVM classifier, the k-Nearest Neighbors classifier (kNN) was
also used and exhibited the worst performance. This seems to be attributed
either to the program's tuning or due to the unpredictability in how a face
will move.

More often than not, these approach has shown significant accuracy in being
able to accurately predict which image blongs to which test subject using
an LBP classifier.

