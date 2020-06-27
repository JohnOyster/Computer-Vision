Welcome to CIS 693 -- Project 4 documentation!
==============================================
Abstract
--------
This project looks at Scale-Invariant Feature Transform (SIFT) method for
identifying key points in an image. The idea behind this technique is
to look for points on interest within an image and record or register them
to be used at a later time for some computational operation.  In this
project, the SIFT technique will be used to register key points of an
image and use them to transform the second of a pair of images into
alignment with the first using the key points obtained from the SIFT
technique. An implementation of this will be show in Python 3.6.x using
the OpenCV library.

Introduction
------------
The Scale-Invariant Feature Transform (SIFT) method is a computer vision
technique that is looking to find definable aspects of an image and
register those points of interest.  A fundamental idea behind this approach
is based in the Harris corner detection method. In that method, well
defined cornet shapes within an image as identified and registered to be
used at a later date. These key points are programatically saved to be
utilized later.

With a serious of key points that have been registered, it is possible
to now perform different geometric operations within the image to attempt
to extract key data as well at align the image towards the previously
identified data. In this way we can used this technique to either
simply align an image into the same perspective as a source image; or
more applicatively, these key points can be used to stitch different
images together to provide a larger destination image or larger topographical
analysis of a series of images.,


Local Binary Pattern (LBP) Functions
------------------------------------
.. automodule:: SIFT.sift
   :members:

Classifier Output
-----------------
.. program-output:: python3 ../../SIFT/sift.py

Analysis Question
-----------------
This project posed the following question, "Which feature matching
algorithm works better and why?"

Both the OpenCv approach as well as the custom approach that we defined
within this project produce a very similar error value. This is
because both matching approaches use the same intent in finding the
distance between key points and using that matched distance value in
later computations. That's why using the Eucledian distance, L1 norm,
or Hamming distance will result in a similar result.

Conclusion
----------
This technique is beneficial for a bunch of different applications.
By identifying the key points and a descriptor of its features it
is possible to apply this technique towards corner detection, as well
as image registration. In this project we looked at using this
technique to register an an image to then in turn apply it towards
an image manipulation that used the an Affine matrix to transform
an image from one perspective into another. In the case of this
project, we were successful in the image transformation with both
a call to OpenCv as well as implementing our own custom approach
using the Euclidean distance between key points.