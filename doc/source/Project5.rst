Welcome to CIS 693 -- Project 4 documentation!
==============================================
Abstract
--------
This project looks a at two different approaches to 3D image
reconstruction. In the first approach the intrinsic camera properties
are used to take a pinhole camera model and attempt to develop a 3D
model of an image based upon its input RGB images and corresponding
depth images. The images under test are captured from a Microsoft
Kinect v1 camera and two different view points will be provided. From
This the Pinhole approach will be used to make a cloud plot of a
reconstructed images and then the Open3D library will be used to
use the ICP Point-to-Plane method to make a 3D reconstruction using
the Poise Graph technique.

Introduction
------------
3D reconstruction is required for many different applications. in this
project the approach to trying to reconstruct a 3D model based upon
input images will be examined. On the one hand, some images need to
use mathematical techniques to reconstruct depth data from an image.
This is aided in using multiple different perspectives and trying
approaches like triangulation to achieve a reconstruction of depth
data from a series of 2D images.

In this project two techniques will be looked at.  The first technique
will be to define the parameters of a camera to project what the actual
3D image reconstructing. This will be done from the ideal case where
the lens is presented as a pin hole and allowed to have its point
transformation project from the camera plane to the image plane

The second approach will be to use two different sets of images and their
depth maps to reconstruct a stereo image of a scene using the Open3D library.
This will use a pose graph to register the two view points and produce a
singular point cloud of the the view points of both sets of images.




Local Binary Pattern (LBP) Functions
------------------------------------
.. automodule:: Project5.3d_reconstruct
   :members:

Classifier Output
-----------------
.. program-output:: python3 ../../Project5/3d_reconstruct.py


Conclusion
----------
This project examines a few different ways to reconstruct a 3d image
from 2d data.  Through experimentation the first approach required far
fewer requirements on what is needed; while the second approach, using
ICP reconstruction was much more efficient. In both cases this provided
an examination on different ways in which a 3D reconstruction can be
generated.