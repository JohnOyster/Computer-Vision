#!/usr/bin/env python3
"""CIS 693 - Project 5.

Author: John Oyster
Date:   June 26, 2020
Description:

    The objective of this project is to perform 3D indoor scene reconstruction.

    1. Write a program to create a 3D point cloud model based on the pinhole
       camera model. Use one RGB-D image (image1 rgb with image1 depth) that
       is captured using a Kinect camera, see included Data5.

       The essential parameters of RGB-D Microsoft Kinect Camera are as following:

       - Focal length:
            f_x = 521
            f_y = 521
       - Camera center:
            c_x = 325
            c_y = 250

    2. Repeat part 1 using two RGB-D images (image1 rgb with image1 depth and
       image2 rgb with image2 depth) to estimate the camera position assuming
       the camera position of image1 is at the origin (0,0,0).

       - To estimate the camera transformation matrix (T), apply Iterative
         Closed Point (ICP) method using point-to-plane error metric (or you
         may use any built-in function).
       - Once the T is estimated, transform point cloud of image 2 to image 1
         camera coordinates (you use any built-in function).
       - At the end, merge the two 3D point clouds.
"""
#  Copyright (c) 2020. John Oyster in agreement with Cleveland State University.
import copy
import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import open3d as o3d


def point_cloud_pinhole(rgb_image, depth_image):
    # Get image properties
    width, height = rgb_image.shape[:2]
    depth_resolution = np.iinfo(depth_image.dtype).max

    # Define camera properties
    # Focal length:
    f_x = 521.0
    f_y = 521.0

    # Camera center:
    c_x = 325.0
    c_y = 250.0

    # Check images
    if rgb_image.shape[:2] != depth_image.shape[:2]:
        print("[DEBUG] Images need to have same shape")

    # Need to estimate the Image Transformation Matrix Ti
    # Where,
    #            Ti = [[R_i, t_i],
    #                  [0,   1]]
    #            R_i is from SO(3) rotation matrix
    #            t_ is from R^3 translation vector
    #
    #            p = pi(P) = (f_x (X/Z) + c_x,  f_y (y/Z) + c_y)^T
    cloud_list = []
    for row in range(width):
        for col in range(height):
            depth_val = np.float32(depth_image[row, col]) / 10000
            if depth_val == 0:
                continue
            projected_x = depth_val * (row - c_x) / f_x
            projected_y = depth_val * (col - c_y) / f_y
            r_pixel, g_pixel, b_pixel = rgb_image[row, col]
            datum = (projected_x, projected_y, depth_val, r_pixel, g_pixel, b_pixel)
            cloud_list.append(datum)

    return np.array(cloud_list)


def plot_point_cloud(cloud_list):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(cloud_list[:, 0].min(), cloud_list[:, 0].max())
    ax.set_ylim(cloud_list[:, 1].min(), cloud_list[:, 1].max())
    ax.set_zlim(cloud_list[:, 2].min(), cloud_list[:, 2].max())
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    ax.set_zlabel('Depth')
    ax.scatter(cloud_list[:, 0], cloud_list[:, 1], cloud_list[:, 2])
    plt.show()


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


# =====================================================================
# Following functions taken from Open3D Documentation
# =====================================================================
def pairwise_registration(source, target):
    voxel_size = 0.05
    max_correspondence_distance_coarse = voxel_size * 15
    max_correspondence_distance_fine = voxel_size * 1.5
    print("Apply point-to-plane ICP")
    icp_coarse = o3d.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp


def full_registration(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):
    pose_graph = o3d.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id])
            print("Build o3d.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.registration.PoseGraphNode(np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.registration.PoseGraphEdge(source_id,
                                                   target_id,
                                                   transformation_icp,
                                                   information_icp,
                                                   uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.registration.PoseGraphEdge(source_id,
                                                   target_id,
                                                   transformation_icp,
                                                   information_icp,
                                                   uncertain=True))
    return pose_graph

# =====================================================================
# Above Functions taken from Open3D Documentation
# =====================================================================


def main():
    """Execute this routine if this file is called directly.

    This function is used to test the parameters of the SIFT method
    and make sure that it works.

    :return:        Errno = 0 if good
    :rtype:         int
    """
    # Get Images
    image1_rgb = cv2.imread("./Data/rgb/image1.png")
    image1_rgb = cv2.cvtColor(image1_rgb, cv2.COLOR_BGR2RGB)
    image1_depth = cv2.imread("./Data/depth/image1.png")
    image1_depth = cv2.cvtColor(image1_depth, cv2.COLOR_BGR2GRAY)
    image2_rgb = cv2.imread("./Data/rgb/image2.png")
    image2_rgb = cv2.cvtColor(image2_rgb, cv2.COLOR_BGR2RGB)
    image2_depth = cv2.imread("./Data/depth/image2.png")
    image2_depth = cv2.cvtColor(image2_depth, cv2.COLOR_BGR2GRAY)

    # -----------------------------------------------------------------
    # Part 1 - Pinhole Camera Model
    pin_cloud = point_cloud_pinhole(image1_rgb, image1_depth)
    plot_point_cloud(pin_cloud)

    # -----------------------------------------------------------------
    # Part 2

    # Define parameters
    voxel_size = 0.02

    # Define camera parameters from Part 1
    camera_parameters = o3d.camera.PinholeCameraIntrinsic()
    camera_parameters.set_intrinsics(640, 480, 521.0, 521.0, 325.0, 250.0)

    # Acquire RGB and Depth images using Open3D --> To RGBD Format
    # Scaling factor of 10000 taken from course slides
    image1_rgb = o3d.io.read_image("./Data/rgb/image1.png")
    image1_depth = o3d.io.read_image("./Data/depth/image1.png")
    image2_rgb = o3d.io.read_image("./Data/rgb/image2.png")
    image2_depth = o3d.io.read_image("./Data/depth/image2.png")
    image1_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        image1_rgb, image1_depth, depth_scale=10000, convert_rgb_to_intensity=False)
    image2_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        image2_rgb, image2_depth, depth_scale=10000, convert_rgb_to_intensity=False)

    # Plot sample source data
    plt.subplot(1, 2, 1)
    plt.title('RGB Image')
    plt.imshow(image1_rgbd.color)
    plt.subplot(1, 2, 2)
    plt.title('Depth Image')
    plt.imshow(image1_rgbd.depth)
    plt.show()

    # Generate Point clouds from RGBD images
    image1_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(image1_rgbd, camera_parameters)
    image2_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(image2_rgbd, camera_parameters)

    # Define initial conditions to perform ICP Point-to-Plane registration
    threshold = 0.02
    # Initial transformation matrix from Open3D documentation
    transformation_initial = np.asarray([[0.862, 0.011, -0.507, 0.5],
                                         [-0.139, 0.967, -0.215, 0.7],
                                         [0.487, 0.255, 0.835, -1.4],
                                         [0.0, 0.0, 0.0, 1.0]])

    # Perform the Initial alignment
    print("[INFO] Initial Alignment")
    print(image2_pcd)
    evaluation = o3d.registration.evaluate_registration(
        image1_pcd, image2_pcd, threshold, transformation_initial)
    print(evaluation)

    # Normalize both point clouds using Voxel sampling to get better results
    # NOTE: This was recommendation in Open3D documentation
    image1_pcd.voxel_down_sample(voxel_size=voxel_size)
    image2_pcd.voxel_down_sample(voxel_size=voxel_size)

    # Initialize the vertex information from the point clouds
    image1_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    image2_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Perform ICP Point-to-Plane registration and find true Transformation matrix
    print("[INFO] Apply point-to-plane ICP")
    registered_images = o3d.registration.registration_icp(
        image1_pcd, image2_pcd, threshold, transformation_initial,
        o3d.registration.TransformationEstimationPointToPlane())
    print(registered_images)
    print("[INFO] Transformation Matrix:")
    print(registered_images.transformation)
    draw_registration_result(image1_pcd, image2_pcd, registered_images.transformation)

    # Merge Point Clouds
    pcds = [image1_pcd, image2_pcd]
    print("[INFO] Full registration")
    max_correspondence_distance_coarse = voxel_size * 15
    max_correspondence_distance_fine = voxel_size * 1.5
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        pose_graph = full_registration(pcds,
                                       max_correspondence_distance_coarse,
                                       max_correspondence_distance_fine)

    pcd_combined = o3d.geometry.PointCloud()
    pcd_combined = image1_pcd.transform(pose_graph.nodes[0].pose)
    pcd_combined += image2_pcd.transform(pose_graph.nodes[1].pose)
    pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=voxel_size)
    o3d.visualization.draw_geometries([pcd_combined_down])

    return 0


if __name__ == '__main__':
    main()

    cv2.destroyAllWindows()
