#!/usr/bin/env python

import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import os
from time import time
from gradslam.slam.pointfusion import PointFusion
from gradslam.slam.icpslam import ICPSLAM
from gradslam import Pointclouds, RGBDImages
from threading import RLock
from scipy import interpolate
# ROS
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import message_filters
import tf2_ros
from ros_numpy import msgify, numpify
from tf.transformations import quaternion_from_matrix


def interpolate_missing_pixels(
        image: np.ndarray,
        mask: np.ndarray,
        method: str = 'nearest',
        fill_value: int = 0
):
    """
    :param image: a 2D image
    :param mask: a 2D boolean image, True indicates missing values
    :param method: interpolation method, one of
        'nearest', 'linear', 'cubic'.
    :param fill_value: which value to use for filling up data outside the
        convex hull of known pixel values.
        Default is 0, Has no effect for 'nearest'.
    :return: the image with missing values interpolated
    """

    h, w = image.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))

    known_x = xx[~mask]
    known_y = yy[~mask]
    known_v = image[~mask]
    missing_x = xx[mask]
    missing_y = yy[mask]

    interp_values = interpolate.griddata(
        (known_x, known_y), known_v, (missing_x, missing_y),
        method=method, fill_value=fill_value
    )

    interp_image = image.copy()
    interp_image[missing_y, missing_x] = interp_values

    return interp_image


class GradslamROS:
    def __init__(self, odometry='gt', height: int = 240, width: int = 320):
        self.bridge = CvBridge()
        self.tf = tf2_ros.Buffer()
        self.tf_sub = tf2_ros.TransformListener(self.tf)
        self.world_frame = 'subt'
        self.robot_frame = 'X1_ground_truth'
        self.camera = 'front'  # 'right', 'front'
        self.camera_frame = f'X1/base_link/{self.camera}_realsense_optical'
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.slam = PointFusion(odom=odometry, dsratio=4, device=self.device)
        self.width, self.height = width, height
        self.pointclouds = Pointclouds(device=self.device)
        self.prev_frame = None
        self.route = Path()
        self.route.header.frame_id = self.world_frame
        self.route_pub = rospy.Publisher('~route', Path, queue_size=2)
        self.pc_pub = rospy.Publisher('~cloud', PointCloud2, queue_size=1)
        self.extrinsics_lock = RLock()
        self.map_step = 16
        self.depth_pub = rospy.Publisher('~depth_proc', Image, queue_size=1)

        self.robot2camera = self.get_extrinsics()
        rospy.logdebug(f'Got extrinsics: {self.robot2camera}')

        # Subscribe to topics
        caminfo_sub = message_filters.Subscriber(f'/X1/{self.camera}_rgbd/optical/camera_info', CameraInfo)
        rgb_sub = message_filters.Subscriber(f'/X1/{self.camera}_rgbd/optical/image_raw', Image)
        depth_sub = message_filters.Subscriber(f'/X1/{self.camera}_rgbd/depth/optical/image_raw', Image)

        # Synchronize the topics by time
        ats = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub, caminfo_sub], queue_size=5, slop=0.05)
        ats.registerCallback(self.callback)

    def get_extrinsics(self):
        with self.extrinsics_lock:
            while not rospy.is_shutdown():
                try:
                    robot2camera = self.tf.lookup_transform(self.robot_frame, self.camera_frame,
                                                            rospy.Time.now(), rospy.Duration.from_sec(1.0))
                    robot2camera = numpify(robot2camera.transform)
                    return robot2camera
                except (tf2_ros.TransformException, tf2_ros.ExtrapolationException) as ex:
                    rospy.logwarn('Could not transform from robot %s to camera %s: %s.',
                                  self.robot_frame, self.camera_frame, ex)

    def callback(self, rgb_msg, depth_msg, caminfo_msg):
        t0 = time()
        try:
            world2robot = self.tf.lookup_transform(self.world_frame, self.robot_frame,
                                          rospy.Time.now(), rospy.Duration.from_sec(1.0))
            world2robot = numpify(world2robot.transform)
        except (tf2_ros.TransformException, tf2_ros.ExtrapolationException) as ex:
            rospy.logwarn('Could not transform from world %s to robot %s: %s.',
                          self.world_frame, self.robot_frame, ex)
            return
        rospy.logdebug('Transformation search took: %.3f', time() - t0)
        world2camera = world2robot @ self.robot2camera
        pose_gt = torch.as_tensor(world2camera, dtype=torch.float32).view(1, 1, 4, 4)

        try:
            # get rgb image
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, rgb_msg.encoding)
            rgb_image = np.asarray(rgb_image, dtype=np.float32)
            rgb_image = cv2.resize(rgb_image,
                                   (self.width, self.height),
                                   interpolation=cv2.INTER_LINEAR)
            # get depth image
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, depth_msg.encoding)
            depth_image = np.asarray(depth_image, dtype=np.float32)
            if self.slam.odom != 'gt':
                # depth_image = cv2.medianBlur(depth_image, 5)  # to filter inf outliers
                # depth_image[depth_image == np.inf] = np.max(depth_image[depth_image != np.inf])  # np.nan, 10.0
                depth_image = interpolate_missing_pixels(depth_image,
                                                         mask=np.asarray(depth_image == np.inf),
                                                         method='nearest',
                                                         fill_value=10.0)
                # depth_image = cv2.resize(depth_image,
                #                          (self.width, self.height),
                #                          interpolation=cv2.INTER_NEAREST)
                # depth_proc_msg = msgify(Image, depth_image, encoding=depth_msg.encoding)
                # depth_proc_msg.header = depth_msg.header
                # self.depth_pub.publish(depth_proc_msg)
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        # get intrinsic params
        # TODO: subscribe ones in another callback function
        k = torch.as_tensor(caminfo_msg.K, dtype=torch.float32).view(3, 3)
        K = torch.eye(4)
        K[:3, :3] = k
        intrins = K.view(1, 1, 4, 4)

        assert rgb_image.shape[:2] == depth_image.shape
        w, h = rgb_image.shape[:2]
        rgb_image = torch.from_numpy(rgb_image).view(1, 1, w, h, 3)
        depth_image = torch.from_numpy(depth_image).view(1, 1, w, h, 1)

        # create gradslam input
        live_frame = RGBDImages(rgb_image, depth_image, intrins, pose_gt).to(self.device)
        rospy.logdebug('Data preprocessing took: %.3f', time()-t0)

        # SLAM inference
        t0 = time()
        self.pointclouds, live_frame.poses = self.slam.step(self.pointclouds, live_frame, self.prev_frame)
        self.prev_frame = live_frame if self.slam.odom != 'gt' else None
        rospy.logdebug(f"Position: {live_frame.poses[..., :3, 3].squeeze()}")
        rospy.logdebug('SLAM inference took: %.3f', time() - t0)

        # publish odometry / path
        # TODO: publish ground truth path as well
        t0 = time()
        assert live_frame.poses.shape == (1, 1, 4, 4)
        pose = PoseStamped()
        pose.header.frame_id = self.world_frame
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = live_frame.poses[..., 0, 3]
        pose.pose.position.y = live_frame.poses[..., 1, 3]
        pose.pose.position.z = live_frame.poses[..., 2, 3]
        q = quaternion_from_matrix(live_frame.poses[0, 0].cpu().numpy())
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]

        self.route.poses.append(pose)
        self.route.header.stamp = rospy.Time.now()
        self.route_pub.publish(self.route)

        # publish point cloud
        n_pts = np.ceil(self.pointclouds.points_padded.shape[1] / self.map_step).astype(int)
        cloud = np.zeros((n_pts,), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                          ('r', 'f4'), ('g', 'f4'), ('b', 'f4')])
        for i, f in enumerate(['x', 'y', 'z']):
            cloud[f] = self.pointclouds.points_padded[..., i].squeeze().cpu().numpy()[::self.map_step]
        for i, f in enumerate(['r', 'g', 'b']):
            cloud[f] = self.pointclouds.colors_padded[..., i].squeeze().cpu().numpy()[::self.map_step] / 255.
        pc_msg = msgify(PointCloud2, cloud)
        pc_msg.header.stamp = rospy.Time.now()
        pc_msg.header.frame_id = self.world_frame
        self.pc_pub.publish(pc_msg)
        rospy.logdebug('Data publishing took: %.3f', time() - t0)


if __name__ == '__main__':
    rospy.init_node('gradslam_ros', log_level=rospy.INFO)
    odometry = rospy.get_param('~odometry')  # gt, icp, gradicp
    proc = GradslamROS(odometry=odometry)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
