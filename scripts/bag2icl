#!/usr/bin/env python

import numpy as np
import cv2
import os
from time import time
from scipy import interpolate

import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
import message_filters
import tf2_ros
from ros_numpy import msgify, numpify
import rospkg


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


class Processor:
    def __init__(self, height: int = 240, width: int = 320):
        self.bridge = CvBridge()
        self.tf = tf2_ros.Buffer()
        self.tf_sub = tf2_ros.TransformListener(self.tf)
        self.world_frame = 'subt'
        self.camera_frame = 'X1_ground_truth'  # 'X1/base_link/front_realsense_optical'
        self.folder_name = 'explorer_x1_rgbd_traj_{0}/living_room_traj1_frei_png'.format(time())
        self.rgb_path = os.path.join(rospkg.RosPack().get_path('gradslam_ros'),
                                     f'data/{self.folder_name}/rgb/')
        self.depth_path = os.path.join(rospkg.RosPack().get_path('gradslam_ros'),
                                       f'data/{self.folder_name}/depth/')
        self.caminfo_path = os.path.join(rospkg.RosPack().get_path('gradslam_ros'),
                                         f'data/{self.folder_name}/caminfo/')
        self.tfs_path = os.path.join(rospkg.RosPack().get_path('gradslam_ros'),
                                     f'data/{self.folder_name}/livingRoom1n.gt.sim')
        self.assocs_path = os.path.join(rospkg.RosPack().get_path('gradslam_ros'),
                                        f'data/{self.folder_name}/associations.txt')
        self.image_n = 0
        self.save_data = rospy.get_param('~save_data', True)
        self.width, self.height = width, height

        if not os.path.isdir(os.path.join(rospkg.RosPack().get_path('gradslam_ros'),
                                          f'data/{self.folder_name}')):
            os.makedirs(self.rgb_path)
            os.makedirs(self.depth_path)
            os.makedirs(self.caminfo_path)

        # Subscribe to topics
        info_sub = message_filters.Subscriber('/X1/front_rgbd/optical/camera_info', CameraInfo)
        rgb_sub = message_filters.Subscriber('/X1/front_rgbd/optical/image_raw', Image)
        depth_sub = message_filters.Subscriber('/X1/front_rgbd/depth/optical/image_raw', Image)

        # Synchronize the topics by time
        ats = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub, info_sub], queue_size=5, slop=0.1)
        ats.registerCallback(self.callback)

    def callback(self, rgb_msg, depth_msg, caminfo_msg):
        t0 = time()
        try:
            tf = self.tf.lookup_transform(self.world_frame, self.camera_frame,
                                          rospy.Time.now(), rospy.Duration.from_sec(1.0))
            rospy.logdebug('Found transform in %.3f sec', time()-t0)
        except tf2_ros.TransformException as ex:
            rospy.logerr('Could not transform from world %s to camera %s: %s.',
                         self.world_frame, self.camera_frame, ex)
            return
        try:
            # get rgb image
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, rgb_msg.encoding)
            rgb_image = np.asarray(rgb_image, dtype=np.float32)
            rgb_image = cv2.resize(rgb_image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            # get depth image
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, depth_msg.encoding)
            depth_image = np.asarray(depth_image, dtype=np.float32)
            depth_image = cv2.resize(depth_image, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
            depth_image = interpolate_missing_pixels(depth_image,
                                                     mask=np.asarray(depth_image == np.inf),
                                                     method='nearest',
                                                     fill_value=10.0)
        except CvBridgeError as e:
            rospy.logerr(e)
            return
        # get pose
        T = numpify(tf.transform)
        # get intrinsics
        K = np.asarray(caminfo_msg.K, dtype=np.float32).reshape([3, 3])
        assert rgb_image.shape[:2] == depth_image.shape
        assert T.shape == (4, 4)

        if self.save_data:
            # write images
            np.save(self.rgb_path + str(self.image_n) + '.npy', rgb_image)
            cv2.imwrite(self.rgb_path+str(self.image_n)+'.png', rgb_image)
            np.save(self.depth_path + str(self.image_n)+'.npy', depth_image)
            depth_image = 1000. * depth_image
            depth_image = depth_image.astype(np.uint16)
            cv2.imwrite(self.depth_path + str(self.image_n) + '.png', depth_image)

            # write intrinsics
            np.save(self.caminfo_path + str(self.image_n) + '.npy', K)

            # write associations
            with open(self.assocs_path, 'a') as f:
                f.write(str(self.image_n)+' depth/'+str(self.image_n)+'.png '+str(self.image_n)+' rgb/'+str(self.image_n)+'.png')
                f.write('\n')

            # write transformations
            with open(self.tfs_path, 'a') as f:
                for line in np.matrix(T[:3, :]):
                    np.savetxt(f, line, fmt='%.2f')
                f.write('\n')

            self.image_n += 1
            rospy.loginfo('Writing took: %.3f sec', time() - t0)


if __name__ == '__main__':
    rospy.init_node('bag2data', log_level=rospy.DEBUG)
    ip = Processor()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
