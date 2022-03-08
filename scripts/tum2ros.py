#!/usr/bin/env python

import torch
from torch.utils.data import DataLoader
from gradslam.datasets.icl import ICL
from gradslam.datasets.tum import TUM
from gradslam.slam.pointfusion import PointFusion
from gradslam import Pointclouds, RGBDImages
from gradslam.datasets.datautils import normalize_image
from time import time
from tqdm import tqdm
import numpy as np
# ROS
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, TransformStamped
import message_filters
import tf2_ros
from ros_numpy import msgify, numpify
from tf.transformations import quaternion_from_matrix
import rospkg


def publish_tf_pose(pose, quat, child_frame_id, frame_id="world"):
    assert len(pose) == 3
    assert len(quat) == 4
    br = tf2_ros.TransformBroadcaster()
    t = TransformStamped()
    t.header.stamp = rospy.Time.now()
    t.header.frame_id = frame_id
    t.child_frame_id = child_frame_id
    t.transform.translation.x = pose[0]
    t.transform.translation.y = pose[1]
    t.transform.translation.z = pose[2]
    t.transform.rotation.x = quat[0]
    t.transform.rotation.y = quat[1]
    t.transform.rotation.z = quat[2]
    t.transform.rotation.w = quat[3]
    br.sendTransform(t)


def get_camera_info_msg(image_width, image_height, K,
                        frame_id="camera_frame",
                        distortion_model="plumb_bob"):
    camera_info_msg = CameraInfo()
    camera_info_msg.header.frame_id = frame_id
    camera_info_msg.header.stamp = rospy.Time.now()
    camera_info_msg.width = image_width
    camera_info_msg.height = image_height
    camera_info_msg.K = K
    camera_info_msg.distortion_model = distortion_model
    return camera_info_msg


class GradslamROS:
    def __init__(self,
                 dataset_path: str,
                 seqlen: float = 100,
                 odometry: str = 'gt',
                 width: int = 320,
                 height: int = 240):
        # select device
        self.width, self.height = width, height
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        dataset = TUM(dataset_path, seqlen=seqlen, height=height, width=width)
        loader = DataLoader(dataset=dataset, batch_size=1)
        colors, depths, intrinsics, poses, *_ = next(iter(loader))
        self.rgbdimages = RGBDImages(colors, depths, intrinsics, poses, channels_first=False, device=self.device)
        self.slam = PointFusion(odom=odometry, dsratio=1, device=self.device)

        self.world_frame = 'world'
        self.route = Path()
        self.route.header.frame_id = self.world_frame
        self.route_pub = rospy.Publisher('~route', Path, queue_size=2)
        self.pc_pub = rospy.Publisher('~cloud', PointCloud2, queue_size=1)
        self.caminfo_pub = rospy.Publisher('~frustum', CameraInfo, queue_size=1)

    def run(self):
        # SLAM: step by step
        seq_len = self.rgbdimages.shape[1]
        initial_poses = torch.eye(4, device=self.device).view(1, 1, 4, 4)
        prev_frame = None
        pointclouds = Pointclouds(device=self.device)
        for s in tqdm(range(seq_len)):
            if rospy.is_shutdown():
                break
            t1 = time()
            live_frame = self.rgbdimages[:, s]
            # print(torch.min(live_frame.depth_image), torch.max(live_frame.depth_image), live_frame.depth_image.dtype)
            # if s == 0 and live_frame.poses is None:
            #     live_frame.poses = initial_poses

            pointclouds, live_frame.poses = self.slam.step(pointclouds, live_frame, prev_frame)
            rospy.logdebug(f'SLAM step took {(time() - t1):.3f} sec')
            prev_frame = live_frame if self.slam.odom != 'gt' else None

            # publish odometry / path
            assert live_frame.poses.shape == (1, 1, 4, 4)
            pose = PoseStamped()
            pose.header.frame_id = self.world_frame
            pose.header.stamp = rospy.Time.now()
            p = live_frame.poses[..., :3, 3].squeeze()
            pose.pose.position.x = p[0]
            pose.pose.position.y = p[1]
            pose.pose.position.z = p[2]
            q = quaternion_from_matrix(live_frame.poses[0, 0].cpu().numpy())
            pose.pose.orientation.x = q[0]
            pose.pose.orientation.y = q[1]
            pose.pose.orientation.z = q[2]
            pose.pose.orientation.w = q[3]
            # publish camera tf frame
            publish_tf_pose(pose=p, quat=q, child_frame_id='camera_frame')
            caminfo_msg = get_camera_info_msg(image_width=self.width, image_height=self.height,
                                              K=live_frame.intrinsics.squeeze(),
                                              frame_id='camera_frame')
            self.caminfo_pub.publish(caminfo_msg)

            self.route.poses.append(pose)
            self.route.header.stamp = rospy.Time.now()
            self.route_pub.publish(self.route)

            # publish point cloud
            n_pts = pointclouds.points_padded.shape[1]
            cloud = np.zeros((n_pts,), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                              ('r', 'f4'), ('g', 'f4'), ('b', 'f4')])
            for i, f in enumerate(['x', 'y', 'z']):
                cloud[f] = pointclouds.points_padded[..., i].squeeze().cpu().numpy()
            for i, f in enumerate(['r', 'g', 'b']):
                cloud[f] = pointclouds.colors_padded[..., i].squeeze().cpu().numpy() / 255.
            pc_msg = msgify(PointCloud2, cloud)
            pc_msg.header.stamp = rospy.Time.now()
            pc_msg.header.frame_id = self.world_frame
            self.pc_pub.publish(pc_msg)


if __name__ == "__main__":
    rospy.init_node('gradslam_ros', log_level=rospy.INFO)
    dataset_path = '/home/ruslan/subt/thirdparty/gradslam/examples/tutorials/TUM/'
    odometry = rospy.get_param('~odometry')  # gt, icp, gradicp
    proc = GradslamROS(dataset_path=dataset_path, odometry=odometry, seqlen=500)
    try:
        proc.run()
    except KeyboardInterrupt:
        print("Shutting down")
