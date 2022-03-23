from __future__ import absolute_import, division, print_function
import os
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
from tqdm import tqdm
import datetime
import torch
import gradslam
from gradslam import Pointclouds, RGBDImages
from gradslam.slam import PointFusion
import open3d


# TODO: use os and file path instead
# data_dir = "/home/ruslan/data/datasets/kitti_raw"
data_dir = "/home/jachym/KITTI/kitti_raw"
DEPTH_DATA_DIR = "/home/jachym/KITTI/depth_selection/val_selection_cropped"
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

sequence_names = [
    '2011_09_26',
    '2011_09_28',
    '2011_09_29',
    '2011_09_30',
    '2011_10_03'
]


class KITTIRawPoses(object):

    def __init__(self, seq, subseq, path=None):
        if path is None:
            path = os.path.join(data_dir, seq, subseq)
        self.path = path
        self.poses = self.get_poses()
        self.ts = self.get_timestamps(sensor='gps')
        self.ids = range(len(self.ts))
        self.stamps = self.get_stamps(subseq)

    @staticmethod
    def gps_to_ecef(lat, lon, alt, zero_origin=False):
        # https://gis.stackexchange.com/questions/230160/converting-wgs84-to-ecef-in-python
        rad_lat = lat * (np.pi / 180.0)
        rad_lon = lon * (np.pi / 180.0)

        a = 6378137.0
        finv = 298.257223563
        f = 1 / finv
        e2 = 1 - (1 - f) * (1 - f)
        v = a / np.sqrt(1 - e2 * np.sin(rad_lat) * np.sin(rad_lat))

        x = (v + alt) * np.cos(rad_lat) * np.cos(rad_lon)
        y = (v + alt) * np.cos(rad_lat) * np.sin(rad_lon)
        # TODO: check why height from latitude is too high
        # z = (v * (1 - e2) + alt) * np.sin(rad_lat)
        z = alt

        if zero_origin:
            x, y, z = x - x[0], y - y[0], z - z[0]
        return x, y, z

    def get_gps_pose(self, fname):
        assert isinstance(fname, str)
        gps_data = np.genfromtxt(fname)
        lat, lon, alt = gps_data[:3]
        roll, pitch, yaw = gps_data[3:6]
        R = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=False).as_matrix()
        x, y, z = self.gps_to_ecef(lat, lon, alt)
        # convert to 4x4 matrix
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = np.array([x, y, z])
        return pose

    def get_poses(self, zero_origin=False):
        poses = []
        for fname in np.sort(glob.glob(os.path.join(self.path, 'oxts/data/*.txt'))):
            pose = self.get_gps_pose(fname)
            poses.append(pose)
        poses = np.asarray(poses)
        if zero_origin:
            pose0 = poses[0]
            # poses[:, :3, 3] -= pose0[:3, 3]
            poses = np.matmul(np.linalg.inv(pose0)[None], poses)
        return poses

    def get_stamps(self, subseq):
        stamps = list()
        for file in np.sort(glob.glob(os.path.join(self.path, 'oxts/data/*.txt'))):
            file = file.replace(self.path + "/oxts/data/", "").replace(".txt", "")
            stamp = subseq + "_" + file
            stamps.append(stamp)
        return stamps

    def get_timestamps(self, sensor, zero_origin=False):
        assert isinstance(sensor, str)
        assert sensor == 'gps' or sensor == 'lidar'
        if sensor == 'gps':
            folder = 'oxts'
        elif sensor == 'lidar':
            folder = 'velodyne_points'
        else:
            raise ValueError
        timestamps = []
        ts = np.genfromtxt(os.path.join(self.path, folder, 'timestamps.txt'), dtype=str)
        for t in ts:
            date = t[0]
            day_time, sec = t[1].split(".")
            sec = float('0.' + sec)
            stamp = datetime.datetime.strptime("%s_%s" % (date, day_time), "%Y-%m-%d_%H:%M:%S").timestamp() + sec
            timestamps.append(stamp)
        if zero_origin:
            timestamps = [t - timestamps[0] for t in timestamps]
        return timestamps

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.ts[i], self.poses[i]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class KITTIDepth:

    """
    loads depth images, rgb images and intrinsics from
    KITTI depth: http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion
    """

    def __init__(self, path):
        # path directory should contain folders: depth, rgb, intrinsics
        self.path = path
        self.stamps = self.get_stamps()
        self.depths = self.get_depths()
        self.intrinsics = self.get_intrinsics()
        self.rgbs = self.get_rgbs()

    def get_rgbs(self):
        rgbs = list()
        rgb_dir = os.path.join(self.path, "rgb")
        for file in sorted(glob.glob(rgb_dir + "/*.png")):
            rgb = torch.tensor(cv2.imread(file, cv2.IMREAD_UNCHANGED))
            # transform file into gradslam format (B, S, H, W, CH), for now we will use B=1, S=1
            # TODO: change this to allow custom minibatch size
            rgb = torch.unsqueeze(torch.unsqueeze(rgb, 0), 0)
            rgbs.append(rgb)
        return rgbs

    def get_depths(self):
        depths = list()
        depth_dir = os.path.join(self.path, "depth")
        for file in sorted(glob.glob(depth_dir + "/*.png")):
            # TODO: see if there are any better ways to store values, e.g. preprocessing like in DEPTH completion demo
            depth = torch.tensor(cv2.imread(file, cv2.IMREAD_UNCHANGED).astype('int32'))
            # transform file into gradslam format (B, S, H, W, CH), for now we will use B=1, S=1
            depth = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(depth, -1), 0), 0)
            depths.append(depth)
        return depths

    def get_intrinsics(self):
        intrinsics = list()
        intr_dir = os.path.join(self.path, "intrinsics")
        for file in sorted(glob.glob(intr_dir + "/*.txt")):
            intrinsics_i = torch.tensor(np.loadtxt(file), dtype=torch.double)
            # reshape intrinsics into a 4x4 matrix
            intrinsics_i = torch.reshape(intrinsics_i, (3, 3))
            intrinsics_i = torch.cat((intrinsics_i, torch.zeros(3, 1)), 1)
            intrinsics_i = torch.cat((intrinsics_i, torch.zeros(1, 4)), 0)
            intrinsics_i[3, 3] = 1.
            intrinsics_i = torch.unsqueeze(torch.unsqueeze(intrinsics_i, 0), 0)
            intrinsics.append(intrinsics_i)
        return intrinsics

    def get_stamps(self):
        # identifiers of the pictures and corresponding data
        # identifier is in format: SEQ_SUBSEQ_IMAGENUMBER
        # e.g. 2011_09_26_drive_0002_sync_0000000008
        # 2011_09_26_drive_0002_sync_groundtruth_depth_0000000026_image_03.png = format of the depth names
        stamps = dict()
        depth_dir = os.path.join(self.path, "depth")
        depth_files = sorted(glob.glob(depth_dir + "/*.png"))
        for i in range(len(depth_files)):
            depth_file = depth_files[i].replace(depth_dir + "/", "")
            identifier = depth_file[0:26] + depth_file[44:55]     # slice id from depth file name
            stamps[identifier] = i
        return stamps

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass


class Dataset:
    def __init__(self, seq, subseq, depth_path):
        poses_dataset = KITTIRawPoses(seq, subseq)
        self.poses_raw = poses_dataset.poses
        self.stamps_raw = poses_dataset.stamps
        depth_dataset = KITTIDepth(depth_path)
        self.depths = depth_dataset.depths
        self.intrinsics = depth_dataset.intrinsics
        self.rgbs = depth_dataset.rgbs
        self.stams_depth = depth_dataset.stamps
        self.data = self.create_correspondences()

    def create_correspondences(self):
        data = list()   # list of tuples (colors, depth, intrinsics, poses)
        for stamp in self.stamps_raw:
            if stamp in self.stams_depth:
                i = self.stams_depth[stamp]     # index of depth data corresponding to raw data
                # format poses for gradslam
                poses = torch.tensor(self.poses_raw[self.stamps_raw.index(stamp)])
                poses = torch.unsqueeze(torch.unsqueeze(poses, 0), 0)
                d = (self.rgbs[i], self.depths[i], self.intrinsics[i], poses)
                data.append(d)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def poses_demo():
    # np.random.seed(135)
    #seq = np.random.choice(sequence_names)
    seq = "2011_09_26"
    while True:
        #subseq = np.random.choice(os.listdir(os.path.join(data_dir, seq)))
        subseq = "2011_09_26_drive_0002_sync"
        if '2011_' in subseq:
            break
    ds = KITTIRawPoses(seq=seq, subseq=subseq)

    xs, ys, zs = ds.poses[:, 0, 3], ds.poses[:, 1, 3], ds.poses[:, 2, 3]

    plt.figure()
    plt.title("%s/%s" % (seq, subseq))
    # plt.subplot(1, 2, 1)
    plt.plot(xs, ys, '.')
    plt.grid()
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.axis('equal')

    # plt.subplot(1, 2, 2)
    # plt.xlabel('time [sec]')
    # plt.ylabel('Z [m]')
    # plt.plot(ds.ts, zs, '.')
    # plt.grid()
    # plt.axis('equal')
    plt.show()


def ts_demo():
    # np.random.seed(135)
    # seq = np.random.choice(sequence_names)
    seq = "2011_09_26"
    while True:
        subseq = np.random.choice(os.listdir(os.path.join(data_dir, seq)))
        if '2011_' in subseq:
            break
    ds = KITTIRawPoses(seq=seq, subseq=subseq)

    ts_gps = ds.get_timestamps(sensor='gps', zero_origin=True)
    ts_velo = ds.get_timestamps(sensor='lidar', zero_origin=True)

    plt.figure()
    plt.plot(ts_gps[::5], '.', label='gps')
    plt.plot(ts_velo[::5], '.', label='lidar')
    plt.legend()
    plt.grid()
    plt.show()


def gradslam_demo():

    # constructs global map using gradslam, visualizes resulting pointcloud
    seq = "2011_09_26"
    subseq = "2011_09_26_drive_0002_sync"
    # subseq = "2011_09_26_drive_0005_sync"
    # subseq = "2011_09_26_drive_0023_sync"
    print("###### Loading data ... ######")
    dataset = Dataset(seq, subseq, DEPTH_DATA_DIR)
    print("###### Data loaded ######")

    # create global map
    slam = PointFusion(device=DEVICE, odom="gt", dsratio=4)
    prev_frame = None
    pointclouds = Pointclouds(device=DEVICE)
    for s in range(len(dataset)):
        colors, depths, intrinsics, poses, *_ = dataset[s]
        depths = depths.float()
        colors = colors.float()
        intrinsics = intrinsics.float()
        poses = poses.float()
        live_frame = RGBDImages(colors, depths, intrinsics, poses).to(DEVICE)
        pointclouds, *_ = slam.step(pointclouds, live_frame, prev_frame)
        prev_frame = live_frame
        print(f"###### Creating map, step: {s}/{len(dataset) - 1} ######")

    # visualize using open3d
    pc = pointclouds.points_list[0]
    pcd_gt = open3d.geometry.PointCloud()
    pcd_gt.points = open3d.utility.Vector3dVector(pc.cpu().detach().numpy())
    open3d.visualization.draw_geometries([pcd_gt])


def main():
    # for _ in range(3):
    #     poses_demo()
    # ts_demo()
    gradslam_demo()


if __name__ == '__main__':
    main()
