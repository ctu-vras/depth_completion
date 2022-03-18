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


# TODO: use os and file path instead
data_dir = "/home/ruslan/data/datasets/kitti_raw"

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
    def __init__(self):
        self.ts = None
        self.depths = None

    def get_rgbd(self):
        pass

    def get_timestamps(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass


class Dataset:
    def __init__(self, seq, subseq):
        self.poses_raw = KITTIRawPoses(seq, subseq).poses
        self.depths = KITTIDepth().depths

    def get_timestamps(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass


def poses_demo():
    # np.random.seed(135)
    seq = np.random.choice(sequence_names)
    subseq = np.random.choice(glob.glob(os.path.join(data_dir, seq, '2011_*')))
    ds = KITTIRawPoses(seq=seq, subseq=subseq)

    xs, ys, zs = ds.poses[:, 0, 3], ds.poses[:, 1, 3], ds.poses[:, 2, 3]

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.title('X, Y [m]')
    plt.plot(xs, ys, '.')
    plt.grid()
    plt.axis('equal')

    plt.subplot(1, 2, 2)
    plt.title('Z [m]')
    plt.plot(ds.ts, zs, '.')
    plt.grid()
    plt.axis('equal')
    plt.show()


def ts_demo():
    # np.random.seed(135)
    seq = np.random.choice(sequence_names)
    subseq = np.random.choice(glob.glob(os.path.join(data_dir, seq, '2011_*')))
    ds = KITTIRawPoses(seq=seq, subseq=subseq)

    ts_gps = ds.get_timestamps(sensor='gps', zero_origin=True)
    ts_velo = ds.get_timestamps(sensor='lidar', zero_origin=True)

    plt.figure()
    plt.plot(ts_gps[::5], '.', label='gps')
    plt.plot(ts_velo[::5], '.', label='lidar')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    for _ in range(5):
        poses_demo()
    # ts_demo()
