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


# TODO: use os and file path instead
data_dir = "/home/ruslan/data/datasets/kitti_raw"

sequence_names = [
    '2011_09_26',
    '2011_09_28',
    '2011_09_29',
    '2011_09_30',
    '2011_10_03'
]


def skew_sim(a):
    A = np.array([[0, -a[2], a[1]],
                  [a[2], 0, -a[0]],
                  [-a[1], a[0], 0]])
    return A


def angle_axis_to_R(theta, a):
    R = np.cos(theta)*np.eye(3) + (1 - np.cos(theta))*np.matmul(a.reshape([3, 1]), a.reshape([1, 3])) +\
        np.sin(theta)*skew_sim(a)
    return R


class KITTIRaw(object):

    def __init__(self, seq, subseq, path=None):
        if path is None:
            path = os.path.join(data_dir, seq, subseq)
        self.name = os.path.join(seq, subseq)
        self.path = path

    @staticmethod
    def get_gps_lidar_transform():
        pass

    def local_cloud(self, id):
        pass

    def visualize_cloud(self, input):
        import open3d as o3d
        if isinstance(input, int):
            id = input
            cloud = self.local_cloud(id)[:, :3]
        elif isinstance(input, np.ndarray):
            cloud = input[:, :3]
        else:
            raise ValueError('Support np.array ot int index values')
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud)
        o3d.visualization.draw_geometries([pcd])

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

    def get_gps_pose(self, id):
        gps_data = np.genfromtxt(os.path.join(self.path, 'oxts/data/', '%010d.txt' % id))
        lat, lon, alt = gps_data[:3]
        roll, pitch, yaw = gps_data[3:6]
        R = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=False).as_matrix()
        x, y, z = self.gps_to_ecef(lat, lon, alt)
        # convert to 4x4 matrix
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = np.array([x, y, z])
        return pose

    def get_cloud_poses(self):
        pass

    def cloud_pose(self, id):
        return self.poses[id]

    def get_timestamps(self, sensor, zero_origin=False):
        assert isinstance(sensor, str)
        assert sensor == 'imu' or sensor == 'gps-latlong'
        timestamps = []
        not_strictly_increasing_idxs = []
        j = 0
        for i in self.parts_range:
            part = 'Part%d' % i
            files = np.genfromtxt(os.path.join(self.path, part, '%s.txt' % sensor), dtype=str)[:, 0].tolist()
            for t in files:
                t = np.asarray(t.split("_"), dtype=np.float)
                timestamp = t[3] * 3600 + t[4] * 60 + t[5] + t[6] / 1000.
                timestamps.append(timestamp)
                # timestamps should be in increasing order for poses SLERP, that is why we keep non strictly
                # increasing indexes
                if j > 1:
                    if timestamps[j] - timestamps[j-1] <= 0:
                        not_strictly_increasing_idxs.append(j)
                j += 1

        timestamps = np.sort(np.asarray(timestamps))
        if zero_origin:
            timestamps = timestamps - timestamps[0]
        return timestamps, not_strictly_increasing_idxs

    def get_global_cloud(self, pts_step=1, poses_step=10):
        clouds = []
        for id in tqdm(range(0, len(self), poses_step)):
            cloud = self.local_cloud(id)[:, :3]
            pose = self.cloud_pose(id)
            cloud = np.matmul(cloud, pose[:3, :3]) + pose[:3, 3:].reshape([1, 3])
            clouds.append(cloud)
            # print('%i points read from dataset %s, cloud %i.' % (cloud.shape[0], ds.name, id))
        cloud = np.concatenate(clouds)[::pts_step, :]
        return cloud

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        if isinstance(item, int):
            id = self.ids[item]
            return self.local_cloud(id), self.cloud_pose(id)

        ds = KITTIRaw()
        ds.name = self.name
        ds.path = self.path
        ds.poses = self.poses
        if isinstance(item, (list, tuple)):
            ds.ids = [self.ids[i] for i in item]
        else:
            assert isinstance(item, slice)
            ds.ids = self.ids[item]
        return ds

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def demo():
    seq = sequence_names[0]
    subseq = np.random.choice(glob.glob(os.path.join(data_dir, seq, '2011_*')))
    ds = KITTIRaw(seq=seq, subseq=subseq)

    poses = []
    for i, fname in enumerate(glob.glob(os.path.join(ds.path, 'oxts/data/*.txt'))):
        pose = ds.get_gps_pose(i)
        poses.append(pose)
    poses = np.asarray(poses)

    assert poses.shape == (len(poses), 4, 4)
    # zero_origin:
    xs, ys, zs = poses[:, 0, 3], poses[:, 1, 3], poses[:, 2, 3]
    xs, ys, zs = xs - xs[0], ys - ys[0], zs - zs[0]

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.title('X, Y [m]')
    plt.plot(xs, ys, '.')
    plt.grid()
    plt.axis('equal')

    plt.subplot(1, 2, 2)
    plt.title('Z [m]')
    plt.plot(zs, '.')
    plt.grid()
    # plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    demo()
