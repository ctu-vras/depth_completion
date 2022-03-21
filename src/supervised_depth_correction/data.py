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
from PIL import Image


# TODO: use os and file path instead
data_dir_raw = "/home/ruslan/data/datasets/kitti_raw"
data_dir_depth = "/home/ruslan/data/datasets/kitti_depth"

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
            path = os.path.join(data_dir_raw, seq)
        self.path = path
        self.subseq = subseq
        self.poses = self.get_poses()
        self.ts = self.get_timestamps(sensor='gps')
        self.calibrations = self.load_calib()
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
        for fname in np.sort(glob.glob(os.path.join(self.path, self.subseq, 'oxts/data/*.txt'))):
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
        ts = np.genfromtxt(os.path.join(self.path, self.subseq, folder, 'timestamps.txt'), dtype=str)
        for t in ts:
            date = t[0]
            day_time, sec = t[1].split(".")
            sec = float('0.' + sec)
            stamp = datetime.datetime.strptime("%s_%s" % (date, day_time), "%Y-%m-%d_%H:%M:%S").timestamp() + sec
            timestamps.append(stamp)
        if zero_origin:
            timestamps = [t - timestamps[0] for t in timestamps]
        return timestamps

    def load_calib(self):
        """
        load calib poses and times.
        """
        calib = self.parse_calibration(os.path.join(self.path, "calib_cam_to_cam.txt"))
        return calib

    @staticmethod
    def parse_calibration(filename):
        """ read calibration file with given filename

            Returns
            -------
            dict
                Calibration matrices as 4x4 numpy arrays.
        """
        calib = {}
        calib_file = open(filename)
        for line in calib_file:
            # load intrinsics
            if 'K_' in line:
                key, content = line.split(':')
                values = [float(v) for v in content.strip().split()]

                K = np.zeros((3, 3))
                K[0, 0:3] = values[0:3]
                K[1, 0:3] = values[3:6]
                K[2, 0:3] = values[6:9]

                calib[key] = K
        calib_file.close()
        return calib

    def get_intrins(self):
        pass

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.ts[i], self.poses[i]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class KITTIDepth:
    # TODO: add rgb-d data from KITTI depth: http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion
    def __init__(self, seq, mode='val', path=None):
        assert mode == 'train' or mode == 'val'
        if path is None:
            path = os.path.join(data_dir_depth, mode, seq)
        self.path = path
        self.ts = None
        self.depths = None
        self.name = seq

    @staticmethod
    def read_depth(filename):
        """
        Depth maps (annotated and raw Velodyne scans) are saved as uint16 PNG images,
        which can be opened with either MATLAB, libpng++ or the latest version of
        Python's pillow (from PIL import Image). A 0 value indicates an invalid pixel
        (ie, no ground truth exists, or the estimation algorithm didn't produce an
        estimate for that pixel). Otherwise, the depth for a pixel can be computed
        in meters by converting the uint16 value to float and dividing it by 256.0:

        disp(u,v)  = ((float)I(u,v))/256.0;
        valid(u,v) = I(u,v)>0;
        Args:
            filename: str, full path to depth image

        Returns:
            depth: np.array, depth image
        """
        depth_png = np.array(Image.open(filename), dtype=int)
        # make sure we have a proper 16bit depth map here.. not 8bit!
        assert (np.max(depth_png) > 255)
        depth = depth_png.astype(np.float) / 256.
        depth[depth_png == 0] = -1.
        return depth

    def get_depth_map(self, id, gt=False, cam='left'):
        depth_folder = 'groundtruth' if gt else 'velodyne_raw'
        camera_n = 2 if cam == 'left' else 3
        fpath = os.path.join(self.path, 'proj_depth', depth_folder, 'image_%02d/%010d.png' % (camera_n, id))
        depth_map = self.read_depth(fpath)
        return depth_map

    def get_timestamps(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass


class Dataset:
    # TODO: merge corresponnding rgb-d (from KITTI Depth) and poses (from KITTI Raw), construct global map
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
    while True:
        subseq = np.random.choice(os.listdir(os.path.join(data_dir_raw, seq)))
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


def gps_velo_times_demo():
    # np.random.seed(135)
    seq = np.random.choice(sequence_names)
    while True:
        subseq = np.random.choice(os.listdir(os.path.join(data_dir_raw, seq)))
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


def depths_demo():
    seq = '2011_09_26_drive_0002_sync'
    ds = KITTIDepth(seq=seq)
    ds.get_depth_map(5)


def main():
    for _ in range(3):
        poses_demo()
    gps_velo_times_demo()
    depths_demo()


if __name__ == '__main__':
    main()
