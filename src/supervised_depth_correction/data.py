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
import open3d as o3d
from PIL import Image


# TODO: use os and file path instead
DATA_DIR = "/home/ruslan/data/datasets/kitti_raw"
# DATA_DIR = "/home/jachym/KITTI/kitti_raw"
# DEPTH_DATA_DIR = "/home/jachym/KITTI/depth_selection/val_selection_cropped"
DEPTH_DATA_DIR = "/home/ruslan/data/datasets/kitti_depth/depth_selection/val_selection_cropped"
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

sequence_names = [
    '2011_09_26',
    '2011_09_28',
    '2011_09_29',
    '2011_09_30',
    '2011_10_03'
]


class KITTIRawPoses(object):

    def __init__(self, subseq, path=None):
        if path is None:
            seq = subseq[:10]
            path = os.path.join(DATA_DIR, seq)
        self.path = path
        self.subseq = subseq
        self.gps2cam_transform = self.get_calibrations()
        self.poses = self.get_cam_poses()
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

    def get_cam_poses(self, zero_origin=True):
        poses = []
        for fname in np.sort(glob.glob(os.path.join(self.path, self.subseq, 'oxts/data/*.txt'))):
            pose = self.get_gps_pose(fname)
            poses.append(pose)
        poses = np.asarray(poses)
        if zero_origin:
            pose0 = poses[0]
            poses = np.matmul(np.linalg.inv(pose0)[None], poses)
        # TODO: now we have Tr(gps -> cam0), we need Tr(gps->cam2 or cam3)
        poses = np.matmul(poses, self.gps2cam_transform[None])
        return poses

    def get_calibrations(self):
        # Load calibration matrices
        cal1 = self.load_calib_file("calib_imu_to_velo.txt")
        cal2 = self.load_calib_file("calib_velo_to_cam.txt")
        cal12 = np.matmul(cal1, cal2)
        return cal12

    def load_calib_file(self, file):
        with open(os.path.join(self.path, file)) as f:
            f.readline()
            line2 = f.readline().rstrip().split()
            line2.pop(0)
            R = np.array(line2).astype(float).reshape((3, 3))
            line3 = f.readline().rstrip().split()
            line3.pop(0)
            t = np.array(line3).astype(float).reshape((3, 1))
            temp = np.zeros((1, 4), dtype=float)
            temp[0, 3] = 1.0
            cal = np.concatenate((R, t), 1)
            cal = np.concatenate((cal, temp), 0)
        return cal

    def get_timestamps(self, sensor='gps', zero_origin=False):
        assert isinstance(sensor, str)
        assert sensor == 'gps' or sensor == 'lidar'
        if sensor == 'gps':
            sensor_folder = 'oxts'
        elif sensor == 'lidar':
            sensor_folder = 'velodyne_points'
        else:
            raise ValueError
        timestamps = []
        ts = np.genfromtxt(os.path.join(self.path, self.subseq, sensor_folder, 'timestamps.txt'), dtype=str)
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
        assert i in self.ids
        return self.ts[i], self.poses[i]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class KITTIDepthSelection:

    """
    loads depth images, rgb images and intrinsics from
    KITTI depth: http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion
    """

    def __init__(self, subseq, path=None, camera="left"):
        # path directory should contain folders: depth, rgb, intrinsics
        if path is None:
            path = DEPTH_DATA_DIR
        self.path = path
        self.subseq = subseq
        self.image = "image_02" if camera == 'left' else "image_03"
        self.ids = self.get_ids()

    def get_rgb(self, i):
        file = os.path.join(self.path, "image", "%s_image_%010d_%s.png" % (self.subseq, i, self.image))
        rgb = np.asarray(Image.open(file))
        return rgb

    def get_depth(self, i, gt=True, to_depth_map=False):
        """
        Depth maps (annotated and raw Velodyne scans) are saved as uint16 PNG images,
        which can be opened with either MATLAB, libpng++ or the latest version of
        Python's pillow (from PIL import Image). A 0 value indicates an invalid pixel
        (ie, no ground truth exists, or the estimation algorithm didn't produce an
        estimate for that pixel). Otherwise, the depth for a pixel can be computed
        in meters by converting the uint16 value to float and dividing it by 256.0:
        disp(u,v)  = ((float)I(u,v))/256.0;
        valid(u,v) = I(u,v)>0;
        """
        depth_label = "groundtruth_depth" if gt else "velodyne_raw"
        file = os.path.join(self.path, depth_label, "%s_%s_%010d_%s.png" % (self.subseq, depth_label, i, self.image))
        depth = np.array(Image.open(file), dtype=int)
        # make sure we have a proper 16bit depth map here.. not 8bit!
        assert (np.max(depth) > 255)
        if to_depth_map:
            depth = depth.astype(np.float) / 256.
            depth[depth == 0] = -1.
        return depth

    def get_intrinsics(self, i):
        file = os.path.join(self.path, "intrinsics", "%s_image_%010d_%s.txt" % (self.subseq, i, self.image))
        K = np.loadtxt(file).reshape(3, 3)
        return K

    def get_ids(self, gt=True):
        ids = list()
        depth_label = "groundtruth_depth" if gt else "velodyne_raw"
        depth_files = sorted(glob.glob(os.path.join(self.path, depth_label,
                                                    "%s_%s_*_%s.png" % (self.subseq, depth_label, self.image))))
        for depth_file in depth_files:
            id = int(depth_file[-23:-13])
            ids.append(id)
        return ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        assert i in self.ids
        intrins = self.get_intrinsics(i)
        rgb = self.get_rgb(i)
        depth = self.get_depth(i)
        return rgb, depth, intrins


class Dataset:
    def __init__(self, seq, subseq, depth_path):
        poses_dataset = KITTIRawPoses(seq, subseq)
        self.poses_raw = poses_dataset.poses
        self.stamps_raw = poses_dataset.stamps
        depth_dataset = KITTIDepthSelection(depth_path)
        self.depths = depth_dataset.depths
        self.detph_files = depth_dataset.depth_files
        self.intrinsics = depth_dataset.intrinsics
        self.intrinsics_files = depth_dataset.intrinsics_files
        self.rgbs = depth_dataset.rgbs
        self.rgb_files = depth_dataset.rgb_files
        self.stams_depth = depth_dataset.stamps

        self.data = self.create_correspondences()
        self.data_o3d = self.get_corresp_o3d()

    def create_correspondences(self):
        data = list()   # list of tuples (colors, depth, intrinsics, poses)
        for stamp in self.stamps_raw:
            if stamp in self.stams_depth:
                i = self.stams_depth[stamp]     # index of depth data corresponding to raw data
                # format poses for gradslam
                pose = torch.tensor(self.poses_raw[self.stamps_raw.index(stamp)])
                pose = torch.unsqueeze(torch.unsqueeze(pose, 0), 0)
                d = (self.rgbs[i], self.depths[i], self.intrinsics[i], pose)
                data.append(d)
        return data

    def get_corresp_o3d(self):
        # returns list of correspondences as open3d objects and poses
        data = list()  # list of tuples (colors_files, depth_files, intrinsics_files, poses)
        for stamp in self.stamps_raw:
            if stamp in self.stams_depth:
                i = self.stams_depth[stamp]     # index of depth data corresponding to raw data
                # format poses for gradslam
                pose = self.poses_raw[self.stamps_raw.index(stamp)]
                depth_raw = open3d.io.read_image(self.detph_files[i])
                color_raw = open3d.io.read_image(self.rgb_files[i])
                rgbd_image = open3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)
                w, h = np.asarray(rgbd_image.color).shape
                K = np.loadtxt(self.intrinsics_files[i]).reshape(3, 3)
                intrinsics = open3d.camera.PinholeCameraIntrinsic(width=w, height=h,
                                                                  fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2])
                d = (rgbd_image, intrinsics, pose)
                data.append(d)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def poses_demo():
    np.random.seed(135)
    seq = np.random.choice(sequence_names)
    while True:
        subseq = np.random.choice(os.listdir(os.path.join(DATA_DIR, seq)))
        if '2011_' in subseq:
            break

    # subseq = "2011_09_26_drive_0002_sync"

    ds = KITTIRawPoses(subseq=subseq)
    xs, ys, zs = ds.poses[:, 0, 3], ds.poses[:, 1, 3], ds.poses[:, 2, 3]

    plt.figure()
    plt.title("%s" % subseq)
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
    seq = np.random.choice(sequence_names)
    while True:
        subseq = np.random.choice(os.listdir(os.path.join(DATA_DIR, seq)))
        if '2011_' in subseq:
            break

    ds = KITTIRawPoses(subseq=subseq)

    ts_gps = ds.get_timestamps(sensor='gps', zero_origin=True)
    ts_velo = ds.get_timestamps(sensor='lidar', zero_origin=True)

    plt.figure()
    plt.title("%s" % subseq)
    plt.plot(ts_gps[::5], '.', label='gps')
    plt.plot(ts_velo[::5], '.', label='lidar')
    plt.legend()
    plt.grid()
    plt.show()


def gradslam_demo():
    from gradslam import Pointclouds, RGBDImages
    from gradslam.slam import PointFusion

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
    global_map = Pointclouds(device=DEVICE)
    for s in range(0, len(dataset), 5):
        colors, depths, intrinsics, poses, *_ = dataset[s]
        depths = depths.float()
        colors = colors.float()
        intrinsics = intrinsics.float()
        poses = poses.float()
        live_frame = RGBDImages(colors, depths, intrinsics, poses).to(DEVICE)
        pointclouds, *_ = slam.step(pointclouds, live_frame, prev_frame)
        # pointclouds, *_ = slam(live_frame)
        prev_frame = live_frame
        # global_map.append_points(pointclouds)
        print(f"###### Creating map, step: {s}/{len(dataset) - 1} ######")

    # visualize using open3d
    pc = pointclouds.points_list[0]
    pcd_gt = open3d.geometry.PointCloud()
    pcd_gt.points = open3d.utility.Vector3dVector(pc.cpu().detach().numpy())
    open3d.visualization.draw_geometries([pcd_gt])


def demo():
    subseq = "2011_09_26_drive_0002_sync"
    # subseq = "2011_09_26_drive_0005_sync"
    # subseq = "2011_09_26_drive_0023_sync"

    ds_depth = KITTIDepthSelection(subseq=subseq)
    ds_poses = KITTIRawPoses(subseq=subseq)

    poses = ds_poses.poses
    depth_poses = poses[ds_depth.ids]

    plt.figure()
    plt.title("%s" % subseq)
    plt.plot(poses[:, 0, 3], poses[:, 1, 3], '.')
    plt.plot(depth_poses[:, 0, 3], depth_poses[:, 1, 3], 'o')
    plt.grid()
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.axis('equal')
    plt.show()

    global_map = list()
    # using poses convert pcs to one coord frame, create and visualize map
    for i in ds_depth.ids:
        rgb_img_raw, depth_img_raw, K = ds_depth[i]

        rgb_img = o3d.geometry.Image(rgb_img_raw)
        depth_img = o3d.geometry.Image(np.asarray(depth_img_raw, dtype=np.uint16))
        rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(color=rgb_img, depth=depth_img)

        w, h = rgb_img_raw.shape[:2]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width=w, height=h,
                                                      fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2])

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd_img, intrinsic=intrinsic)

        # Flip it, otherwise the pointcloud will be upside down
        # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        # o3d.visualization.draw_geometries([pcd])

        _, pose = ds_poses[i]

        pcd.transform(pose)
        global_map.append(pcd)

    o3d.visualization.draw_geometries(global_map)


def main():
    # poses_demo()
    # ts_demo()
    # gradslam_demo()
    demo()


if __name__ == '__main__':
    main()
