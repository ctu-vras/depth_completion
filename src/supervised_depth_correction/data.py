from __future__ import absolute_import, division, print_function
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
from tqdm import tqdm
import datetime
import torch
from PIL import Image


# TODO: use os and file path instead
RAW_DATA_DIR = "/home/ruslan/data/datasets/kitti_raw"
DEPTH_DATA_DIR = "/home/ruslan/data/datasets/kitti_depth/"
DEPTH_SELECTION_DATA_DIR = "/home/ruslan/data/datasets/kitti_depth/depth_selection/val_selection_cropped"
# RAW_DATA_DIR = "/home/jachym/KITTI/kitti_raw"
# DEPTH_DATA_DIR = "/home/jachym/KITTI/depth_selection/val_selection_cropped"

sequence_names = [
    '2011_09_26',
    '2011_09_28',
    '2011_09_29',
    '2011_09_30',
    '2011_10_03'
]


class KITTIRaw(object):

    def __init__(self, subseq, path=None):
        if path is None:
            seq = subseq[:10]
            path = os.path.join(RAW_DATA_DIR, seq)
        self.path = path
        self.subseq = subseq
        self.gps2cam_transform = self.get_imu2cam_transform()
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

    def get_cam_poses(self, zero_origin=False):
        poses = []
        for fname in np.sort(glob.glob(os.path.join(self.path, self.subseq, 'oxts/data/*.txt'))):
            pose = self.get_gps_pose(fname)
            poses.append(pose)
        poses = np.asarray(poses)

        # TODO: now we have Tr(gps -> cam0), we need Tr(gps->cam2 or cam3)
        poses = np.matmul(poses, self.gps2cam_transform[None])

        if zero_origin:
            # move poses to 0 origin:
            Tr_inv = np.linalg.inv(poses[0])
            poses = np.asarray([np.matmul(Tr_inv, pose) for pose in poses])

        return poses

    def get_imu2cam_transform(self):
        # Load calibration matrices
        TrImuToVelo = self.get_transform("calib_imu_to_velo.txt")
        TrVeloToCam = self.get_transform("calib_velo_to_cam.txt")
        TrImuToCam0 = np.matmul(TrImuToVelo, TrVeloToCam)
        return np.asarray(TrImuToCam0)

    def get_transform(self, file):
        """Read calibration from file.
            :param str file: File name.
            :return numpy.matrix: Calibration.
            """
        fpath = os.path.join(self.path, file)
        with open(fpath, 'r') as f:
            s = f.read()
        i_r = s.index('R:')
        i_t = s.index('T:')
        i_t_end = i_t+2 + s[i_t+2:].index('\n')
        rotation = np.mat(s[i_r + 2:i_t], dtype=np.float64).reshape((3, 3))
        translation = np.mat(s[i_t + 2:i_t_end], dtype=np.float64).reshape((3, 1))
        transform = np.bmat([[rotation, translation], [[[0, 0, 0, 1]]]])
        assert transform.shape == (4, 4)
        return transform

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

    def get_intrinsics(self, cam_id):
        """Read calibration from file.
            :param int cam_id: Camera id: 0, 1, 2, 3.
            :return numpy.matrix: Intrinsics matrix.
        """
        fpath = os.path.join(self.path, "calib_cam_to_cam.txt")
        with open(fpath, 'r') as f:
            s = f.read()
        i_K_start = s.index('K_%02d:' % cam_id)
        i_K_end = i_K_start + 5 + s[i_K_start + 5:].index('\n')
        K = np.mat(s[i_K_start + 5:i_K_end], dtype=np.float64).reshape((3, 3))
        return K

    def get_rgb(self, i, image="image_02"):
        file = os.path.join(self.path, self.subseq, image, "data/%010d.png" % i)
        rgb = np.asarray(Image.open(file))
        return rgb

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        assert i in self.ids
        return self.poses[i]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class KITTIDepth:
    """
    RGB-D data from KITTI depth: http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion
    """
    def __init__(self, subseq=None, mode='val', path=None, camera="left"):
        assert mode == 'train' or mode == 'val'
        if path is None:
            path = os.path.join(DEPTH_DATA_DIR, mode, subseq)
        self.path = path
        self.image = "image_02" if camera == 'left' else "image_03"
        self.ids = self.get_ids()
        self.depths = None
        self.subseq = subseq
        self.raw = KITTIRaw(subseq=subseq)

    def get_depth(self, id, gt=False):
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
            id: int
            gt: bool
        Returns:
            depth: np.array, depth image
        """
        depth_folder = 'groundtruth' if gt else 'velodyne_raw'
        fpath = os.path.join(self.path, 'proj_depth', depth_folder, self.image, '%010d.png' % id)

        depth_png = np.array(Image.open(fpath), dtype=int)
        # make sure we have a proper 16bit depth map here.. not 8bit!
        assert (np.max(depth_png) > 255)
        depth = depth_png.astype(np.float) / 256.
        depth[depth_png == 0] = -1.
        return depth

    def get_rgb(self, i):
        rgb = self.raw.get_rgb(i, image=self.image)
        return rgb

    def get_intrinsics(self, i):
        camera_n = int(self.image[-2:])
        K = self.raw.get_intrinsics(camera_n)
        return K

    def get_ids(self, gt=True):
        ids = list()
        depth_label = "groundtruth" if gt else "velodyne_raw"
        depth_files = sorted(glob.glob(os.path.join(self.path, "proj_depth", depth_label, self.image, "*")))
        for depth_file in depth_files:
            id = int(depth_file[-14:-4])
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


class KITTIDepthSelection(KITTIDepth):

    """
    loads depth images, rgb images and intrinsics from
    KITTI depth selection: http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion
    """

    def __init__(self, subseq, path=None):
        super(KITTIDepth, self).__init__()
        # path directory should contain folders: depth, rgb, intrinsics
        if path is None:
            path = DEPTH_SELECTION_DATA_DIR
        self.path = path
        self.subseq = subseq
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
        r, c = depth.shape[:2]
        # make sure we have a proper 16bit depth map here.. not 8bit!
        assert (np.max(depth) > 255)
        if to_depth_map:
            depth = depth.astype(np.float) / 256.
            depth[depth == 0] = -1.
        return depth.reshape([r, c, 1])

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


class Dataset:
    def __init__(self, subseq, selection=False):
        self.ds_poses = KITTIRaw(subseq=subseq)
        if selection:
            self.ds_depths = KITTIDepthSelection(subseq=subseq)
        else:
            self.ds_depths = KITTIDepth(subseq=subseq)
        self.poses = self.ds_poses.poses
        self.ids = self.ds_depths.ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        """
        Provides input data, that could be used with GradSLAM
        Args:
            item: int

        Returns:
            data: list(colors, depths, intrinsics, poses)
                  colors: torch.Tensor (B x N x W x H x Crgb)
                  depths: torch.Tensor (B x N x W x H x Cd)
                  intrinsics: torch.Tensor (B x N x 4 x 4)
                  poses: torch.Tensor (B x N x 4 x 4)
        """
        assert item in self.ids
        colors, depths, K = self.ds_depths[item]
        poses = self.ds_poses[item]

        intrinsics = np.eye(4)
        intrinsics[:3, :3] = K

        data = [colors, depths, intrinsics, poses]
        data = [torch.as_tensor(d[None][None], dtype=torch.float32) for d in data]
        return data


def poses_demo():
    np.random.seed(135)
    seq = np.random.choice(sequence_names)
    while True:
        subseq = np.random.choice(os.listdir(os.path.join(RAW_DATA_DIR, seq)))
        if '2011_' in subseq:
            break
    # subseq = "2011_09_26_drive_0002_sync"

    ds = KITTIRaw(subseq=subseq)
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
    np.random.seed(135)
    seq = np.random.choice(sequence_names)
    while True:
        subseq = np.random.choice(os.listdir(os.path.join(RAW_DATA_DIR, seq)))
        if '2011_' in subseq:
            break

    ds = KITTIRaw(subseq=subseq)

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
    import open3d as o3d

    # constructs global map using gradslam, visualizes resulting pointcloud
    subseq = "2011_09_26_drive_0002_sync"
    # subseq = "2011_09_26_drive_0005_sync"
    # subseq = "2011_09_26_drive_0023_sync"

    ds = Dataset(subseq)
    device = torch.device('cpu')

    # create global map
    slam = PointFusion(device=device, odom="gt", dsratio=1)
    prev_frame = None
    pointclouds = Pointclouds(device=device)

    for i in ds.ids:
        colors, depths, intrinsics, poses = ds[i]

        live_frame = RGBDImages(colors, depths, intrinsics, poses).to(device)
        pointclouds, live_frame.poses = slam.step(pointclouds, live_frame, prev_frame)

        prev_frame = live_frame

    # visualize using open3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointclouds.points_list[0].cpu().detach().numpy())
    o3d.visualization.draw_geometries([pcd])


def depth_demo():
    import open3d as o3d

    # subseq = "2011_09_26_drive_0002_sync"
    subseq = "2011_09_26_drive_0005_sync"
    # subseq = "2011_09_26_drive_0023_sync"

    ds = Dataset(subseq=subseq)

    all_poses = ds.poses
    depth_poses = ds.poses[ds.ds_depths.ids]

    plt.figure()
    plt.title("%s" % subseq)
    plt.plot(all_poses[:, 0, 3], all_poses[:, 1, 3], '.')
    plt.plot(depth_poses[:, 0, 3], depth_poses[:, 1, 3], 'o')
    plt.grid()
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.axis('equal')
    plt.show()

    global_map = list()
    # using poses convert pcs to one coord frame, create and visualize map
    for i in ds.ids[::5]:
        rgb_img_raw, depth_img_raw, K, pose = ds[i]

        rgb_img_raw = np.asarray(rgb_img_raw.cpu().numpy().squeeze(), dtype=np.uint8)
        depth_img_raw = depth_img_raw.cpu().numpy().squeeze()
        K = K.cpu().numpy().squeeze()
        pose = pose.squeeze().cpu().numpy()

        K = K[:3, :3]
        w, h = rgb_img_raw.shape[:2]

        rgb_img = o3d.geometry.Image(rgb_img_raw)
        depth_img = o3d.geometry.Image(np.asarray(depth_img_raw, dtype=np.uint16))
        rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(color=rgb_img, depth=depth_img)

        intrinsic = o3d.camera.PinholeCameraIntrinsic(width=w, height=h,
                                                      fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2])

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd_img, intrinsic=intrinsic)
        # pcd = o3d.geometry.PointCloud.create_from_depth_image(depth=depth_img, intrinsic=intrinsic)

        pcd.transform(pose)

        # Flip it, otherwise the pointcloud will be upside down
        # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        # o3d.visualization.draw_geometries([pcd])

        global_map.append(pcd)

    o3d.visualization.draw_geometries(global_map)


def pykitti_demo():
    import pykitti

    # subseq = "2011_09_26_drive_0002_sync"
    seq = "2011_09_26"
    drive = subseq = "0002"

    # The 'frames' argument is optional - default: None, which loads the whole dataset.
    # Calibration, timestamps, and IMU data are read automatically.
    # Camera and velodyne data are available via properties that create generators
    # when accessed, or through getter methods that provide random access.
    data = pykitti.raw(RAW_DATA_DIR, seq, drive, frames=range(0, 50, 5))

    # dataset.calib:         Calibration data are accessible as a named tuple
    # dataset.timestamps:    Timestamps are parsed into a list of datetime objects
    # dataset.oxts:          List of OXTS packets and 6-dof poses as named tuples
    # dataset.camN:          Returns a generator that loads individual images from camera N
    # dataset.get_camN(idx): Returns the image from camera N at idx
    # dataset.gray:          Returns a generator that loads monochrome stereo pairs (cam0, cam1)
    # dataset.get_gray(idx): Returns the monochrome stereo pair at idx
    # dataset.rgb:           Returns a generator that loads RGB stereo pairs (cam2, cam3)
    # dataset.get_rgb(idx):  Returns the RGB stereo pair at idx
    # dataset.velo:          Returns a generator that loads velodyne scans as [x,y,z,reflectance]
    # dataset.get_velo(idx): Returns the velodyne scan at idx

    point_velo = np.array([0, 0, 0, 1])
    point_cam0 = data.calib.T_cam0_velo.dot(point_velo)

    point_imu = np.array([0, 0, 0, 1])
    point_w = [o.T_w_imu.dot(point_imu) for o in data.oxts]

    for cam0_image in data.cam0:
        # do something
        pass

    cam2_image, cam3_image = data.get_rgb(3)


def main():
    # poses_demo()
    # ts_demo()
    # gradslam_demo()
    # pykitti_demo()
    depth_demo()


if __name__ == '__main__':
    main()
