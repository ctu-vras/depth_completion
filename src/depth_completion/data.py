from __future__ import absolute_import, division, print_function
import os
import glob

import cv2.cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image
import pykitti
from depth_completion.io import write, append


# the following paths assume that you have the datasets or symlinks in depth_completion/data folder
RAW_DATA_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'KITTI', 'raw'))
DEPTH_DATA_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'KITTI', 'depth'))
DEPTH_SELECTION_DATA_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', 'data',
                                                         'KITTI', 'depth', 'depth_selection', 'val_selection_cropped'))

sequence_names = [
    '2011_09_26',
    '2011_09_28',
    '2011_09_29',
    '2011_09_30',
    '2011_10_03'
]


class KITTIRaw(object):

    def __init__(self, subseq):
        self.data = pykitti.raw(RAW_DATA_DIR, date=subseq[:10], drive=subseq[-9:-5])
        self.poses = self.get_cam_poses()
        self.ts = self.get_timestamps()
        self.ids = range(len(self.ts))

    def get_cam_poses(self, zero_origin=False):
        # poses of GPS in world frame
        poses = np.asarray([o.T_w_imu for o in self.data.oxts])
        # we need poses of camera (2: left or 3: right) in world frame
        poses = np.matmul(poses, np.linalg.inv(self.data.calib.T_cam2_imu[None]))
        if zero_origin:
            # move poses to 0 origin:
            Tr_inv = np.linalg.inv(poses[0])
            poses = np.asarray([np.matmul(Tr_inv, pose) for pose in poses])
        return poses

    def get_timestamps(self, zero_origin=False):
        timestamps = [t.timestamp() for t in self.data.timestamps]
        if zero_origin:
            timestamps = [t - timestamps[0] for t in timestamps]
        return timestamps

    def get_intrinsics(self, cam_id):
        """Read calibration from file.
            :param int cam_id: Camera id: 0, 1, 2, 3.
            :return numpy.matrix: Intrinsics matrix.
        """
        if cam_id == 0:
            K = self.data.calib.K_cam0
        elif cam_id == 1:
            K = self.data.calib.K_cam1
        elif cam_id == 2:
            K = self.data.calib.K_cam2
        elif cam_id == 3:
            K = self.data.calib.K_cam3
        else:
            raise ValueError('Not supported camera id')
        return K

    def get_rgb(self, i, image="image_02"):
        # file = os.path.join(self.path, self.subseq, image, "data/%010d.png" % i)
        # rgb = np.asarray(Image.open(file))
        left, right = self.data.get_rgb(i)
        if image == "image_02":
            rgb = left
        elif image == "image_03":
            rgb = right
        else:
            raise ValueError("Not support camera index (should be image_02 or image_03 for left or right camera)")
        return np.asarray(rgb)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.poses[self.ids[i]]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class KITTIDepth:
    """
    RGB-D data from KITTI depth: http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion
    """
    def __init__(self, subseq=None, mode='train', path=None, camera="left", depth_type="sparse"):
        assert mode == 'train' or mode == 'val'
        assert depth_type == "sparse" or depth_type == "dense" or depth_type == "pred"
        if path is None:
            path = os.path.join(DEPTH_DATA_DIR, mode, subseq)
        self.path = path
        self.image = "image_02" if camera == 'left' else "image_03"
        self.depth_type = depth_type
        self.subseq = subseq
        self.raw = KITTIRaw(subseq=subseq)
        self.ids = self.get_ids()

    def get_depth(self, id):
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
        Returns:
            depth: np.array, depth image
        """
        if self.depth_type == "dense":
            depth_folder = 'groundtruth'
        elif self.depth_type == "sparse":
            depth_folder = 'velodyne_raw'
        else:
            depth_folder = 'prediction'
        fpath = os.path.join(self.path, 'proj_depth', depth_folder, self.image, '%010d.png' % id)

        depth_png = np.array(Image.open(fpath), dtype=int)
        # make sure we have a proper 16bit depth map here.. not 8bit!
        assert (np.max(depth_png) > 255)
        depth = depth_png.astype(float) / 256.
        depth[depth_png == 0] = -1.
        r, c = depth.shape[:2]
        return depth.reshape([r, c, 1])

    def get_rgb(self, i):
        rgb = self.raw.get_rgb(i, image=self.image)
        return rgb

    def get_intrinsics(self, i):
        camera_n = int(self.image[-2:])
        K = self.raw.get_intrinsics(camera_n)
        return K

    def get_ids(self):
        ids = list()
        if self.depth_type == "dense":
            depth_label = 'groundtruth'
        elif self.depth_type == "sparse":
            depth_label = 'velodyne_raw'
        else:
            depth_label = 'prediction'
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

    def __init__(self, subseq, path=None, depth_type="raw", camera='left'):
        KITTIDepth.__init__(self, subseq=subseq, path=path, depth_type=depth_type, camera=camera)
        # path directory should contain folders: depth, rgb, intrinsics
        if path is None:
            path = DEPTH_SELECTION_DATA_DIR
        self.path = path
        self.depth_type = depth_type
        self.subseq = subseq
        self.ids = self.get_ids()

    def get_rgb(self, i):
        file = os.path.join(self.path, "image", "%s_image_%010d_%s.png" % (self.subseq, i, self.image))
        rgb = np.asarray(Image.open(file))
        return rgb

    def get_depth(self, i, to_depth_map=False):
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
        if self.depth_type == "dense":
            depth_folder = 'groundtruth'
        elif self.depth_type == "sparse":
            depth_folder = 'velodyne_raw'
        else:
            depth_folder = 'prediction'
        file = os.path.join(self.path, depth_folder, "%s_%s_%010d_%s.png" % (self.subseq, depth_folder, i, self.image))
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

    def get_ids(self):
        ids = list()
        if self.depth_type == "dense":
            depth_label = 'groundtruth'
        elif self.depth_type == "sparse":
            depth_label = 'velodyne_raw'
        else:
            depth_label = 'prediction'
        depth_files = sorted(glob.glob(os.path.join(self.path, depth_label,
                                                    "%s_%s_*_%s.png" % (self.subseq, depth_label, self.image))))
        for depth_file in depth_files:
            id = int(depth_file[-23:-13])
            ids.append(id)
        return ids


class Dataset:
    def __init__(self, subseq,
                 selection=False,
                 depth_type="sparse",
                 depth_set='train',
                 camera='left',
                 zero_origin=True,
                 device=torch.device('cpu')):
        self.ds_poses = KITTIRaw(subseq=subseq)
        if selection:
            self.ds_depths = KITTIDepthSelection(subseq=subseq, depth_type=depth_type, camera=camera)
        else:
            self.ds_depths = KITTIDepth(subseq=subseq, depth_type=depth_type, mode=depth_set, camera=camera)
        self.poses = self.ds_poses.poses
        self.ids = self.ds_depths.ids

        # move poses to origin to 0:
        if zero_origin:
            Tr_inv = np.linalg.inv(self.poses[0])
            self.poses = np.asarray([np.matmul(Tr_inv, pose) for pose in self.poses])
        self.device = device

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        """
        Provides input data, that could be used with GradSLAM
        Args:
            i: int

        Returns:
            data: list(colors, depths, intrinsics, poses)
                  colors: torch.Tensor (B x N x W x H x Crgb)
                  depths: torch.Tensor (B x N x W x H x Cd)
                  intrinsics: torch.Tensor (B x N x 4 x 4)
                  poses: torch.Tensor (B x N x 4 x 4)
        """
        colors, depths, K = self.ds_depths[self.ids[i]]

        # for p1, p2 in zip(self.poses, self.ds_poses.poses):
        #     assert np.allclose(p1, p2)
        poses = self.poses[i]

        intrinsics = np.eye(4)
        intrinsics[:3, :3] = K

        data = [colors, depths, intrinsics, poses]
        data = [torch.as_tensor(d[None][None], dtype=torch.float32).to(self.device) for d in data]
        return data

    def get_gt_poses(self):
        return torch.as_tensor(self.poses[None][None], dtype=torch.float32).to(self.device)


def poses_demo():
    # np.random.seed(135)
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
    plt.subplot(1, 2, 1)
    plt.plot(xs, ys, '.')
    plt.grid()
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.axis('equal')

    plt.subplot(1, 2, 2)
    plt.xlabel('time [sec]')
    plt.ylabel('Z [m]')
    plt.plot(ds.ts, zs, '.')
    plt.grid()
    plt.axis('equal')
    plt.show()


def ts_demo():
    # np.random.seed(135)
    seq = np.random.choice(sequence_names)
    while True:
        subseq = np.random.choice(os.listdir(os.path.join(RAW_DATA_DIR, seq)))
        if '2011_' in subseq:
            break

    ds = KITTIRaw(subseq=subseq)

    ts_gps = ds.get_timestamps(zero_origin=True)

    plt.figure()
    plt.title("%s" % subseq)
    plt.plot(ts_gps[::5], '.', label='gps')
    plt.legend()
    plt.grid()
    plt.show()


def depth_demo():
    import open3d as o3d

    subseqs = [
        "2011_09_26_drive_0001_sync",
        "2011_09_26_drive_0009_sync",
        "2011_09_26_drive_0011_sync"
    ]
    subseq = np.random.choice(subseqs, 1)[0]

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
    for i in tqdm(range(0, len(ds), 5)):
        rgb_img_raw, depth_img_raw, K, pose = ds[i]

        rgb_img_raw = np.asarray(rgb_img_raw.cpu().numpy().squeeze(), dtype=np.uint8)
        depth_img_raw = depth_img_raw.cpu().numpy().squeeze()
        K = K.cpu().numpy().squeeze()
        pose = pose.squeeze().cpu().numpy()

        K = K[:3, :3]
        w, h = rgb_img_raw.shape[:2]

        rgb_img = o3d.geometry.Image(rgb_img_raw)
        # TODO: correct the depth image creation (depth map is most likely in meters?)
        depth_img = o3d.geometry.Image(np.asarray(depth_img_raw, dtype=np.uint16))
        rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(color=rgb_img, depth=depth_img)

        intrinsic = o3d.camera.PinholeCameraIntrinsic(width=w, height=h,
                                                      fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2])

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd_img, intrinsic=intrinsic)
        # pcd = o3d.geometry.PointCloud.create_from_depth_image(depth=depth_img, intrinsic=intrinsic)

        pcd.transform(pose)

        global_map.append(pcd)

    o3d.visualization.draw_geometries(global_map)


def gradslam_demo():
    from gradslam import Pointclouds, RGBDImages
    from gradslam.slam import PointFusion
    import open3d as o3d

    # constructs global map using gradslam
    subseqs = [
               "2011_09_26_drive_0001_sync",
               "2011_09_26_drive_0009_sync",
               "2011_09_26_drive_0011_sync",
               "2011_09_26_drive_0018_sync",
               "2011_09_30_drive_0027_sync"
               ]
    # np.random.seed(135)
    subseq = np.random.choice(subseqs, 1)[0]
    # subseq = subseqs[4]
    print('Sequence name: %s' % subseq)

    ds = Dataset(subseq, depth_type='dense', depth_set='train', zero_origin=False)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    pose_provider = 'gt'
    assert pose_provider == 'gt' or pose_provider == 'icp' or pose_provider == 'gradicp'

    # create global map
    slam = PointFusion(device=device, odom=pose_provider, dsratio=4)
    prev_frame = None
    pointclouds = Pointclouds(device=device)

    for i in tqdm(range(0, len(ds), 5)):
        colors, depths, intrinsics, poses = ds[i]

        live_frame = RGBDImages(colors, depths, intrinsics, poses).to(device)
        pointclouds, live_frame.poses = slam.step(pointclouds, live_frame, prev_frame)

        prev_frame = live_frame

        # # write slam poses
        # slam_pose = live_frame.poses.detach().squeeze()
        # append(os.path.join(os.path.dirname(__file__), '..', '..', 'config/results/', 'slam_poses.txt'),
        #        ', '.join(['%.6f' % x for x in slam_pose.flatten()]) + '\n')

    # visualize using open3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointclouds.points_list[0].detach().cpu())
    o3d.visualization.draw_geometries([pcd.voxel_down_sample(voxel_size=0.5)])


def main():
    # for _ in range(6):
    #     poses_demo()
    # for _ in range(5):
    #     ts_demo()
    gradslam_demo()
    # depth_demo()


if __name__ == '__main__':
    main()
