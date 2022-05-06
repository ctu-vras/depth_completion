import gradslam
import open3d as o3d
import matplotlib.pyplot as plt
import torch
import os
from PIL import Image
from scipy import interpolate
import numpy as np
from .models import SparseConvNet
from scipy.spatial import ConvexHull
import torch


def point_visibility(pts, origin, radius=None, param=3.0):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html
    assert isinstance(pts, torch.Tensor)
    assert isinstance(origin, torch.Tensor)
    assert pts.device == origin.device
    assert pts.dim() == origin.dim()
    assert pts.shape[-1] == origin.shape[-1]
    dirs = pts - origin
    dist = torch.norm(dirs, dim=-1, keepdim=True)
    assert dist.shape == (pts.shape[0], 1)
    if radius is None:
        radius = dist.max() * 10.**param
    # Mirror points behind the sphere.
    pts_flipped = pts + 2.0 * (radius - dist) * (dirs / dist)
    # TODO: Allow flexible leading dimensions (convhull needs (npts, ndims)).
    conv_hull = ConvexHull(pts_flipped.detach().cpu().numpy())
    # TODO: Use distance from convex hull to smooth the indicator?
    mask = torch.zeros((pts.shape[0],), device=pts.device)
    mask[conv_hull.vertices] = True
    return torch.as_tensor(mask, dtype=torch.bool)


def interpolate_missing_pixels(
        image: np.ndarray,
        mask: np.ndarray,
        method: str = 'nearest',
        fill_value: float = -1.0
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
    if isinstance(image, np.ndarray):
        xx, yy = np.meshgrid(np.arange(h), np.arange(w))
    elif isinstance(image, torch.Tensor):
        xx, yy = torch.meshgrid(torch.arange(h), torch.arange(w))
    else:
        raise AssertionError('Input image and mask must be both either np.ndarrays or torch.Tensors')

    known_x = xx[~mask]
    known_y = yy[~mask]
    known_v = image[~mask]
    missing_x = xx[mask]
    missing_y = yy[mask]

    interp_values = interpolate.griddata(
        (known_x, known_y), known_v, (missing_x, missing_y),
        method=method, fill_value=fill_value
    )

    if isinstance(image, np.ndarray):
        interp_image = image.copy()
    elif isinstance(image, torch.Tensor):
        interp_image = image.clone()
        interp_values = torch.as_tensor(interp_values, dtype=interp_image.dtype)

    interp_image[missing_x, missing_y] = interp_values

    return interp_image


def filter_depth_outliers(depths):
    """
    Filters out points that are too close or too far away from camera
    """
    depths[depths < 2] = float('nan')
    depths[depths > 15] = float('nan')
    return depths


def filter_pointcloud_outliers(pc):
    """
    Args:
        pc: <gradslam.Pointclouds> or <torch.Tensor>
    """
    assert isinstance(pc, gradslam.Pointclouds)
    pcd = pc.open3d(0)
    o3d.visualization.draw_geometries([pcd])
    pcd = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    o3d.visualization.draw_geometries([pcd])

    return pcd


def depth_filterring_demo():
    from gradslam import Pointclouds, RGBDImages
    from gradslam.slam import PointFusion
    from supervised_depth_correction.data import Dataset
    from tqdm import tqdm
    import cv2

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    subseq = "2011_09_26_drive_0002_sync"
    depth_set = "val"

    ds = Dataset(subseq, depth_type="dense", depth_set=depth_set, camera='left', zero_origin=False, device=device)
    ds_pred = Dataset(subseq, depth_type="pred", depth_set=depth_set, camera='left', zero_origin=False, device=device)
    assert len(ds) > 0
    assert len(ds) == len(ds_pred)

    def generate_point_cloud(gt=True, filter=True):
        slam = PointFusion(device=device, odom='gt', dsratio=1)
        prev_frame = None
        pointclouds = Pointclouds(device=device)

        data = ds if gt else ds_pred

        # for i in tqdm(range(0, len(ds), 5)):
        for i in tqdm(range(0, 1)):
            colors, depths, intrinsics, poses = data[i]

            if filter:
                (B, L, H, W, C) = depths.shape

                # mask = torch.logical_or(depths < 2.0, depths > 40.0).to(device)
                # mask = depths > 30.0
                # mask = depths < 3.0
                # depths[mask] = float('nan')
                # depths = interpolate_missing_pixels(depths.squeeze(), mask.squeeze(), 'linear')

                # https://machinelearningknowledge.ai/bilateral-filtering-in-python-opencv-with-cv2-bilateralfilter/
                # depths = cv2.bilateralFilter(depths.squeeze().cpu().numpy(), -1, 0.03, 4.5)
                depths = cv2.bilateralFilter(depths.squeeze().cpu().numpy(), 5, 80, 80)
                depths = torch.as_tensor(depths.reshape([B, L, H, W, 1])).to(device)

            live_frame = RGBDImages(colors, depths, intrinsics, poses)
            pointclouds, live_frame.poses = slam.step(pointclouds, live_frame, prev_frame)

            prev_frame = live_frame
        pts = pointclouds.points_list[0]
        return pts

    pts = generate_point_cloud(gt=False, filter=False)
    pts_filtered = generate_point_cloud(gt=False, filter=True)
    pts_gt = generate_point_cloud(gt=True, filter=False)

    print('Images are the same:', torch.allclose(pts, pts_filtered))

    # visualize using open3d
    pcds = []
    for i, p in enumerate([pts, pts_filtered, pts_gt]):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(p.cpu() + np.array([100 * i, 0.0, 0.0]))
        pcds.append(pcd)
    o3d.visualization.draw_geometries([p.voxel_down_sample(voxel_size=0.5) for p in pcds])


def depth_outlier_removal_demo():
    from gradslam import Pointclouds, RGBDImages
    from gradslam.slam import PointFusion
    from supervised_depth_correction.data import Dataset
    from tqdm import tqdm

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    subseq = "2011_09_26_drive_0002_sync"
    depth_set = "val"

    ds_pred = Dataset(subseq, depth_type="pred", depth_set=depth_set, camera='left', zero_origin=False, device=device)

    slam = PointFusion(device=device, odom='gt', dsratio=1)
    prev_frame = None
    pointclouds = Pointclouds(device=device)

    # for i in tqdm(range(0, len(ds_pred), 5)):
    for i in tqdm(range(0, 1)):
        colors, depths, intrinsics, poses = ds_pred[i]

        live_frame = RGBDImages(colors, depths, intrinsics, poses)
        pointclouds, live_frame.poses = slam.step(pointclouds, live_frame, prev_frame)

        prev_frame = live_frame

    # visualize using open3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointclouds.points_list[0])
    pcd = pcd.voxel_down_sample(voxel_size=0.5)

    print("Statistical oulier removal")
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.15)

    inlier_cloud = pcd.select_by_index(ind)
    outlier_cloud = pcd.select_by_index(ind, invert=True)

    print('Filtered %d out of %d points' % (len(outlier_cloud.points),
                                            len(outlier_cloud.points) + len(inlier_cloud.points)))

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


def hidden_points_removal_demo():
    # http://www.open3d.org/docs/latest/tutorial/Basic/pointcloud.html#Hidden-point-removal
    from gradslam import Pointclouds, RGBDImages
    from gradslam.slam import PointFusion
    from supervised_depth_correction.data import Dataset
    from tqdm import tqdm

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    subseq = "2011_09_26_drive_0002_sync"
    depth_set = "val"

    ds_pred = Dataset(subseq, depth_type="pred", depth_set=depth_set, camera='left', zero_origin=False, device=device)

    slam = PointFusion(device=device, odom='gt', dsratio=1)
    prev_frame = None
    pointclouds = Pointclouds(device=device)

    for i in tqdm(range(0, 1)):
        colors, depths, intrinsics, poses = ds_pred[i]

        live_frame = RGBDImages(colors, depths, intrinsics, poses)
        pointclouds, live_frame.poses = slam.step(pointclouds, live_frame, prev_frame)

        prev_frame = live_frame

    pts = pointclouds.points_list[0]
    origin = torch.as_tensor([[0.0, 0.0, 0.0]], dtype=pts.dtype).to(device)
    vis_mask = point_visibility(pts, origin=origin, param=2.2)
    pts = pts[vis_mask]

    # visualize using open3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    # depth_filterring_demo()
    # depth_outlier_removal_demo()
    hidden_points_removal_demo()
