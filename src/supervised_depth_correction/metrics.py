import torch
import pytorch3d
from pytorch3d.loss import chamfer_distance
from chamferdist import ChamferDistance


def chamfer_loss(pointclouds, pointclouds_gt, py3d=True):
    """
    pointclouds, pointclouds_gt: <gradslam.structures.pointclouds>
    computes chamfer distance between two pointclouds
    :return: <torch.tensor>
    """
    pcd = pointclouds.points_list[0]  # get pointcloud as torch tensor and transform into correct shape
    pcd_gt = pointclouds_gt.points_list[0]
    if py3d:
        cd = chamfer_distance(torch.unsqueeze(pcd, 0), torch.unsqueeze(pcd_gt, 0))[0]
    else:
        chamferDist = ChamferDistance()
        cd = chamferDist(pcd_gt[None], pcd[None], bidirectional=True)
    return cd


def MSE(img1, img2):
    """
    MSE between gt and predicted depth images
    :param depth_gt: <torch.tensor>
    :param depth_pred: <torch.tensor>
    :return: <torch.tensor>
    """
    assert img1.shape == img2.shape
    mse = torch.sum((img1 - img2) ** 2)
    mse /= float(img1.shape[0] * img1.shape[1] * img1.shape[2])
    return mse


def MAE(img1, img2):
    """
    MAE between gt and predicted depth images
    :param depth_gt: <torch.tensor>
    :param depth_pred: <torch.tensor>
    :return: <torch.tensor>
    """
    assert img1.shape == img2.shape
    mse = torch.sum(img1 - img2)
    mse /= float(img1.shape[0] * img1.shape[1] * img1.shape[2])
    return mse


def localization_accuracy(poses1, poses2):
    """
    Computes sum of euclidian distances between carthesian coordinates of each transformation matrix
    Args:
        poses1: list of <torch.tensor>
        poses2: list of <torch.tensor>

    Returns: <torch.tensor>
    """
    assert len(poses1) == len(poses2)
    err = 0
    for i in range(len(poses1)):
        pose1 = poses1[i]
        pose2 = poses2[i]
        x1, y1, z1 = pose1[..., 0, 3], pose1[..., 1, 3], pose1[..., 2, 3]
        x2, y2, z2 = pose2[..., 0, 3], pose2[..., 1, 3], pose2[..., 2, 3]
        dist = ( (x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2 ) ** (1/2)
        err += dist
    return err
