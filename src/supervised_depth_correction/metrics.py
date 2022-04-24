import torch
from pytorch3d.loss import chamfer_distance
from chamferdist import ChamferDistance
from .transform import *


def chamfer_loss(pointclouds, pointclouds_gt, sample_step=1):
    """
    pointclouds, pointclouds_gt: <gradslam.structures.pointclouds>
    computes chamfer distance between two pointclouds
    :return: <torch.tensor>
    """
    pcd = pointclouds.points_list[0]  # get point cloud as torch tensor and transform into correct shape
    pcd_gt = pointclouds_gt.points_list[0]
    if sample_step > 1:
        # randomly unifromly sample pts from clouds with step equal to sample_step
        pcd = pcd[torch.randint(pcd.shape[0], (pcd.shape[0]//sample_step,)), :]
        pcd_gt = pcd_gt[torch.randint(pcd_gt.shape[0], (pcd_gt.shape[0] // sample_step,)), :]
    cd = chamfer_distance(pcd[None], pcd_gt[None])[0]
    # chamferDist = ChamferDistance()
    # cd = chamferDist(pcd_gt[None], pcd[None], bidirectional=True)
    return cd


def RMSE(gt, pred, mask=True):
    """
    RMSE between gt and predicted depth images
    :param depth_gt: <torch.tensor>
    :param depth_pred: <torch.tensor>
    :return: <torch.tensor>
    """
    assert gt.shape == pred.shape

    if mask:
        masked = (gt > 0)
        rmse = torch.sum( ((gt[masked] - pred[masked]) ** 2) ** (1/2) )
    else:
        rmse = torch.sum( ((gt - pred) ** 2) ** (1/2) )
    rmse /= float(gt.shape[0] * gt.shape[1] * gt.shape[2])
    return rmse


def MAE(gt, pred, mask=True):
    """
    MAE between gt and predicted depth images
    """
    assert gt.shape == pred.shape

    # https://medium.com/human-in-a-machine-world/mae-and-rmse-which-metric-is-better-e60ac3bde13d
    if mask:
        masked = (gt > 0)
        mae = torch.sum(torch.abs(gt[masked] - pred[masked]))
    else:
        mae = torch.sum(torch.abs(gt - pred))
    mae /= float(gt.shape[0] * gt.shape[1] * gt.shape[2])
    return mae


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
        delta_T = delta_transform(pose1.squeeze(), pose2.squeeze())
        dist, rot_dist = translation_norm(delta_T), rotation_angle(delta_T)
        err += dist + rot_dist
    return err