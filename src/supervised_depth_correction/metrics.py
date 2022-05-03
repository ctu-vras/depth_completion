import torch
from .transform import *


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


def localization_accuracy(poses1, poses2, trans_rot_combined=True):
    """
    Computes sum of euclidian distances between carthesian coordinates of each transformation matrix
    Args:
        poses1: list of <torch.tensor>
        poses2: list of <torch.tensor>
        trans_rot_combined: bool, Whether to return sum of translational and rotational errors or separate values

    Returns: <torch.tensor>
    """
    assert len(poses1) == len(poses2)
    err_dist = 0
    err_rot = 0
    for i in range(len(poses1)):
        pose1 = poses1[i]
        pose2 = poses2[i]
        delta_T = delta_transform(pose1.squeeze(), pose2.squeeze())
        trans_delta, rot_delta = translation_norm(delta_T), rotation_angle(delta_T)
        err_dist += trans_delta
        err_rot += rot_delta
    err_dist /= len(poses1)
    err_rot /= len(poses1)
    if trans_rot_combined:
        return err_dist + err_rot
    else:
        return err_dist, err_rot
