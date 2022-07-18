import torch
from pytorch3d.loss import chamfer_distance
# from chamferdist import ChamferDistance


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
    return cd


def MSE(gt, pred, mask=True):
    """
    MSE between gt and predicted depth images
    :param depth_gt: <torch.tensor>
    :param depth_pred: <torch.tensor>
    :return: <torch.tensor>
    """
    assert gt.shape == pred.shape

    if mask:
        masked = (gt > 0)
        mse = torch.sum(((gt[masked] - pred[masked]) ** 2)) / torch.sum(masked.int()).float()
    else:
        mse = torch.sum(((gt - pred) ** 2)) / torch.numel(pred).float()
    return mse
