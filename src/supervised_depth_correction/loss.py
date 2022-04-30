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


def MSE_loss(img_gt, img_pred):
    pass
