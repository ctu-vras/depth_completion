import gradslam as gs
import os
import torch
from gradslam import Pointclouds, RGBDImages
from gradslam.datasets import TUM, ICL
from gradslam.slam import PointFusion, ICPSLAM
from torch.utils.data import DataLoader
import open3d as o3d

from pytorch3d.loss import chamfer_distance


def plot_pc2compare(pc1, pc2):
    """
    pc1, pc2: torch tensor in shape (NUM_POINTS, 3)
    Plots two pointclouds in the same plot
    """
    p1 = o3d.geometry.PointCloud()
    p1.points = o3d.utility.Vector3dVector(pc1.cpu().detach().numpy())
    p2 = o3d.geometry.PointCloud()
    p2.points = o3d.utility.Vector3dVector(pc2.cpu().detach().numpy())
    o3d.visualization.draw_geometries([p1, p2])


def chamfer_loss(pointclouds, pointclouds_gt):
    """
    pointclouds, pointclouds_gt: <gradslam.structures.pointclouds>
    computes chamfer distance between two pointclouds
    :return: <torch.tensor>
    """
    pc = torch.unsqueeze(pointclouds.points_list[0], 0)  # get pointcloud as torch tensor and transform into correct shape
    pc_gt = torch.unsqueeze(pointclouds_gt.points_list[0], 0)

    # print("chamfer distance type:", type(chamfer_distance(pc, pc_gt)))
    #return chamfer_distance(pc, pc_gt)[0]
    return chamfer_distance(pc.to(torch.device("cuda")), pc_gt.to(torch.device("cuda")))[0]




