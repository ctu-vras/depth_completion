import open3d as o3d
import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import imageio
from tqdm import tqdm
from gradslam.slam import PointFusion
import pytorch3d.structures as structs
from pytorch3d.io import load_obj, save_obj
from pytorch3d.ops import sample_points_from_meshes
from gradslam import RGBDImages
from gradslam.datasets import ICL
from torch.utils.data import DataLoader

from chamferdist_forward import chamfer_distance_forward

# dataset_path = '/home/jachym/BAKAL/gradslam/ros_ws/src/gradslam_ros/data/explorer_x1_rgbd_traj/'
# mesh_path = '/home/jachym/BAKAL/gradslam/ros_ws/src/gradslam_ros/data/meshes-20211022T092424Z-001/meshes/simple_cave_01.obj'
dataset_path = '/home/ruslan/subt/DepthCorrection/depth_completion/ros_ws/src/gradslam_ros/data/explorer_x1_rgbd_traj/'
mesh_path = '/home/ruslan/subt/DepthCorrection/depth_completion/ros_ws/src/gradslam_ros/data/meshes/simple_cave_01.obj'
N = 100
w = 320
h = 240
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# load data
dataset = ICL(dataset_path, seqlen=N, height=h, width=w)
loader = DataLoader(dataset=dataset, batch_size=1)
colors, depths, intrinsics, poses, *_ = next(iter(loader))

# get pointcloud using gradslam
rgbdimages = RGBDImages(colors, depths, intrinsics, poses)
slam = PointFusion(device=device, odom='gt', dsratio=1)
pointclouds, recovered_poses = slam(rgbdimages)

# load gt mesh
verts_gt, faces_gt, aux_gt = load_obj(mesh_path)
faces_idx = faces_gt.verts_idx.to(device)
verts_gt = verts_gt.to(device)
mesh_gt = structs.Meshes(verts=[verts_gt], faces=[faces_idx])
points_mesh = sample_points_from_meshes(mesh_gt, 30000)  # sample points from mesh, otherwise there would be too many

# plot pc from gradslam
points_gradslam = pointclouds.points_list[0]
pc_gradslam = o3d.geometry.PointCloud()
pc_gradslam.points = o3d.utility.Vector3dVector(points_gradslam.cpu().detach().numpy())
#o3d.visualization.draw_geometries([pc_gradslam])

# plot mesh
pc_mesh = o3d.geometry.PointCloud()
pc_mesh.points = o3d.utility.Vector3dVector(points_mesh[0].cpu().detach().numpy())
#o3d.visualization.draw_geometries([pc_mesh])

# plot both meshes together
o3d.visualization.draw_geometries([pc_gradslam, pc_mesh])

# compute chamfer distance for poinclouds
points_gradslam = torch.unsqueeze(points_gradslam, 0)
print(points_mesh.shape, points_gradslam.shape)
dist = chamfer_distance_forward(points_gradslam, points_mesh)
print(dist)
