# TODO:
#   Load local clouds (folder point_clouds) and corresponding poses (file livingRoom1n.gt.sim)
#   Create global map = pointcloud of transposed local pcs (concatenation of T0P0 '+' T1P1)
#   attention T matrices are 3x4 Rotation + translation but in homogeneous coords
#   visualize using open3D
#   IF works properly try it with simple cave 1 and compute mesh face dist
#   Try to visualize PC extracted from mesh as well and plot both pcs together
#   Create separate github repository

# -------------------- IMPLEMENTATION -------------------- #
import open3d as o3d
import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import imageio
from tqdm import tqdm
from pytorch3d.loss import point_mesh_face_distance
import pytorch3d.structures as structs
from pytorch3d.io import load_obj, save_obj

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def open_poses(file):
    poses = list()

    with open(file) as f:
        lines = f.readlines()
        mat = list()
        for line in lines:
            if line in {"\n"}:
                mat.append([0, 0, 0, 1])    # we need transformation matrix in homogeneous coords
                poses.append(mat)
                mat = list()
            else:
                line = line.replace("\n", "")
                vals = list(map(float, line.split()))
                mat.append(vals)
    return np.array(poses)


data_path = "/home/jachym/BAKAL/gradslam/ros_ws/src/gradslam_ros/data/explorer_x1_rgbd_traj/living_room_traj1_frei_png/"
mesh_path = '/home/jachym/BAKAL/gradslam/ros_ws/src/gradslam_ros/data/meshes-20211022T092424Z-001/meshes/simple_cave_01.obj'

# --------- loading point clouds and poses --------- #
pc_folder = os.path.join(data_path, "point_clouds")
raw_pcs = next(os.walk(pc_folder))[2]
poses = open_poses(os.path.join(data_path, "livingRoom1n.gt.sim"))
# assert len(raw_pcs) == poses.shape[0]   # poses = 590 number of pc = 584
pc = None

# concatenate pcs
#for i in range(len(raw_pcs)):
for i in range(50):
    print(i)
    # load point clouds
    pts = np.load(os.path.join(pc_folder, raw_pcs[i]))
    # turn points into homogeneous coords
    hom = np.tile(np.array([1]), (pts.shape[0], 1))
    pts = np.concatenate((pts, hom), axis=1)
    # transform
    pts = np.transpose(pts)
    pose = poses[i]
    pci = np.matmul(pose, pts)

    if pc is None:
        pc = pts
    else:
        pc = np.concatenate((pc, pci), axis=1)

pcd = o3d.geometry.PointCloud()
pc = np.delete(pc, 3, axis=0)   # remove one at the end (back from hom coords)
pc = pc.transpose()     # we need shape nbr_points x 3
pcd.points = o3d.utility.Vector3dVector(pc)
o3d.visualization.draw_geometries([pcd])
print(f"dims {pc.shape}")
pc = torch.from_numpy(pc)
pc = pc.float()
pc3d = structs.Pointclouds([pc])

# --------- compare with mesh --------- #
"""verts_gt, faces_gt, aux_gt = load_obj(mesh_path)
faces_idx = faces_gt.verts_idx.to(device)
verts_gt = verts_gt.to(device)
mesh_gt = structs.Meshes(verts=[verts_gt], faces=[faces_idx])
# plot mesh
pcmesh = o3d.geometry.PointCloud()
pcmesh.points = o3d.utility.Vector3dVector(verts_gt.cpu().detach().numpy())
o3d.visualization.draw_geometries([pcmesh])
dist = point_mesh_face_distance(mesh_gt, pc3d)
print(dist)"""



