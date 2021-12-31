import torch
from gradslam import Pointclouds, RGBDImages
from gradslam.slam import PointFusion
from time import time
from pytorch3d.loss import point_mesh_face_distance
import pytorch3d.structures as structs
from pytorch3d.io import load_obj, save_obj


SLAM_DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# run slam algorithm
def SlamPerFrame(depth_pred, rgb_data, device):
    """
    This function does not work properly at this time
    """
    # load dataset
    intrinsics = torch.rand([1, 1, 4, 4])  # TODO Load from cmainfo
    depth_pred = depth_pred.permute((1,2,0))   # CH, H, W -> H, W, CH
    depth_pred = torch.unsqueeze(depth_pred, dim=0)
    depth_pred = torch.unsqueeze(depth_pred, dim=0)
    rgb_data = torch.unsqueeze(rgb_data, dim=0)
    #print(rgb_data.shape)

    # create rgbdimages object
    rgbdimages = RGBDImages(rgb_data, depth_pred, intrinsics)

    # step by step SLAM
    slam = PointFusion(device=device)

    pointclouds = Pointclouds(device=device)
    batch_size, seq_len = rgbdimages.shape[:2]
    initial_poses = torch.eye(4, device=device).view(1, 1, 4, 4).repeat(batch_size, 1, 1, 1)
    prev_frame = None
    #print(rgbdimages.shape)
    t0 = time()

    for s in range(seq_len):
        # t0 = time()
        live_frame = rgbdimages[:, s].to(device)
        #print(f"live frame {live_frame.shape}")
        if s == 0 and live_frame.poses is None:
            live_frame.poses = initial_poses
        pointclouds, live_frame.poses = slam.step(pointclouds, live_frame, prev_frame)
        #print(f"Live frame poses.shape{live_frame.poses.shape}, pointclouds.points_padded.shape {pointclouds.points_padded.shape}")
        #print(f'SLAM step took {(time()-t0):.3f} sec')
        #print(f'Image shape: {live_frame.rgb_image.shape}')
        prev_frame = live_frame
    # return as pytorch3d pointcloud
    print(f"slam took {time()-t0}")
    return structs.Pointclouds(pointclouds.points_list, pointclouds.normals_list)


def Slam(colors, depths, intrinsics, poses, device, odom='gt'):
    rgbdimages = RGBDImages(colors, depths, intrinsics, poses).to(device)
    #slam = PointFusion(device=device, odom=odom, dsratio=4)
    slam = PointFusion(device=SLAM_DEVICE, odom=odom, dsratio=4)
    pointclouds, recovered_poses = slam(rgbdimages)
    #return pointclouds, recovered_poses
    return pointclouds.to("cpu"), recovered_poses.to("cpu")


def gt_mesh_dist(depth_pred, rgb_data, mesh, device='cpu'):
    t0 = time()
    pc = SlamPerFrame(depth_pred, rgb_data, device)
    verts_gt, faces_gt, aux_gt = load_obj(mesh)
    # verts is a FloatTensor of shape (V, 3) where V is the number of vertices in the mesh
    # faces is an object which contains the following LongTensors: verts_idx, normals_idx and textures_idx
    # For this tutorial, normals and textures are ignored.
    faces_idx = faces_gt.verts_idx.to(device)
    verts_gt = verts_gt.to(device)

    # We construct a Meshes structure for the target mesh
    mesh_gt = structs.Meshes(verts=[verts_gt], faces=[faces_idx])

    # compute distance between recovered slam pc and gt mesh
    d = point_mesh_face_distance(mesh_gt, pc)
    print(f"point mesh dist took {time() - t0}")
    return torch.tensor([d], requires_grad=True)
