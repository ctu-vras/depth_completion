import os
import time
import torch
from supervised_depth_correction.data import Dataset
from supervised_depth_correction.models import SparseConvNet
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import pytorch3d
from pytorch3d.loss import chamfer_distance
from gradslam import Pointclouds, RGBDImages
from gradslam.slam import PointFusion
from chamferdist import ChamferDistance


"""
Demo to show that gradient are propagated through SLAM pipeline with KITTI dataset
training is dome only on single image for simplicity
"""


# -------------- declare global parameters -------------- #
CUDA_DEVICE = 0
DEVICE = torch.device(f"cuda:{CUDA_DEVICE}" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 1
NUM_EPISODES = 200
LR = 0.001
WEIGHT_DECAY = 0.01
EPISODES = 150
VISUALIZE = False

SUBSEQ = "2011_09_26_drive_0001_sync"
USE_DEPTH_SELECTION = False


def train():
    print("###### Loading data ######")
    dataset_gt = Dataset(subseq=SUBSEQ, selection=USE_DEPTH_SELECTION, gt=True)
    dataset_sparse = Dataset(subseq=SUBSEQ, selection=USE_DEPTH_SELECTION, gt=False)
    print("###### Data loaded ######")

    print("###### Setting up training ######")
    model = SparseConvNet()
    model = model.train()
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_training = []
    log_dir = f'./results/depth_completion_gradslam_{time.time()}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    print("###### Starting training ... ######")

    ################# learning loop #################
    for episode in range(EPISODES + 1):
        loss_episode = []
        # for i in dataset_gt.ids:
        i = dataset_gt.ids[0]   # just to test the process
        rgb_img_gt, depth_img_gt, K, pose_gt = dataset_gt[i]
        rgb_img, depth_img_sparse, K, pose = dataset_sparse[i]

        depth_img_gt = depth_img_gt.to(DEVICE)
        depth_img_sparse = depth_img_sparse.to(DEVICE)

        # do forward pass
        mask = (depth_img_sparse > 0).float()
        pred = model(depth_img_sparse, mask)

        # create map with gradslam
        rgbdimages_gt = RGBDImages(rgb_img_gt.to(DEVICE), depth_img_gt, K.to(DEVICE), pose_gt.to(DEVICE)).to(DEVICE)
        slam = PointFusion(device=DEVICE, odom='gradicp', dsratio=4)
        pointclouds_gt, recovered_poses_gt = slam(rgbdimages_gt)

        rgbdimages = RGBDImages(rgb_img.to(DEVICE), pred, K.to(DEVICE), pose.to(DEVICE)).to(DEVICE)
        slam = PointFusion(device=DEVICE, odom='gradicp', dsratio=4)
        pointclouds, recovered_poses = slam(rgbdimages)

        # do backward pass
        optimizer.zero_grad()
        loss = chamfer_loss(pointclouds, pointclouds_gt)
        loss.backward()
        optimizer.step()
        loss_episode.append(loss)

        # compute average episode loss
        loss_episode.append(loss)
        loss_episode = sum(loss_episode) / len(loss_episode)  # average loss per episode
        print(f"EPISODE {episode}/{EPISODES}, loss: {loss_episode}")
        loss_training.append(loss_episode)

        # result visualization
        if episode == EPISODES or episode % (EPISODES // 4) == 0:
            torch.save(model.state_dict(), os.path.join(log_dir, f"weights-{episode}.pth"))

            # convert depth into np images
            depth_img_gt_np = depth_img_gt.detach().cpu().numpy().squeeze()
            depth_img_sparse_np = depth_img_sparse.detach().cpu().numpy().squeeze()
            pred_np = pred.detach().cpu().numpy().squeeze()

            # plot images
            fig = plt.figure()
            ax = fig.add_subplot(3, 1, 1)
            plt.imshow(depth_img_gt_np)
            ax.set_title('Ground truth')
            ax = fig.add_subplot(3, 1, 2)
            plt.imshow(depth_img_sparse_np)
            ax.set_title('Sparse')
            ax = fig.add_subplot(3, 1, 3)
            plt.imshow(pred_np)
            ax.set_title('Prediction')
            fig.tight_layout(h_pad=1)
            # save plot
            plt.savefig(os.path.join(log_dir, f'plot-{episode}.png'))
            if VISUALIZE:
                # show plot
                plt.show()
    # END learning loop

    # plot training loss over episodes
    x_ax = [i for i in range(len(loss_training))]
    y_ax = [loss.detach().cpu().numpy() for loss in loss_training]
    plt.plot(x_ax, y_ax)
    plt.xlabel('Episode')
    plt.ylabel('Training loss')
    plt.title('Loss over episodes')
    plt.savefig(os.path.join(log_dir, 'Training_loss.png'))
    if VISUALIZE:
        plt.show()


def chamfer_loss(pointclouds, pointclouds_gt, py3d=False):
    """
    pointclouds, pointclouds_gt: <gradslam.structures.pointclouds>
    computes chamfer distance between two pointclouds
    :return: <torch.tensor>
    """
    pcd = pointclouds.points_list[0]  # get pointcloud as torch tensor and transform into correct shape
    pcd_gt = pointclouds_gt.points_list[0]
    if py3d:
        pc = pytorch3d.structures.Pointclouds([pcd])
        pc_gt = pytorch3d.structures.Pointclouds([pcd_gt])
        # return chamfer_distance(pc.to(torch.device(DEVICE)), pc_gt.to(torch.device(DEVICE)))[0]
        cd = chamfer_distance(pc, pc_gt)[0]
    else:
        chamferDist = ChamferDistance()
        cd = chamferDist(pcd_gt[None], pcd[None], bidirectional=True)
    return cd


def main():
    train()


if __name__ == '__main__':
    main()
