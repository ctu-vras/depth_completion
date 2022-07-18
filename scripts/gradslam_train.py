import os
import time
import torch
from depth_completion.data import Dataset
from depth_completion.models import SparseConvNet
import matplotlib.pyplot as plt
from gradslam import Pointclouds, RGBDImages
from gradslam.slam import PointFusion
from depth_completion.utils import plot_pc
from depth_completion.loss import chamfer_loss
"""
Demo to show that gradient are propagated through SLAM pipeline with KITTI dataset
training is dome only on single image for simplicity
"""


# -------------- declare global parameters -------------- #
CUDA_DEVICE = 0
DEVICE = torch.device(f"cuda:{CUDA_DEVICE}" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 1
LR = 0.001
WEIGHT_DECAY = 0.01
EPISODES = 150
VISUALIZE = True

SUBSEQ = "2011_09_26_drive_0001_sync"
USE_DEPTH_SELECTION = False
LOG_DIR = os.path.join(os.path.dirname(__file__), '..', f'config/results/depth_completion_gradslam_{time.time()}')


def train():
    print("###### Loading data ######")
    dataset_dense = Dataset(subseq=SUBSEQ, selection=USE_DEPTH_SELECTION, depth_type="dense")
    dataset_sparse = Dataset(subseq=SUBSEQ, selection=USE_DEPTH_SELECTION, depth_type="sparse")
    print("###### Data loaded ######")

    print("###### Setting up training ######")
    model = SparseConvNet()
    model = model.train()
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_training = []
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    print("###### Starting training ... ######")

    ################# learning loop #################
    for episode in range(EPISODES + 1):
        loss_episode = []
        # for i in dataset_gt.ids:
        i = dataset_dense[0]   # just to test the process
        rgb_img_gt, depth_img_gt, K, pose_gt = dataset_dense[i]
        rgb_img, depth_img_sparse, K, pose = dataset_sparse[i]

        depth_img_gt = depth_img_gt.to(DEVICE)
        depth_img_sparse = depth_img_sparse.to(DEVICE)

        # do forward pass
        mask = (depth_img_sparse > 0).float()
        pred = model(depth_img_sparse, mask)

        # create map with gradslam
        rgbdimages_gt = RGBDImages(rgb_img_gt.to(DEVICE), depth_img_gt, K.to(DEVICE), pose_gt.to(DEVICE)).to(DEVICE)
        slam = PointFusion(device=DEVICE, odom='gradicp', dsratio=1)
        pointclouds_gt, recovered_poses_gt = slam(rgbdimages_gt)

        rgbdimages = RGBDImages(rgb_img.to(DEVICE), pred, K.to(DEVICE), pose.to(DEVICE)).to(DEVICE)
        slam = PointFusion(device=DEVICE, odom='gradicp', dsratio=1)
        pointclouds, recovered_poses = slam(rgbdimages)

        # do backward pass
        optimizer.zero_grad()
        loss = chamfer_loss(pointclouds, pointclouds_gt, sample_step=5)
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
            torch.save(model.state_dict(), os.path.join(LOG_DIR, f"weights-{episode}.pth"))

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
            plt.savefig(os.path.join(LOG_DIR, f'plot-{episode}.png'))
            if VISUALIZE:
                # show plot
                plt.show()
                # plot_pc(pointclouds)
                # plot_pc(pointclouds_gt)
    # END learning loop

    # plot training loss over episodes
    x_ax = [i for i in range(len(loss_training))]
    y_ax = [loss.detach().cpu().numpy() for loss in loss_training]
    plt.plot(x_ax, y_ax)
    plt.xlabel('Episode')
    plt.ylabel('Training loss')
    plt.title('Loss over episodes')
    plt.savefig(os.path.join(LOG_DIR, 'Training_loss.png'))
    if VISUALIZE:
        plt.show()


def main():
    train()


if __name__ == '__main__':
    main()
