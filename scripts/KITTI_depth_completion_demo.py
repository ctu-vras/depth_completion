from supervised_depth_correction.data import Dataset
from supervised_depth_correction.models import DnCNN_c, SparseConvNet
import torch
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

"""
Model aims to convert sparse depth images into dense ones
"""

# -------------- declare global parameters -------------- #
# DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
CUDA_DEVICE = 0
DEVICE = torch.device(f"cuda:{CUDA_DEVICE}" if torch.cuda.is_available() else "cpu")
RAW_DATA_DIR = "/home/jachym/KITTI/kitti_raw"
DEPTH_DATA_DIR = "/home/jachym/KITTI/depth_selection/val_selection_cropped"
# DEPTH_DATA_DIR = "/home/ruslan/data/datasets/kitti_depth/depth_selection/val_selection_cropped"
# RAW_DATA_DIR = "/home.nfs/stanejac/KITTI/kitti_raw"
# DEPTH_DATA_DIR = "/home.nfs/stanejac/KITTI/depth_selection/val_selection_cropped"
EPISODES = 1000
LR = 0.001
WEIGHT_DECAY = 0.01


def main():
    # prepare datasets, model and optimizer
    print("###### Loading data ######")
    subseq = "2011_09_26_drive_0002_sync"
    dataset_gt = Dataset(subseq=subseq, gt=True)
    dataset_sparse = Dataset(subseq=subseq, gt=False)
    data_len = len(dataset_gt.ids)
    # model = DnCNN_c(channels=1, num_of_layers=17)
    print("###### Data loaded ######")
    print("###### Setting up training ######")
    model = SparseConvNet()
    model = model.train()
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_training = []
    print("###### Starting training ... ######")

    # learning loop
    for episode in range(EPISODES+1):
        loss_episode = []
        for i in dataset_gt.ids:
            rgb_img_gt, depth_img_gt, K, pose = dataset_gt[i]
            rgb_img_gt, depth_img_sparse, K, pose = dataset_sparse[i]
            #print(torch.sum(depth_img_sparse), torch.sum(depth_img_gt))
            depth_img_gt = depth_img_gt.to(DEVICE)
            depth_img_sparse = depth_img_sparse.to(DEVICE)

            mask = (depth_img_sparse > 0).float()
            pred = model(depth_img_sparse, mask)
            loss = loss_mse(depth_img_gt, pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_episode.append(loss)

        loss_episode = sum(loss_episode) / len(loss_episode)    # average loss per episode
        print(f"EPISODE {episode}/{EPISODES}, loss: {loss_episode}")
        loss_training.append(loss_episode)

        # result visualization
        if episode == EPISODES or episode % (EPISODES // 4) == 0:
            torch.save(model.state_dict(), f"weights-{episode}.pth")

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
            plt.savefig(f'plots{episode}.png')
            # show plot
            plt.show()
    # END training loop

    # plot loss over episodes
    x_ax = [i for i in range(len(loss_training))]
    y_ax = loss_training
    plt.plot(x_ax, y_ax)
    plt.xlabel('Episode')
    plt.ylabel('Training loss')
    plt.title('Loss over episodes')
    plt.savefig(f'Training_loss.png')
    plt.show()


def loss_mse(depth_gt, depth_pred):
    """
    MSE between gt and predicted depth images
    :param depth_gt: <torch.tensor>
    :param depth_pred: <torch.tensor>
    :return: <torch.tensor>
    """
    assert depth_gt.shape == depth_pred.shape
    mse = torch.sum((depth_gt - depth_pred) ** 2)
    mse /= float(depth_gt.shape[0] * depth_gt.shape[1] * depth_gt.shape[2])
    return mse      # might require adding 'requires_grad=True'


if __name__ == '__main__':
    main()
