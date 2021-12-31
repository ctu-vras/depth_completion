import torch
import gradslam
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from DepthCorrectionModule import Dataset, LinTransModel, ModelV1
from gradslam.datasets import TUM, ICL
from Loss import chamfer_loss
from Slam import Slam
from metrics import mse_depth_dataset


# -------------- declare global parameters -------------- #
#DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
DEVICE = "cpu"
BATCH_SIZE = 1
NUM_EPISODES = 51
LR = 0.001
weight_decay = 0.01


def load_model():
    pass


def main():
    # -------------- load data -------------- #
    # used ICL dataset "Living Room 'lr kt2'"
    # data source: https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html
    # used ICL dataset has been shrunk to 1% of its original size to be able to run it locally on GPU
    path_ICL = "/home/jachym/MEGAsync/KYR/Bakalarka/ICL-dataset/"
    path_noisy_data = "/home/jachym/MEGAsync/KYR/Bakalarka/ICL-noise/"

    # -------------- unpack data -------------- #
    dataset_ICL = ICL(path_ICL, seqlen=4)
    loader_ICL = iter(DataLoader(dataset=dataset_ICL, batch_size=BATCH_SIZE))

    # -------------- unpack data -------------- #
    dataset_noise = ICL(path_noisy_data, seqlen=4)
    loader_n = iter(DataLoader(dataset=dataset_noise, batch_size=BATCH_SIZE))

    # -------------- create model and optimizer -------------- #
    # TODO create load model and initialize model versions
    #model = LinTransModel()
    model = ModelV1()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=weight_decay)
    optimizer.zero_grad()

    # -------------- create gt map -------------- #
    total_depths_gt = []    # list of depth from each sequence
    global_map_gt = gradslam.Pointclouds(device=DEVICE)
    step = 0

    for colors_gt, depths_gt, intrinsics_gt, poses_gt, *_ in loader_ICL:
        colors_gt = colors_gt.to(DEVICE)
        depths_gt = depths_gt.to(DEVICE)
        intrinsics_gt = intrinsics_gt.to(DEVICE)
        poses_gt = poses_gt.to(DEVICE)
        # update slam
        pointclouds_gt, recovered_poses_gt = Slam(colors_gt, depths_gt, intrinsics_gt, poses_gt, DEVICE, odom='gt')
        total_depths_gt.append(depths_gt)
        global_map_gt.append_points(pointclouds_gt)     # update map
        print(f"--- Creating GT map, step {step} --- ")
        step += 1

    # -------------- load noisy data -------------- #
    sequence_list = []  # list of sequence data as tuples
    num_seq = 0     # number of sequences from loader

    for colors_n, depths_n, intrinsics_n, poses_n, *_ in loader_n:
        # move to device"
        sequence_list.append( (colors_n, depths_n, intrinsics_n, poses_n) )
        num_seq += 1

    # -------------- train model -------------- #
    model = model.train()
    model = model.to(DEVICE)

    losses = []
    dept_differences = []
    for episode in range(NUM_EPISODES):
        episode_depths_n = []
        global_map_episode = gradslam.Pointclouds(device=DEVICE)

        # go through sequences
        for i in range(num_seq):
            colors_n, depths_n, intrinsics_n, poses_n = sequence_list[i]
            # change depth data into correct shape
            # noisy depth data loads as [B, S, H, W, 3, 1] but should be [B, S, H, W, 1]
            depths_n = torch.squeeze(depths_n[..., 0, :], -4)
            assert depths_n.dim() == 5

            # do a forward pass
            depths_n = model(depths_n)

            # perform slam
            pointclouds_n, recovered_poses_n = Slam(colors_n, depths_n, intrinsics_n, poses_n, DEVICE, odom='gradicp')

            print(f"iteration {i}")

            # append data
            episode_depths_n.append(depths_n)
            global_map_episode.append_points(pointclouds_n)

        # do a backward pass
        loss = chamfer_loss(global_map_episode, global_map_gt)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        mse = mse_depth_dataset(total_depths_gt, episode_depths_n)
        print(f"\nLoss on episode {episode}: {loss}")
        print(f"Mean MSE between noisy data and gt data: {mse}")
        losses.append(loss.cpu().detach().numpy())
        dept_differences.append(mse)

        if episode % 10 == 0:
            torch.save(model.state_dict(), f"weights{episode}.pth")

    #print(colors_n.shape)
    #print(depths_n.shape)


    # -------------- plot results -------------- #
    # plotting the points
    x_ax = [i for i in range(len(losses))]
    y_ax = losses
    plt.plot(x_ax, y_ax)
    plt.xlabel('episode')
    plt.ylabel('loss')
    plt.title('Loss over episodes')
    plt.show()

    x_ax = [i for i in range(len(dept_differences))]
    y_ax = dept_differences
    plt.plot(x_ax, y_ax)
    plt.xlabel('episode')
    plt.ylabel('MSE metric')
    plt.title('MSE over episodes')
    plt.show()


if __name__ == '__main__':
    main()
