import os
import time
import torch
from gradslam import Pointclouds, RGBDImages
from gradslam.slam import PointFusion
# from tqdm import tqdm
# from torch.utils.data import DataLoader
from supervised_depth_correction.data import Dataset
from supervised_depth_correction.io import write, append
from supervised_depth_correction.models import SparseConvNet
from supervised_depth_correction.metrics import MSE, MAE, chamfer_loss, localization_accuracy
from supervised_depth_correction.utils import plot_depth, plot_pc, plot_metric


# ------------------------------------ GLOBAL PARAMETERS ------------------------------------ #
CUDA_DEVICE = 0
DEVICE = torch.device(f"cuda:{CUDA_DEVICE}" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device('cpu')

BATCH_SIZE = 1
LR = 0.001
WEIGHT_DECAY = 0.01

EPISODES = 100
VISUALIZE = False

USE_DEPTH_SELECTION = False
LOG_DIR = os.path.join(os.path.dirname(__file__), '..', f'config/results/depth_completion_gradslam_{time.time()}')
INIT_MODEL_STATE_DICT = os.path.join(os.path.dirname(__file__), 'weights.pth')


# ------------------------------------ Helper functions ------------------------------------ #
def construct_map(ds, predictor=None, pose_provider='gt', max_clouds=6, step=1):
    slam = PointFusion(device=DEVICE, odom=pose_provider, dsratio=1)
    prev_frame = None
    pointclouds = Pointclouds(device=DEVICE)
    trajectory = []

    # TODO: check memory issue
    for i in ds.ids[:max_clouds:step]:
        colors, depths, intrinsics, poses = ds[i]

        # do forward pass
        if predictor is not None:
            mask = (depths > 0).float()
            pred = predictor(depths, mask)
            depths = pred

        live_frame = RGBDImages(colors, depths, intrinsics, poses)
        pointclouds, live_frame.poses = slam.step(pointclouds, live_frame, prev_frame)
        prev_frame = live_frame

        trajectory.append(live_frame.poses)

        # write slam poses
        slam_pose = live_frame.poses.detach().squeeze()
        append(os.path.join(os.path.dirname(__file__), '..', 'config/results/', 'slam_poses.txt'),
               ', '.join(['%.6f' % x for x in slam_pose.flatten()]) + '\n')
    return pointclouds, trajectory, depths


def compute_val_metrics(depth_gt, depth_pred, traj_gt, traj_pred):
    mse = MSE(depth_gt, depth_pred)
    mae = MAE(depth_gt, depth_pred)
    loc_acc = localization_accuracy(traj_gt, traj_pred)
    return mse, mae, loc_acc


def load_model(path=None):
    model = SparseConvNet()
    if path:
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location='cpu'))
        else:
            print('No model weights found. Training from scratch!!!')
    return model


# ------------------------------------ Learning functions------------------------------------ #
def train(subseqs):
    subseq = subseqs[0]
    print("###### Starting training ######")
    print("###### Loading data ######")
    dataset_gt = Dataset(subseq=subseq, selection=USE_DEPTH_SELECTION, gt=True, device=DEVICE)
    dataset_sparse = Dataset(subseq=subseq, selection=USE_DEPTH_SELECTION, gt=False, device=DEVICE)
    print("###### Data loaded ######")

    print("###### Setting up training environment ######")
    model = load_model(INIT_MODEL_STATE_DICT)
    model = model.train()
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_training = []
    mse_training = []
    mae_training = []
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    # create gt global map
    global_map_gt, traj_gt, depth_sample_gt = construct_map(dataset_gt)
    global_map_sparse, traj_sparse, depth_sample_sparse = construct_map(dataset_sparse)
    plot_pc(global_map_gt, "gt_dense", "training", visualize=VISUALIZE, log_dir=LOG_DIR)
    plot_pc(global_map_sparse, "gt_sparse", "training", visualize=VISUALIZE, log_dir=LOG_DIR)
    print("###### Running training loop ######")

    for episode in range(EPISODES + 1):
        global_map_episode, traj_epis, depth_sample_pred = construct_map(dataset_sparse, predictor=model)

        # do backward pass
        optimizer.zero_grad()
        loss = chamfer_loss(global_map_episode, global_map_gt, sample_step=5)
        loss.backward()
        optimizer.step()

        # print and append training metrics
        loss_training.append(loss.detach())
        print(f"EPISODE {episode}/{EPISODES}, loss: {loss}")
        mse, mae, _ = compute_val_metrics(depth_sample_gt, depth_sample_pred, traj_gt, traj_epis)
        print(f"EPISODE {episode}/{EPISODES}, MSE: {mse}")
        print(f"EPISODE {episode}/{EPISODES}, MAE: {mae}")
        mse_training.append(mse.detach())
        mae_training.append(mae.detach())

        # running results save
        if episode == EPISODES or episode % (EPISODES // 2) == 0:
            torch.save(model.state_dict(), os.path.join(LOG_DIR, f"weights-{episode}.pth"))
            plot_pc(global_map_episode, episode, "training", visualize=VISUALIZE, log_dir=LOG_DIR)
            plot_depth(depth_sample_sparse, depth_sample_pred, depth_sample_gt, "training", episode, visualize=VISUALIZE, log_dir=LOG_DIR)
    # END learning loop

    # plot training metrics over episodes
    plot_metric(loss_training, "Training loss", visualize=VISUALIZE, log_dir=LOG_DIR)
    plot_metric(mse_training, "MSE", visualize=VISUALIZE, log_dir=LOG_DIR)
    plot_metric(mae_training, "MAE", visualize=VISUALIZE, log_dir=LOG_DIR)

    return model


def test(subseqs, model):
    """
    Test if results of trained model are comparable on other subsequences
    """
    for subseq in subseqs:
        print(f"###### Starting testing of subseq: {subseq} ######")
        print("###### Loading data ######")
        dataset_gt = Dataset(subseq=subseq, selection=USE_DEPTH_SELECTION, gt=True, device=DEVICE)
        dataset_sparse = Dataset(subseq=subseq, selection=USE_DEPTH_SELECTION, gt=False, device=DEVICE)
        print("###### Data loaded ######")

        print("###### Constructing maps ######")

        global_map_gt, traj_gt, depth_sample_gt = construct_map(dataset_gt)
        plot_pc(global_map_gt, "gt", f"testing-{subseq}", visualize=VISUALIZE, log_dir=LOG_DIR)

        global_map_sparse, traj_sparse, depth_sample_sparse = construct_map(dataset_sparse)
        plot_pc(global_map_gt, "sparse", f"testing-{subseq}", visualize=VISUALIZE, log_dir=LOG_DIR)

        global_map_pred, traj_pred, depth_sample_pred = construct_map(dataset_sparse, model, pose_provider='icp')
        plot_pc(global_map_pred, "pred", f"testing-{subseq}", visualize=VISUALIZE, log_dir=LOG_DIR)

        print("###### Running testing ######")

        mse_sparse, mae_sparse, locc_acc_sparse = compute_val_metrics(depth_sample_gt, depth_sample_sparse, traj_gt, traj_sparse)
        print(f"MSE sparse: {mse_sparse}")
        print(f"MAE sparse: {mae_sparse}")
        print(f"Localization accuracy sparse: {locc_acc_sparse}")

        mse_pred, mae_pred, locc_acc_pred = compute_val_metrics(depth_sample_gt, depth_sample_pred, traj_gt, traj_pred)
        print(f"MSE pred: {mse_pred}")
        print(f"MAE pred: {mae_pred}")
        print(f"Localization accuracy pred: {locc_acc_pred}")

        print("###### ALL DONE ######")


def main():
    train_subseq = ["2011_09_26_drive_0001_sync"]
    test_subseqs = ["2011_09_28_drive_0191_sync"]
    assert not any(x in test_subseqs for x in train_subseq)
    model = train(train_subseq)
    test(test_subseqs, model)


if __name__ == '__main__':
    main()
    # TODO:
    #   make training over multiple subsequences
    #   --> run on sequences sequentially
    #   Make separate singularity image for gradslam?
    #   --> train on server with gt and test locally with gradicp
