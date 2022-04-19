import os
import time
import torch
import open3d as o3d
import matplotlib.pyplot as plt
from gradslam import Pointclouds, RGBDImages
from gradslam.slam import PointFusion
# from tqdm import tqdm
# from torch.utils.data import DataLoader
from supervised_depth_correction.data import Dataset
from supervised_depth_correction.io import write, append
from supervised_depth_correction.models import SparseConvNet
from supervised_depth_correction.metrics import MSE, MAE, chamfer_loss, localization_accuracy


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
def construct_map(ds, predictor=None, pose_provider='gt', max_clouds=5, step=1):
    slam = PointFusion(device=DEVICE, odom=pose_provider, dsratio=1)
    prev_frame = None
    pointclouds = Pointclouds(device=DEVICE)
    trajectory = []

    # TODO: check memory issue (sample global map before loss computation)
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


def plot_depth(depth_sparse, depth_pred, depth_gt, episode, mode):
    # convert depth into np images
    depth_img_gt_np = depth_gt.detach().cpu().numpy().squeeze()
    depth_img_sparse_np = depth_sparse.detach().cpu().numpy().squeeze()
    pred_np = depth_pred.detach().cpu().numpy().squeeze()

    # plot images
    fig = plt.figure()
    ax = fig.add_subplot(3, 1, 1)
    plt.imshow(depth_img_sparse_np)
    ax.set_title('Sparse')
    ax = fig.add_subplot(3, 1, 2)
    plt.imshow(pred_np)
    ax.set_title('Prediction')
    ax = fig.add_subplot(3, 1, 3)
    plt.imshow(depth_img_gt_np)
    ax.set_title('Ground truth')
    fig.tight_layout(h_pad=1)
    # save plot
    plt.savefig(os.path.join(LOG_DIR, f'plot-{mode}-{episode}.png'))
    if VISUALIZE:
        plt.show()


def plot_pc(pc, episode, mode):
    """
    Args:
        pc: <gradslam.Pointclouds>
    """
    pc_o3d = pc.open3d(0)
    if VISUALIZE:
        # Flip it, otherwise the pointcloud will be upside down
        pc_o3d.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        o3d.visualization.draw_geometries([pc_o3d])
    o3d.io.write_point_cloud(os.path.join(LOG_DIR, f'map-{mode}-{episode}.pcd'), pc_o3d)


def compute_val_metrics(depth_gt, depth_pred, traj_gt, traj_pred):
    mse = MSE(depth_gt, depth_pred)
    mae = MAE(depth_gt, depth_pred)
    loc_acc = localization_accuracy(traj_gt, traj_pred)
    return mse, mae, loc_acc


def plot_metric(metric, metric_title):
    """
    Plots graph of metric over episodes
    Args:
        metric: list of <torch.tensor>
        metric_title: string
    """
    x_ax = [i for i in range(len(metric))]
    y_ax = [loss.detach().cpu().numpy() for loss in metric]
    plt.plot(x_ax, y_ax)
    plt.xlabel('Episode')
    plt.ylabel(metric_title)
    plt.title(f'{metric_title} over episodes')
    plt.savefig(os.path.join(LOG_DIR, f'{metric_title}.png'))
    if VISUALIZE:
        plt.show()


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
    # plot_pc(global_map_gt, "gt_dense", "training")
    plot_pc(global_map_sparse, "gt_sparse", "training")
    print("###### Running training loop ######")

    for episode in range(EPISODES + 1):
        global_map_episode, traj_epis, depth_sample_pred = construct_map(dataset_sparse, predictor=model)
        # global_map_episode, traj_epis, depth_sample_pred = construct_map(dataset_sparse, predictor=None)

        # do backward pass
        optimizer.zero_grad()
        loss = chamfer_loss(global_map_episode, global_map_gt)
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
        if episode == EPISODES or episode % (EPISODES // 4) == 0:
            torch.save(model.state_dict(), os.path.join(LOG_DIR, f"weights-{episode}.pth"))
            plot_pc(global_map_episode, episode, "training")
            plot_depth(depth_sample_sparse, depth_sample_pred, depth_sample_gt, "training", episode)
    # END learning loop

    # plot training metrics over episodes
    plot_metric(loss_training, "Training loss")
    plot_metric(mse_training, "MSE")
    plot_metric(mae_training, "MAE")

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
        plot_pc(global_map_gt, "gt", f"testing-{subseq}")

        global_map_sparse, traj_sparse, depth_sample_sparse = construct_map(dataset_sparse)
        plot_pc(global_map_gt, "sparse", f"testing-{subseq}")

        global_map_pred, traj_pred, depth_sample_pred = construct_map(dataset_sparse, model)
        plot_pc(global_map_pred, "pred", f"testing-{subseq}")

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


def main(file=None):
    train_subseq = ["2011_09_26_drive_0001_sync"]
    test_subseqs = ["2011_09_28_drive_0191_sync"]
    assert not any(x in test_subseqs for x in train_subseq)
    model = train(train_subseq)
    test(test_subseqs, model)


if __name__ == '__main__':
    main()
    # TODO:
    #   make training over multiple subsequences
    #   Make separate singularity image for gradslam?
