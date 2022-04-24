import os
import time
import torch
from gradslam import Pointclouds, RGBDImages
from gradslam.slam import PointFusion
from supervised_depth_correction.data import Dataset
from supervised_depth_correction.io import write, append
from supervised_depth_correction.models import SparseConvNet
from supervised_depth_correction.metrics import RMSE, MAE, chamfer_loss, localization_accuracy
from supervised_depth_correction.utils import plot_depth, plot_pc, plot_metric


# ------------------------------------ GLOBAL PARAMETERS ------------------------------------ #
CUDA_DEVICE = 0
DEVICE = torch.device(f"cuda:{CUDA_DEVICE}" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device('cpu')

BATCH_SIZE = 1
LR = 0.001
WEIGHT_DECAY = 0.01

EPISODES_PER_SEQ = 50
VISUALIZE = False
VALIDATION_EPISODE = 5

USE_DEPTH_SELECTION = False
LOG_DIR = os.path.join(os.path.dirname(__file__), '..', f'config/results/depth_completion_gradslam_{time.time()}')
INIT_MODEL_STATE_DICT = os.path.join(os.path.dirname(__file__), 'weights.pth')

TRAIN = False
TEST = True


# ------------------------------------ Helper functions ------------------------------------ #
def construct_map(ds, predictor=None, pose_provider='gt', max_clouds=8, step=1):
    slam = PointFusion(device=DEVICE, odom=pose_provider, dsratio=1)
    prev_frame = None
    pointclouds = Pointclouds(device=DEVICE)
    trajectory = []

    # TODO: check memory issue
    depths = None
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
    rmse = RMSE(depth_gt, depth_pred)
    mae = MAE(depth_gt, depth_pred)
    loc_acc = localization_accuracy(traj_gt, traj_pred)
    return rmse, mae, loc_acc


def load_model(path=None):
    model = SparseConvNet()
    if path:
        if os.path.exists(path):
            print('Loading model weights from %s' % path)
            model.load_state_dict(torch.load(path, map_location='cpu'))
        else:
            print('No model weights found!!!')
    return model


# ------------------------------------ Learning functions------------------------------------ #
def train(train_subseqs, validation_subseq):
    print("###### Setting up training environment ######")
    model = load_model(INIT_MODEL_STATE_DICT)
    model = model.train()
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_training = []
    loss_validation = []
    rmse_training = []
    mae_training = []
    rmse_validation = []
    mae_validation = []
    episode = 0
    num_episodes = EPISODES_PER_SEQ * len(train_subseqs)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    for subseq in train_subseqs:
        print(f"###### Starting training on subseq: {subseq} ######")
        print("###### Loading data ######")
        dataset_gt = Dataset(subseq=subseq, selection=USE_DEPTH_SELECTION, gt=True, device=DEVICE)
        dataset_sparse = Dataset(subseq=subseq, selection=USE_DEPTH_SELECTION, gt=False, device=DEVICE)
        dataset_val_gt = Dataset(subseq=validation_subseq, selection=USE_DEPTH_SELECTION, gt=True, device=DEVICE)
        dataset_val_sparse = Dataset(subseq=validation_subseq, selection=USE_DEPTH_SELECTION, gt=False, device=DEVICE)
        print("###### Data loaded ######")

        # create gt global map
        global_map_gt, traj_gt, depth_sample_gt = construct_map(dataset_gt)
        del dataset_gt
        *_, depth_sample_sparse = construct_map(dataset_sparse)
        global_map_val_gt, traj_val_gt, depth_sample_val_gt = construct_map(dataset_val_gt)
        del dataset_val_gt
        plot_pc(global_map_gt, "gt_dense", "training", visualize=VISUALIZE, log_dir=LOG_DIR)
        # plot_pc(global_map_sparse, "gt_sparse", "training", visualize=VISUALIZE, log_dir=LOG_DIR)
        print("###### Running training loop ######")

        for e in range(EPISODES_PER_SEQ):
            global_map_episode, traj_epis, depth_sample_pred = construct_map(dataset_sparse, predictor=model)

            # do backward pass
            optimizer.zero_grad()
            loss = chamfer_loss(global_map_episode, global_map_gt, sample_step=5)
            loss.backward()
            optimizer.step()

            # print and append training metrics
            loss_training.append(loss.detach())
            print(f"EPISODE {episode}/{num_episodes}, loss: {loss}")
            rmse, mae, _ = compute_val_metrics(depth_sample_gt, depth_sample_pred, traj_gt, traj_epis)
            rmse_training.append(rmse.detach())
            mae_training.append(mae.detach())

            # running results save
            if episode + 1 == num_episodes or episode % (num_episodes // len(train_subseqs)) == 0:
                plot_pc(global_map_episode, episode, "training", visualize=VISUALIZE, log_dir=LOG_DIR)
                plot_depth(depth_sample_sparse, depth_sample_pred, depth_sample_gt, "training", episode, visualize=VISUALIZE, log_dir=LOG_DIR)
            del global_map_episode

            # validation
            if episode + 1 == num_episodes or episode % VALIDATION_EPISODE == 0:
                torch.save(model.state_dict(), os.path.join(LOG_DIR, f"weights-{episode}.pth"))
                global_map_val_episode, traj_val_epis, depth_sample_val_pred = construct_map(dataset_val_sparse, predictor=model)
                loss_val = chamfer_loss(global_map_val_episode, global_map_val_gt, sample_step=5)
                loss_validation.append(loss_val.detach())
                rmse_val, mae_val, _ = compute_val_metrics(depth_sample_val_gt, depth_sample_val_pred, traj_val_gt, traj_val_epis)
                rmse_validation.append(rmse_val.detach())
                mae_validation.append(mae_val.detach())
                print(f"EPISODE {episode}/{num_episodes}, validation loss: {loss_val}")
                append(os.path.join(LOG_DIR, 'Validation loss.txt'),
                       f"EPISODE {episode}/{num_episodes}, validation loss: {loss_val}" + '\n')
                del global_map_val_episode

            episode += 1
        # END learning loop

        # free up space for next subseq
        del dataset_sparse, global_map_gt

    # plot training metrics over episodes
    plot_metric(loss_training, "Training loss", visualize=VISUALIZE, log_dir=LOG_DIR)
    plot_metric(loss_validation, "Validation loss", visualize=VISUALIZE, log_dir=LOG_DIR, val_scaling=VALIDATION_EPISODE)
    plot_metric(rmse_training, "Training RMSE", visualize=VISUALIZE, log_dir=LOG_DIR)
    plot_metric(mae_training, "Training MAE", visualize=VISUALIZE, log_dir=LOG_DIR)
    plot_metric(rmse_validation, "Validation RMSE", visualize=VISUALIZE, log_dir=LOG_DIR, val_scaling=VALIDATION_EPISODE)
    plot_metric(mae_validation, "Validation MAE", visualize=VISUALIZE, log_dir=LOG_DIR, val_scaling=VALIDATION_EPISODE)

    return model


def test(subseqs, model):
    """
    Test if results of trained model are comparable on other subsequences
    """
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    max_clouds = 4

    for subseq in subseqs:
        print(f"###### Starting testing of subseq: {subseq} ######")
        append(os.path.join(LOG_DIR, 'loc_acc_testing.txt'),
               f"testing of subseq: {subseq}" + '\n')
        print("###### Loading data ######")
        dataset_gt = Dataset(subseq=subseq, selection=USE_DEPTH_SELECTION, gt=True, device=DEVICE)
        dataset_sparse = Dataset(subseq=subseq, selection=USE_DEPTH_SELECTION, gt=False, device=DEVICE)
        print("###### Data loaded ######")

        print("###### Constructing maps ######")
        print("GT")
        global_map_gt, traj_gt, depth_sample_gt = construct_map(dataset_gt, max_clouds=max_clouds)
        del dataset_gt
        plot_pc(global_map_gt, "gt", f"testing-{subseq}", visualize=VISUALIZE, log_dir=LOG_DIR)
        print("Sparse")
        global_map_sparse, traj_sparse, depth_sample_sparse = construct_map(dataset_sparse, max_clouds=max_clouds, pose_provider='icp')
        plot_pc(global_map_gt, "sparse", f"testing-{subseq}", visualize=VISUALIZE, log_dir=LOG_DIR)

        print("###### Running testing ######")

        mse_sparse, mae_sparse, locc_acc_sparse = compute_val_metrics(depth_sample_gt, depth_sample_sparse, traj_gt, traj_sparse)
        chamfer_dist = chamfer_loss(global_map_gt, global_map_sparse, sample_step=5)
        print(f"MSE sparse: {mse_sparse}")
        print(f"MAE sparse: {mae_sparse}")
        print(f"Localization accuracy sparse: {locc_acc_sparse}")
        print(f"Chamfer distance sparse: {chamfer_dist}")
        append(os.path.join(LOG_DIR, 'loc_acc_testing.txt'), f"MSE sparse: {mse_sparse}" + '\n')
        append(os.path.join(LOG_DIR, 'loc_acc_testing.txt'), f"MAE sparse: {mae_sparse}" + '\n')
        append(os.path.join(LOG_DIR, 'loc_acc_testing.txt'), f"Chamfer distance sparse: {chamfer_dist}" + '\n')
        append(os.path.join(LOG_DIR, 'loc_acc_testing.txt'), f"Localization accuracy sparse: {locc_acc_sparse}" + '\n')

        print("Pred")
        del global_map_sparse
        global_map_pred, traj_pred, depth_sample_pred = construct_map(dataset_sparse, model, max_clouds=max_clouds,
                                                                      pose_provider='icp')
        plot_pc(global_map_pred, "pred", f"testing-{subseq}", visualize=VISUALIZE, log_dir=LOG_DIR)
        mse_pred, mae_pred, locc_acc_pred = compute_val_metrics(depth_sample_gt, depth_sample_pred, traj_gt, traj_pred)
        chamfer_dist = chamfer_loss(global_map_gt, global_map_pred, sample_step=5)
        print(f"MSE pred: {mse_pred}")
        print(f"MAE pred: {mae_pred}")
        print(f"Localization accuracy pred: {locc_acc_pred}")
        print(f"Chamfer distance pred: {chamfer_dist}")
        append(os.path.join(LOG_DIR, 'loc_acc_testing.txt'), f"-----------\nMSE pred: {mse_pred}" + '\n')
        append(os.path.join(LOG_DIR, 'loc_acc_testing.txt'), f"MAE pred: {mae_pred}" + '\n')
        append(os.path.join(LOG_DIR, 'loc_acc_testing.txt'), f"Chamfer distance pred: {chamfer_dist}" + '\n')
        append(os.path.join(LOG_DIR, 'loc_acc_testing.txt'), f"Localization accuracy pred: {locc_acc_pred}" + '\n\n\n')

        del dataset_sparse, global_map_gt, global_map_pred

    print("###### TESTING DONE ######")


def main():
    train_subseqs = ["2011_09_26_drive_0086_sync", "2011_09_26_drive_0009_sync", "2011_09_28_drive_0187_sync",
                     "2011_09_28_drive_0122_sync", "2011_09_26_drive_0051_sync", "2011_10_03_drive_0034_sync",
                     "2011_09_28_drive_0094_sync", "2011_09_30_drive_0018_sync", "2011_09_28_drive_0095_sync",
                     "2011_09_26_drive_0117_sync", "2011_09_26_drive_0057_sync", "2011_09_28_drive_0075_sync",
                     "2011_09_28_drive_0145_sync", "2011_09_28_drive_0220_sync", "2011_09_26_drive_0101_sync",
                     "2011_09_28_drive_0098_sync", "2011_09_28_drive_0167_sync", "2011_10_03_drive_0042_sync",
                     "2011_09_26_drive_0027_sync", "2011_09_28_drive_0198_sync", "2011_09_26_drive_0011_sync"]
    validation_subseq = "2011_09_28_drive_0168_sync"
    test_subseqs = ["2011_09_26_drive_0001_sync", "2011_09_26_drive_0018_sync"]
    assert not any(x in test_subseqs for x in train_subseqs)
    assert validation_subseq not in train_subseqs and validation_subseq not in test_subseqs
    if TRAIN:
        model = train(train_subseqs, validation_subseq)
    else:
        model = load_model(os.path.join("..", "config", "results", "weights", "weights-980.pth"))
        model = model.to(DEVICE)
        model = model.eval()
    if TEST:
        test(test_subseqs, model)


if __name__ == '__main__':
    main()
    # TODO:
    #   Make separate singularity image for gradslam?
    #   --> train on server with gt and test locally with gradicp
