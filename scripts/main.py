import os
import time
import torch
import torch.nn as nn
from gradslam import Pointclouds, RGBDImages
from gradslam.slam import PointFusion
from supervised_depth_correction.data import Dataset
from supervised_depth_correction.io import write, append
from supervised_depth_correction.utils import load_model, complete_sequence
from supervised_depth_correction.metrics import RMSE, MAE, localization_accuracy
from supervised_depth_correction.loss import chamfer_loss
from supervised_depth_correction.utils import plot_depth, plot_pc, plot_metric, metrics_dataset
from supervised_depth_correction.postprocessing import filter_depth_outliers

# ------------------------------------ GLOBAL PARAMETERS ------------------------------------ #
CUDA_DEVICE = 0
DEVICE = torch.device(f"cuda:{CUDA_DEVICE}" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device('cpu')

BATCH_SIZE = 1
LR = 0.001
WEIGHT_DECAY = 0.01

EPISODES_PER_SEQ = 60
VISUALIZE = False
VALIDATION_EPISODE = 5

USE_DEPTH_SELECTION = False
LOG_DIR = os.path.join(os.path.dirname(__file__), '..', f'config/results/depth_completion_gradslam_{time.time()}')
INIT_MODEL_STATE_DICT = os.path.realpath(os.path.join(os.path.dirname(__file__), '../config/results/weights/weights-415.pth'))

TRAIN = False
TEST = True
COMPLETE_SEQS = True    # True for creating predictions with new model
TRAIN_MODE = "mse"      # "mse" or "chamfer"


# ------------------------------------ Helper functions ------------------------------------ #
def construct_map(ds, predictor=None, pose_provider='gt', max_clouds=7, step=1, dsratio=4):
    slam = PointFusion(device=DEVICE, odom=pose_provider, dsratio=dsratio)
    prev_frame = None
    pointclouds = Pointclouds(device=DEVICE)
    trajectory = []

    # TODO: check memory issue
    depths = None
    for i in range(0, max_clouds, step):
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


def episode_mse(dataset_gt, dataset_sparse, predictor, criterion, optimizer, episode_num=0, val=False, plot=False):
    loss_episode = 0
    mae_episode = 0
    for i in range(len(dataset_gt)):
        optimizer.zero_grad()
        colors_gt, depths_gt, intrinsics_gt, poses_gt = dataset_gt[i]
        colors_sparse, depths_sparse, intrinsics_sparse, poses_sparse = dataset_sparse[i]
        mask = (depths_sparse > 0).float()
        depths_pred = predictor(depths_sparse, mask)
        loss = (criterion(depths_pred, depths_gt) * mask.detach()).sum() / mask.sum()
        if not val:
            loss.backward()
            optimizer.step()
        loss_episode += loss.detach().item()
        mae = MAE(depths_gt.detach(), depths_pred.detach())
        mae_episode += mae
    if plot:
        plot_depth(depths_sparse, depths_pred, depths_gt, "training", episode_num,
                   visualize=VISUALIZE, log_dir=LOG_DIR)
    return loss_episode / len(dataset_gt.ids), mae_episode / len(dataset_gt.ids)


def compute_val_metrics(depth_gt, depth_pred, traj_gt, traj_pred):
    rmse = RMSE(depth_gt, depth_pred)
    mae = MAE(depth_gt, depth_pred)
    loc_acc = localization_accuracy(traj_gt, traj_pred)
    return rmse, mae, loc_acc


# ------------------------------------ Learning functions------------------------------------ #
def train_chamferdist(train_subseqs, validation_subseq):
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
                plot_depth(depth_sample_sparse, depth_sample_pred, depth_sample_gt, "training", episode,
                           visualize=VISUALIZE, log_dir=LOG_DIR)
            del global_map_episode

            # validation
            if episode + 1 == num_episodes or episode % VALIDATION_EPISODE == 0:
                torch.save(model.state_dict(), os.path.join(LOG_DIR, f"weights-{episode}.pth"))
                global_map_val_episode, traj_val_epis, depth_sample_val_pred = construct_map(dataset_val_sparse,
                                                                                             predictor=model)
                loss_val = chamfer_loss(global_map_val_episode, global_map_val_gt, sample_step=5)
                loss_validation.append(loss_val.detach())
                rmse_val, mae_val, _ = compute_val_metrics(depth_sample_val_gt, depth_sample_val_pred, traj_val_gt,
                                                           traj_val_epis)
                rmse_validation.append(rmse_val.detach())
                mae_validation.append(mae_val.detach())
                print(f"EPISODE {episode}/{num_episodes}, validation loss: {loss_val}")
                append(os.path.join(LOG_DIR, 'Validation loss.txt'),
                       f"EPISODE {episode}/{num_episodes}, validation loss: {loss_val}" + '\n')
                append(os.path.join(LOG_DIR, 'Validation MAE.txt'),
                       f"EPISODE {episode}/{num_episodes}, Validation MAE: {mae_val}" + '\n')
                del global_map_val_episode

            episode += 1
        # END learning loop

        # free up space for next subseq
        del dataset_sparse, global_map_gt

    # plot training metrics over episodes
    plot_metric(loss_training, "Training loss", visualize=VISUALIZE, log_dir=LOG_DIR)
    plot_metric(loss_validation, "Validation loss", visualize=VISUALIZE, log_dir=LOG_DIR,
                val_scaling=VALIDATION_EPISODE)
    plot_metric(rmse_training, "Training RMSE", visualize=VISUALIZE, log_dir=LOG_DIR)
    plot_metric(mae_training, "Training MAE", visualize=VISUALIZE, log_dir=LOG_DIR)
    plot_metric(rmse_validation, "Validation RMSE", visualize=VISUALIZE, log_dir=LOG_DIR,
                val_scaling=VALIDATION_EPISODE)
    plot_metric(mae_validation, "Validation MAE", visualize=VISUALIZE, log_dir=LOG_DIR, val_scaling=VALIDATION_EPISODE)
    print(f"Best validation loss on episode {torch.argmin(torch.tensor(loss_validation)) * VALIDATION_EPISODE}")
    print(f"Best validation mae on episode {torch.argmin(torch.tensor(mae_validation)) * VALIDATION_EPISODE}")
    append(os.path.join(LOG_DIR, 'Validation loss.txt'),
           f"Best validation loss on episode {torch.argmin(torch.tensor(loss_validation)) * VALIDATION_EPISODE}" + '\n')
    append(os.path.join(LOG_DIR, 'Validation MAE.txt'),
           f"Best validation mae on episode {torch.argmin(torch.tensor(mae_validation)) * VALIDATION_EPISODE}" + '\n')
    return model


def train_MSE(train_subseqs, validation_subseq):
    print("###### Setting up training environment ######")
    model = load_model(INIT_MODEL_STATE_DICT)
    model = model.train()
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
    loss_training = []
    loss_validation = []
    mae_training = []
    mae_validation = []
    episode = 0

    num_episodes = EPISODES_PER_SEQ * len(train_subseqs)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    print("Loading validation sequence")
    dataset_val_gt = Dataset(subseq=validation_subseq, selection=USE_DEPTH_SELECTION, depth_type="dense",
                             device=DEVICE)
    dataset_val_sparse = Dataset(subseq=validation_subseq, selection=USE_DEPTH_SELECTION, depth_type="sparse",
                                 device=DEVICE)

    for subseq in train_subseqs:
        print(f"###### Starting training on subseq: {subseq} ######")
        print("###### Loading data ######")
        dataset_gt = Dataset(subseq=subseq, selection=USE_DEPTH_SELECTION, depth_type="dense", device=DEVICE)
        dataset_sparse = Dataset(subseq=subseq, selection=USE_DEPTH_SELECTION, depth_type="sparse", device=DEVICE)

        print("###### Running episodes  ######")
        for e in range(EPISODES_PER_SEQ):
            if episode + 1 == num_episodes or episode % (num_episodes // len(train_subseqs)) == 0:
                loss, mae = episode_mse(dataset_gt, dataset_sparse, model, criterion=criterion, episode_num=episode,
                                        optimizer=optimizer, val=False, plot=True)
            else:
                loss, mae = episode_mse(dataset_gt, dataset_sparse, model, criterion=criterion,
                                        optimizer=optimizer, val=False)
            loss_training.append(loss)
            mae_training.append(mae)
            print(f"EPISODE {episode}/{num_episodes}, loss: {loss}")

            # validation
            if episode + 1 == num_episodes or episode % VALIDATION_EPISODE == 0:
                torch.save(model.state_dict(), os.path.join(LOG_DIR, f"weights-{episode}.pth"))
                loss_val, mae_val = episode_mse(dataset_val_gt, dataset_val_sparse, model, criterion=criterion,
                                                optimizer=optimizer, val=True)
                loss_validation.append(loss_val)
                mae_validation.append(mae_val)
                print(f"EPISODE {episode}/{num_episodes}, validation loss: {loss_val}")
                append(os.path.join(LOG_DIR, 'Validation loss.txt'),
                       f"EPISODE {episode}/{num_episodes}, validation loss: {loss_val}" + '\n')
                append(os.path.join(LOG_DIR, 'Validation MAE.txt'),
                       f"EPISODE {episode}/{num_episodes}, Validation MAE: {mae_val}" + '\n')

            episode += 1
        # free up space for next subseq
        del dataset_sparse, dataset_gt
        # END learning loop

    # plot training metrics over episodes
    plot_metric(loss_training, "Training loss", visualize=VISUALIZE, log_dir=LOG_DIR)
    plot_metric(loss_validation, "Validation loss", visualize=VISUALIZE, log_dir=LOG_DIR,
                val_scaling=VALIDATION_EPISODE)
    plot_metric(mae_training, "Training MAE", visualize=VISUALIZE, log_dir=LOG_DIR)
    plot_metric(mae_validation, "Validation MAE", visualize=VISUALIZE, log_dir=LOG_DIR,
                val_scaling=VALIDATION_EPISODE)
    print(f"Best validation loss on episode {torch.argmin(torch.tensor(loss_validation)) * VALIDATION_EPISODE}")
    print(f"Best validation mae on episode {torch.argmin(torch.tensor(mae_validation)) * VALIDATION_EPISODE}")
    append(os.path.join(LOG_DIR, 'Validation loss.txt'),
           f"Best validation loss on episode {torch.argmin(torch.tensor(loss_validation)) * VALIDATION_EPISODE}" + '\n')
    append(os.path.join(LOG_DIR, 'Validation MAE.txt'),
           f"Best validation mae on episode {torch.argmin(torch.tensor(mae_validation)) * VALIDATION_EPISODE}" + '\n')
    return model


def test_loop(dataset, trajectory_gt, max_clouds, dsratio, test_mode):
    map, trajectory, depth_sample = construct_map(dataset, max_clouds=max_clouds, dsratio=dsratio, pose_provider='icp')
    loc_acc = localization_accuracy(trajectory_gt, trajectory)
    append(os.path.join(LOG_DIR, 'loc_acc_testing.txt'), f"Accuracy {test_mode}: {loc_acc}" + '\n')
    del map, trajectory, depth_sample
    return loc_acc


def test(subseqs, model=None, max_clouds=4, dsratio=4):
    """
    Test if results of trained model are comparable on other subsequences
    """
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    path_to_save = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'data', 'KITTI', 'depth', 'train'))

    for subseq in subseqs:
        print(f"\n###### Starting testing of subseq: {subseq} ######")
        append(os.path.join(LOG_DIR, 'loc_acc_testing.txt'),
               f"\ntesting of subseq: {subseq}" + '\n')

        dataset_sparse = Dataset(subseq=subseq, selection=USE_DEPTH_SELECTION, depth_type="sparse", device=DEVICE)
        if model is not None:
            print(f"###### Running depth completion on depth data ######")
            complete_sequence(model, dataset_sparse, path_to_save, subseq, replace=COMPLETE_SEQS)
        trajectory_gt = dataset_sparse.get_gt_poses()
        trajectory_gt = torch.squeeze(trajectory_gt, 0)
        trajectory_gt = torch.squeeze(trajectory_gt, 0)
        trajectory_gt = list(trajectory_gt[:max_clouds, :, :])

        locc_acc_sparse = test_loop(dataset_sparse, trajectory_gt, max_clouds, dsratio, "Sparse")
        print(f"Localization accuracy sparse: {locc_acc_sparse}")
        del dataset_sparse

        dataset_gt = Dataset(subseq=subseq, selection=USE_DEPTH_SELECTION, depth_type="dense", device=DEVICE)
        locc_acc_gt = test_loop(dataset_gt, trajectory_gt, max_clouds, dsratio, "Groundruth")
        print(f"Localization accuracy groundtruth: {locc_acc_gt}")
        del dataset_gt

        dataset_pred = Dataset(subseq=subseq, selection=USE_DEPTH_SELECTION, depth_type="pred", device=DEVICE)
        locc_acc_pred = test_loop(dataset_pred, trajectory_gt, max_clouds, dsratio, "Prediction")
        print(f"Localization accuracy prediction: {locc_acc_pred}")
        del dataset_pred, trajectory_gt

        dataset_pred = Dataset(subseq=subseq, selection=USE_DEPTH_SELECTION, depth_type="pred", device=DEVICE)
        dataset_gt = Dataset(subseq=subseq, selection=USE_DEPTH_SELECTION, depth_type="dense", device=DEVICE)
        print(f"###### Computing metrics ######")
        mae, rmse = metrics_dataset(dataset_gt, dataset_pred)
        print(mae, rmse)
        append(os.path.join(LOG_DIR, 'loc_acc_testing.txt'), f"MAE {mae}, RMSE {rmse}" + '\n')
        del dataset_pred, dataset_gt

    print("###### TESTING DONE ######")


def main():
    train_subseqs = ["2011_09_26_drive_0086_sync", "2011_09_28_drive_0187_sync", "2011_09_28_drive_0087_sync",
                     "2011_09_28_drive_0122_sync", "2011_09_26_drive_0051_sync", "2011_10_03_drive_0034_sync",
                     "2011_09_28_drive_0094_sync", "2011_09_30_drive_0018_sync", "2011_09_28_drive_0095_sync",
                     "2011_09_26_drive_0117_sync", "2011_09_26_drive_0057_sync", "2011_09_28_drive_0075_sync",
                     "2011_09_28_drive_0145_sync", "2011_09_28_drive_0220_sync", "2011_09_26_drive_0101_sync",
                     "2011_09_28_drive_0098_sync", "2011_09_28_drive_0167_sync", "2011_10_03_drive_0042_sync",
                     "2011_09_26_drive_0027_sync", "2011_09_28_drive_0198_sync", "2011_09_26_drive_0011_sync",
                     "2011_09_26_drive_0096_sync", "2011_09_28_drive_0171_sync", "2011_09_30_drive_0018_sync",
                     "2011_09_28_drive_0141_sync"]
    validation_subseq = "2011_09_28_drive_0168_sync"
    test_subseqs = ["2011_09_26_drive_0009_sync", "2011_09_26_drive_0001_sync", "2011_09_26_drive_0018_sync",
                    "2011_10_03_drive_0027_sync"]
    assert not any(x in test_subseqs for x in train_subseqs)
    assert validation_subseq not in train_subseqs and validation_subseq not in test_subseqs
    if TRAIN:
        if TRAIN_MODE == "mse":
            model = train_MSE(train_subseqs, validation_subseq)
        elif TRAIN_MODE == "chamfer":
            model = train_chamferdist(train_subseqs, validation_subseq)
        else:
            print("INVALID TRAIN MODE!!!")
    elif TEST and COMPLETE_SEQS:
        model = load_model(INIT_MODEL_STATE_DICT)
        model = model.to(DEVICE)
        model = model.eval()
    else:
        model = None
    if TEST:
        test(test_subseqs, model=model, max_clouds=30, dsratio=4)


if __name__ == '__main__':
    main()
    # TODO:
    #   improve point/depth image cloud filtering
