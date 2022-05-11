import os
import matplotlib.pyplot as plt
import numpy as np


def load_res(file, scaling_factor=1):
    """
    Loads results from a file
    We expect results in the following format:
    'EPISODE NUM_EPISODE/NUM_EPISODES, Metric type: XX.XXXX'
    metric_factor: <int> - scaling of metric values, should be 1 for chamfer loss and MSE, and 1000 for MAE and RMSE
    """
    # parce lines
    episodes = []
    values = []
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.rstrip().split()
            # pass lines in different format
            if line_list[0] != "EPISODE":
                continue
            episode_num = int(line_list[1].split("/")[0])
            value = float(line_list[-1]) * scaling_factor
            episodes.append(episode_num)
            values.append(value)
    return values, episodes


def plot_training_res(result_folder):
    """
    Result_folder: <os.path> path to folder with following results:
    Training loss
    Training MAE
    Training RMSE
    Validation loss
    Validation MAE
    Validation RMSE
    """

    training_loss, episodes_trn = load_res(os.path.join(result_folder, "Training loss.txt"))
    validation_loss, episodes_val = load_res(os.path.join(result_folder, "Validation loss.txt"))
    # MAE and RMSE are multiplied by 1000 to be in millimeters
    training_MAE, episodes_trn_MAE = load_res(os.path.join(result_folder, "Training loss.txt"), scaling_factor=1000)
    validation_MAE, episodes_val_MAE = load_res(os.path.join(result_folder, "Validation loss.txt"), scaling_factor=1000)
    training_RMSE, episodes_trn_RMSE = load_res(os.path.join(result_folder, "Training loss.txt"), scaling_factor=1000)
    validation_RMSE, episodes_val_RMSE = load_res(os.path.join(result_folder, "Validation loss.txt"), scaling_factor=1000)

    # plot lines
    fig = plt.figure()
    plt.plot(episodes_trn, training_loss, label="Training loss", linewidth=2.5)
    plt.plot(episodes_val, validation_loss, label="Validation loss", linewidth=2.5)
    plt.title("Chamfer loss over episodes")
    plt.xlabel('Episodes')
    plt.ylabel('Chamfer loss [m]')
    plt.legend()
    plt.grid()
    plt.show()
    plt.close(fig)

    fig = plt.figure()
    plt.plot(episodes_trn_MAE, training_MAE, label="Training MAE", linewidth=2.5)
    plt.plot(episodes_val_MAE, validation_MAE, label="Validation MAE", linewidth=2.5)
    plt.title("MAE over episodes")
    plt.xlabel('Episodes')
    plt.ylabel('MAE [mm]')
    plt.legend()
    plt.grid()
    plt.show()
    plt.close(fig)

    fig = plt.figure()
    plt.plot(episodes_trn_RMSE, training_RMSE, label="Training RMSE", linewidth=2.5)
    plt.plot(episodes_val_RMSE, validation_RMSE, label="Validation RMSE", linewidth=2.5)
    plt.title("RMSE over episodes")
    plt.xlabel('Episodes')
    plt.ylabel('RMSE [mm]')
    plt.legend()
    plt.grid()
    plt.show()
    plt.close(fig)


def main():
    # plot results for Chamfer loss trained on 8 frames
    plot_training_res("../results/training/Chamfer/map_from_single_frame")
    plot_training_res("../results/training/Chamfer/map_from_multiple_frames")
    plot_training_res("../results/training/MSE")


if __name__ == '__main__':
    main()
