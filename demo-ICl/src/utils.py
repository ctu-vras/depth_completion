import torch
import cv2
import numpy as np
import glob
from DepthCorrectionModule import LinTransModel, ModelV1
from gradslam.datasets import ICL
from torch.utils.data import DataLoader


def denoise_image(img_n, model, weights_path):
    """
    Loads weights into model, runs image through model and saves result
    """
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model = model.eval()
    img = model(img_n)
    img = torch.squeeze(img, 0)
    img = torch.squeeze(img, 0)
    print(img.shape)
    img = img.cpu().detach().numpy()
    img = np.repeat(img, 3, axis=2)*255
    print(img.shape)
    print(img)
    cv2.imwrite("denoised.png", img)


def noise_img(img, deviation=1):
    """
    :param img: <np.ndarray> of shape N x M x 3
    :param deviation: <int> standart deviation of noise
    :return: <np.ndarray> of shape N x M x 3 with added gaussian noise
    """
    N, M = img.shape[0:2]
    mean = 0
    noise = np.random.normal(mean, deviation, (N, M, 1))
    # noise = np.repeat(noise, 3, axis=2)
    return img + noise


def noise_data(path, noise_path):
    """
    Adds noise to images in path folder and saves results to noise_data_path folder
    :param path: <string> path to folder containing depth data
    :param noise_path: <string> path to save noised images
    :return: None
    """
    path = path + "/"
    deviation = 5
    img_names = image_names = glob.glob(path+"*.png")

    for i in range(len(img_names)):
        img_name = image_names[i]
        img = cv2.imread(img_name)
        img = noise_img(img, deviation)
        img_noise_name = noise_path+'/'+str(i)+'.png'
        cv2.imwrite(img_noise_name, img)


if __name__ == '__main__':
    """path = '/home/jachym/MEGAsync/KYR/Bakalarka/ICL-dataset/living_room_traj2_frei_png/depth'
    path_noise = '/home/jachym/MEGAsync/KYR/Bakalarka/ICL-noise/living_room_traj2_frei_png/depth'
    noise_data(path, path_noise)"""

    img_n_path = "/home/jachym/MEGAsync/KYR/Bakalarka/ICL-noise/living_room_traj2_frei_png/depth/0.png"
    #weights_path = "/home/jachym/MEGAsync/KYR/Bakalarka/src/results/3kernelNet/weights50.pth"
    weights_path = "/home/jachym/MEGAsync/KYR/Bakalarka/src/results/LinearTransformation/weights40.pth"
    path_noisy_data = "/home/jachym/MEGAsync/KYR/Bakalarka/ICL-noise/"
    i = cv2.imread(img_n_path)
    print(i.shape)
    print(i)

    dataset_noise = ICL(path_noisy_data, seqlen=1)
    loader_n = iter(DataLoader(dataset=dataset_noise, batch_size=1))
    colors_n, depths_n, intrinsics_n, poses_n, *_ = next(loader_n)
    depths_n = torch.squeeze(depths_n[..., 0, :], -4)
    depths_n = torch.unsqueeze(depths_n, 0)
    print(depths_n.shape)
    #model = ModelV1()
    model = LinTransModel()
    denoise_image(depths_n, model, weights_path)

