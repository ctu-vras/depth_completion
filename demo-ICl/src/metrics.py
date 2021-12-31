import numpy as np
import cv2
import torch


def mse_single_img(img1, img2):
    """
    Computes 'Mean Squared Error' between the two images
    :param img1: <np.ndarray>
    :param img2: <np.ndarray>
    :return: <float>
    """
    assert img1.shape == img2.shape
    err = np.sum( (img1.astype("float") - img2.astype("float")) ** 2 )
    err /= float(img1.shape[0] * img2.shape[1] * img2.shape[2])
    return err


def mse_depth_dataset(depth1, depth2):
    """
    Computes mean of 'Mean Squared Error' between each two images in the two input datasets
    :param depth1, depth2: <list of torch.tensor> sets of depth images to compare
    """
    assert len(depth1) == len(depth2)
    err = 0
    for i in range(len(depth1)):
        im1 = depth1[i]
        im2 = depth2[i]
        err += mse_single_img(im1.cpu().detach().numpy(), im2.cpu().detach().numpy())
    return err/len(depth1)


if __name__ == '__main__':
    img1 = cv2.imread('/home/jachym/MEGAsync/KYR/Bakalarka/ICL-dataset/living_room_traj2_frei_png/depth/2.png')
    img2 = cv2.imread('/home/jachym/MEGAsync/KYR/Bakalarka/ICL-noise/living_room_traj2_frei_png/depth/2.png')
    print(mse_single_img(img1, img2))
