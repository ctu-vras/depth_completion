from time import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from gradslam.datasets.icl import ICL
from gradslam.datasets.tum import TUM
from gradslam.slam.pointfusion import PointFusion
from gradslam.structures.rgbdimages import RGBDImages
from gradslam.structures.pointclouds import Pointclouds

dataset_path = '/home/ruslan/subt/thirdparty/gradslam/examples/tutorials/ICL/'

if __name__ == "__main__":
    print('Loading the dataset...')
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dataset = ICL(dataset_path, seqlen=1, height=300, width=300)
    loader = DataLoader(dataset=dataset, batch_size=1)

    print('Initializing SLAM...')
    slam = PointFusion(odom='gradicp', device=device)
    pointcloud = Pointclouds(device=device)

    num_frames = 100
    time_elapsed = 0.0

    prev_frame = None

    print('Run...')
    for _ in tqdm(range(num_frames)):
        start = time()
        poses = torch.eye(4).view(1, 1, 4, 4) if prev_frame is None else None
        colors, depths, intrinsics, *_ = next(iter(loader))
        current_frame = RGBDImages(colors, depths, intrinsics, poses, device=device)
        pointcloud, current_frame.poses = slam.step(pointcloud, current_frame, prev_frame)
        prev_frame = current_frame
        time_elapsed += time() - start

    print('FPS', num_frames / time_elapsed)
