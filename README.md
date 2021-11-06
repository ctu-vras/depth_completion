## Using the data from Subt simulator

Download RGBD images from:
[https://drive.google.com/drive/folders/1GuZr6nvmH1_-31vtszih9-CQowisk0VD?usp=sharing](https://drive.google.com/drive/folders/1GuZr6nvmH1_-31vtszih9-CQowisk0VD?usp=sharing)

And place it to the folder:
```
./ros_ws/src/gradslam_ros/data/
```

Explore the depth images data from the simulator (requires
[Open3D](https://github.com/isl-org/Open3D)
installation):
[./notebooks/explore_data.ipynb](https://github.com/RuslanAgishev/supervised_depth_correction/blob/master/notebooks/explore_data.ipynb)

## Using GradSLAM to construct a map

Prerequisite: install [ROS](https://www.ros.org/)

Build the `gradslam_ros` wrapper node:
```bash
cd ./ros_ws/
catkin_make
source devel/setup.bash
```

Construct a map from RGBD images input:
```
roslaunch gradslam_ros bag_inference.launch odom:=gt
```
