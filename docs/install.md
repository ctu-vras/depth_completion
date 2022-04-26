# Installation

## Locally

1. Install [ROS](http://wiki.ros.org/ROS/Installation)
2. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html)
3. Setup python environment:
   ```bash
   mkdir -p ~/catkin_ws/src/
   cd ~/catkin_ws/src/
   git clone https://github.com/RuslanAgishev/supervised_depth_correction.git
   cd supervised_depth_correction
   conda activate my_env
   conda env update -n my_env --file environment.yaml
   ```
4. Build the ROS package:
   ```bash
   cd ~/catkin_ws/
   catkin build supervised_depth_correction
   source devel/setup.bash
   ```

## Singularity image

1. Install [Singularity](https://github.com/RuslanAgishev/supervised_depth_correction/blob/main/docs/singularity.md).

2. Building Singularity image.
   
   One option could be to use the prebuilt singularity image.
   Download it from [here](https://drive.google.com/drive/folders/1sLJKLenocEIsqBZJy4HELBphzaQNVjbe?usp=sharing).

   If you would like to build a singularity image yourself,
   please do the following:

   ```bash
   cd ../singularity
   sudo singularity build supervised_depth_correction.simg supervised_depth_correction.txt
   ```

3. Run demo.

   Ones, you have the singularity image build, it would be possible to run the package inside the environment as follows.
   In the example bellow, first, we bind the up to date package with data to our image.
   Then we source the ROS workspace inside the image.
   The next step is to launch the demo on the provided [bag file](https://drive.google.com/file/d/1kFbH38nbsHm7UR1B9Du3A0BcjLG1CiSR/view?usp=sharing).

   ```bash
   singularity shell --nv --bind /full/path/to/supervised_depth_correction/data/:/opt/ros/depthcorr_ws/src/supervised_depth_correction/data/ supervised_depth_correction.simg

   source /opt/ros/noetic/setup.bash
   source /opt/ros/depthcorr_ws/devel/setup.bash

   roslaunch supervised_depth_correction demo.launch rviz:=True
   ```
