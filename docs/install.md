# Installation

1. Install [ROS](http://wiki.ros.org/ROS/Installation)
2. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html)
3. Setup python environment:
   ```bash
   git clone https://github.com/RuslanAgishev/supervised_depth_correction.git
   cd supervised_depth_correction
   conda activate my_env
   conda env update -n my_env --file environment.yaml
   ```
4. (optional) Build the ROS wrapper for the package:
   ```bash
   cd ./ros_ws/
   catkin build gradslam_ros
   source devel/setup.bash
   ```
