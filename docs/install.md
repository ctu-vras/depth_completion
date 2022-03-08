# Installation

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
