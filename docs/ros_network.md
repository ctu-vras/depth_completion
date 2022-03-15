# ROS network

Example of setting up the network with ROS master running on server machine
([CMP grid](https://cmp.felk.cvut.cz/cmp/hardware/grid/))
and subscribing to the published data locally.

Reference: http://wiki.ros.org/ROS/Tutorials/MultipleMachines

- On the server machine (taylor.felk.cvut.cz):

1. Clone the package:
```bash
mkdir -p ~/catkin_ws/src/ && cd ~/catkin_ws/src/
git clone https://github.com/RuslanAgishev/supervised_depth_correction.git
```
For data transferring between computers on the same network, you can use
[scp](https://linuxize.com/post/how-to-use-scp-command-to-securely-transfer-files/):
```bash
# locally: assumes you have the packages build in the workspace
roscd supervised_depth_correction
scp -r data/ username@taylor.felk.cvut.cz:/home.stud/username/catkin_ws/src/supervised_depth_correction/
```

2. Set ROS environment variables for networking (you can choose the port from those available, the ROS default 11311 should be avoided):
```bash
port=11411
export ROS_IP=$(ip -4 addr | grep -oP '(?<=inet\s)147\.32\.\d+\.\d+')
echo ${ROS_IP}
export ROS_MASTER_URI=http://${ROS_IP}:${port}
export IGN_PARTITION=${ROS_IP}:${USER}:${port} 
```
P.S.: For the taylor machine the `ROS_IP` should be printed as `147.32.84.27`.

3. Run the ROS package from singularity image:
```bash
cd ~/catkin_ws/src/supervised_depth_correction/singularity/

module load Singularity
singularity shell --nv --bind /home.stud/username/catkin_ws/src/supervised_depth_correction/:/opt/ros/depthcorr_ws/src/supervised_depth_correction/ supervised_depth_correction.simg

source /opt/ros/noetic/setup.bash
source /opt/ros/depthcorr_ws/devel/setup.bash

roslaunch supervised_depth_correction demo.launch rviz:=False pub_gt:=True
```

- Locally run Rviz with proper configuration file (IP address should be replaced with that printed out on the server by `echo ${ROS_IP}`):

```ROS_MASTER_URI=http://147.32.84.27:11411 rviz -d ~/catkin_ws/src/supervised_depth_correction/rviz/config.rviz```
