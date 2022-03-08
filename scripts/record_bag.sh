#! /bin/bash

rosbag record /X1/front_rgbd/optical/camera_info \
			  /X1/front_rgbd/optical/image_raw \
			  /X1/front_rgbd/depth/optical/image_raw \
			  /tf tf_static
