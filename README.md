# voxl-tflite-server

How to use tensorflow-lite on VOXL

High Level Overview
===================
The following picture shows at a glance how the different components interact with each other.

![](images/voxl-tflite-server.png)

Build steps
===========
1. (PC) Build the voxl-cross docker from [here](https://gitlab.com/voxl-public/utilities/voxl-docker)
    * ./install-cross-docker.sh
1. (PC) mkdir my-git-source-code
1. (PC) cd my-git-source-code
1. (PC) git clone git@gitlab.com:voxl-public/modal-pipe-architecture/voxl-tflite-server.git
1. (PC) cd <path-to>/voxl-tflite-server/
1. (PC) voxl-docker -i voxl-cross
1. (PC_CROSS_DOCKER) ./install_build_deps.sh
1. (PC_CROSS_DOCKER) ./clean.sh
1. (PC_CROSS_DOCKER) ./build.sh
1. (PC_CROSS_DOCKER) ./make_package.sh

Steps to run
============
1. (PC) cd <path-to>/voxl-tflite-server/
1. (PC) ./install_on_voxl.sh
1. (PC) adb shell
1. (VOXL-Terminal-1) export ROS_IP=IP-ADDRESS-OF-VOXL
1. (VOXL-Terminal-1) source /opt/ros/indigo/setup.bash
1. (VOXL-Terminal-1) roscore
    * It should print a line that looks something like: ROS_MASTER_URI=http://AAA.BBB.CCC.DDD:XYZW/
1. (PC) source /opt/ros/kinetic/setup.bash
1. (PC) export ROS_IP=IP-ADDRESS-OF-PC
1. (PC) source /opt/ros/xxxx/setup.bash
1. (PC) export ROS_MASTER_URI=http://AAA.BBB.CCC.DDD:XYZW/
1. (PC) rviz
    - Click on "Add" at the bottom-left
    - Select "Image"
    - Change Display Name to "My-Camera-Image"
    - On the left column expand the "My-Camera-Image"
    - Click on the "Image Topic" and in the right column enter "/voxl_hires_image" 
1. (PC) adb shell
1. (VOXL-Terminal-2) voxl-camera-server
1. (PC) adb shell
1. (VOXL-Terminal-3) voxl-tflite-server -m mobilenet
1. (PC) adb shell
1. (PC) source /opt/ros/indigo/setup.bash
1. (VOXL-Terminal-4) vi /opt/ros/indigo/share/voxl_mpa_cam_ros/launch/voxl_mpa_cam_ros.launch
    * Change line 15 to the new pipe name "default="/run/mpa/tflite/image/"    
1. (VOXL-Terminal-4) python /usr/bin/launch_voxl_mpa_cam_ros.py
1. In order to run the pydnet model
    * (VOXL) voxl-tflite-server -m pydnet
    * (VOXL) Check the outputs in the /usr/bin/dnn/data directory
    * (VOXL) The output files are the ones that have "-depth" in the filename
