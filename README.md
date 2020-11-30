# voxl-mpa-tflite-server

How to use tensorflow-lite on VOXL

High Level Overview
===================
The following picture shows at a glance how the different components interact with each other.

Build steps
===========
1. (PC) Get the voxl-cross64 docker from [here]
1. (PC) mkdir my-git-source-code
1. (PC) cd my-git-source-code
1. (PC) git clone XXXXXX
1. (PC) cd <path-to>/voxl-mpa-tflite-server/
1. (PC) Download build-dependencies.tar.gz from [here]
1. (PC) Save build-dependencies.tar.gz in the "<path-to>/voxl-mpa-tflite-server/modalai" directory
1. (PC) sudo docker run -v $PWD:/opt/data/workspace/ -it voxl-cross64
1. (PC_CROSS64_DOCKER) cd /opt/data/workspace
1. (PC_CROSS64_DOCKER) ./install_build_deps.sh
1. (PC_CROSS64_DOCKER) ./build_aarch64.sh
1. (PC_CROSS64_DOCKER) ./make_package.sh

Steps to run
============
1. (PC) Download 64-bit opencv-4-3-0 and gpu-libs IPK from [here]
1. (PC) Save downloaded IPK file in the "<path-to>/voxl-mpa-tflite/modalai" directory
1. (PC) cd <path-to>/voxl-mpa-tflite-server/
1. (PC) ./install_on_voxl.sh
1. (PC) adb shell
1. (VOXL) cd /bin/dnn
1. (VOXL) voxl-mpa-tflite -m pydnet
    - Check the outputs in the /bin/dnn directories
1. (PC) Install the voxl-camera-server IPK on VOXL from [here]
1. (VOXL-Terminal-1) voxl-camera-server -c /etc/modalai/voxl-camera-server.conf
1. (VOXL-Terminal-2) voxl-mpa-tflite -m mobilenet
    - Ctrl+C Terminal-1
    - Ctrl+C Terminal-2
