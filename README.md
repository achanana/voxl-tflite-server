# voxl-tflite-server

Use mobilenet for object detection through tensorflow lite on VOXL.

dependencies:
* libmodal_pipe
* libmodal_json
* opencv
* voxl-tflite

This README covers building this package.

## Build Instructions
===========

1) prerequisite: latest voxl-cross docker image

Follow the instructions here:

https://gitlab.com/voxl-public/voxl-docker


2) Launch Docker and make sure this project directory is mounted inside the Docker.

```bash
~/git/voxl-tflite-server# voxl-docker -i voxl-cross
bash-4.3$ ls
README.md         clean.sh  include                ipk              service
bash_completions  config    install_build_deps.sh  make_package.sh
build.sh          dnn       install_on_voxl.sh     server

3) Install dependencies inside the docker. Specify the dependencies should be pulled from either the development (dev) or stable modalai package repos. If building the master branch you should specify `stable`, otherwise `dev`.

```bash
./install_build_deps.sh stable
```

4) Compile inside the docker.

```bash
./build.sh
```

5) Make an ipk package inside the docker.

```bash
./make_package.sh
Package Name:  voxl-tflite-server
version Number:  x.x.x
ar: creating voxl-tflite-server_x.x.x.ipk

DONE
```

This will make a new voxl-tflite-server_x.x.x.ipk file in your working directory. The name and version number came from the ipk/control/control file. If you are updating the package version, edit it there.


## Deploy to VOXL

You can now push the ipk package to the VOXL and install with opkg however you like. To do this over ADB, you may use the included helper script: install_on_voxl.sh.

Do this OUTSIDE of docker as your docker image probably doesn't have usb permissions for ADB.

```bash
~/git/voxl-tflite-server$ ./install_on_voxl.sh
pushing voxl-tflite-server_x.x.x.ipk to target
searching for ADB device
adb device found
voxl-tflite-server_x.x.x.ipk: 1 file pushed. 2.1 MB/s (51392 bytes in 0.023s)
Removing package voxl-tflite-server from root...
Installing voxl-tflite-server (x.x.x) on root.
Configuring voxl-tflite-server

Done installing voxl-tflite-server
```
