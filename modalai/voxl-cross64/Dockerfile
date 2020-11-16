FROM ubuntu:16.04

RUN apt-get update

# Install required libraries
RUN apt-get -y install apt-utils wget gawk gperf help2man texinfo gcc gperf bison flex texinfo make libncurses5-dev python-dev unzip

#RUN apt-get -y install libavcodec-dev:arm64 libavformat-dev:arm64 libswscale-dev:arm64 libjpeg-dev:arm64 libpng-dev:arm64 libtiff-dev:arm64 libdc1394-22:arm64 libdc1394-22-dev:arm64 libv4l-dev:arm64 ffmpeg:arm64

RUN apt-get -y install --no-install-recommends libgtk2.0-dev
RUN apt-get clean --yes

RUN apt install -y g++-4.9-aarch64-linux-gnu 
RUN apt install -y git build-essential
RUN apt install -y cmake
RUN apt install -y vim
