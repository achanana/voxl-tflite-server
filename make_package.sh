#!/bin/bash
################################################################################
# Copyright (c) 2019 ModalAI, Inc. All rights reserved.
#
# creates an ipk package from compiled ros nodes.
# be sure to build everything first with build.sh in docker
# run this on host pc
# UPDATE VERSION IN CONTROL FILE, NOT HERE!!!
#
# author: james@modalai.com
################################################################################

set -e # exit on error to prevent bad ipk from being generated

################################################################################
# variables
################################################################################
VERSION=$(cat ipk/control/control | grep "Version" | cut -d' ' -f 2)
PACKAGE=$(cat ipk/control/control | grep "Package" | cut -d' ' -f 2)
IPK_NAME=${PACKAGE}_${VERSION}.ipk

DATA_DIR=ipk/data
CONTROL_DIR=ipk/control

echo ""
echo "Package Name: " $PACKAGE
echo "version Number: " $VERSION

################################################################################
# start with a little cleanup to remove old files
################################################################################
sudo rm -rf $DATA_DIR
mkdir $DATA_DIR

rm -rf ipk/control.tar.gz
rm -rf ipk/data.tar.gz
rm -rf $IPK_NAME

################################################################################
## copy useful files into data directory
################################################################################

cd build && sudo make DESTDIR=../ipk/data PREFIX=/usr install && cd -

sudo mkdir -p $DATA_DIR/etc/systemd/system/ 2>/dev/null > /dev/null
sudo cp service/*.service $DATA_DIR/etc/systemd/system/ 2>/dev/null > /dev/null
sudo mkdir $DATA_DIR/usr/bin/dnn/
sudo cp -r ./dnn/*  $DATA_DIR/usr/bin/dnn/ 2>/dev/null

sudo mkdir -p $DATA_DIR/usr/bin/
sudo cp config/voxl-configure-tflite.sh $DATA_DIR/usr/bin/voxl-configure-tflite
sudo chmod +x $DATA_DIR/usr/bin/voxl-configure-tflite

sudo mkdir -p $DATA_DIR/usr/share/bash-completion/completions
sudo cp bash_completions/* $DATA_DIR/usr/share/bash-completion/completions


################################################################################
# pack the control, data, and final ipk archives
################################################################################

cd $CONTROL_DIR/
tar --create --gzip -f ../control.tar.gz *
cd ../../

cd $DATA_DIR/
tar --create --gzip -f ../data.tar.gz *
cd ../../

ar -r $IPK_NAME ipk/control.tar.gz ipk/data.tar.gz ipk/debian-binary

echo ""
echo DONE