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

cd build64 && sudo make DESTDIR=../ipk/data PREFIX=/usr install && cd -

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