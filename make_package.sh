#!/bin/bash
################################################################################
# * Copyright 2020 ModalAI Inc.
# *
# * Redistribution and use in source and binary forms, with or without
# * modification, are permitted provided that the following conditions are met:
# *
# * 1. Redistributions of source code must retain the above copyright notice,
# *    this list of conditions and the following disclaimer.
# *
# * 2. Redistributions in binary form must reproduce the above copyright notice,
# *    this list of conditions and the following disclaimer in the documentation
# *    and/or other materials provided with the distribution.
# *
# * 3. Neither the name of the copyright holder nor the names of its contributors
# *    may be used to endorse or promote products derived from this software
# *    without specific prior written permission.
# *
# * 4. The Software is used solely in conjunction with devices provided by
# *    ModalAI Inc.
# *
# * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# * POSSIBILITY OF SUCH DAMAGE.
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
echo "Package Name   : " $PACKAGE
echo "version Number : " $VERSION
echo "IPK Name       : " $IPK_NAME

################################################################################
# start with a little cleanup to remove old files
################################################################################
sudo rm -rf ipk/control.tar.gz 2>/dev/null
sudo rm -rf $DATA_DIR/ 2>/dev/null
sudo rm -rf ipk/data.tar.gz 2>/dev/null
sudo rm -rf $IPK_NAME 2>/dev/null

################################################################################
## copy useful files into data directory
################################################################################
#./build_aarch64.sh
sudo mkdir -p $DATA_DIR/usr/bin/dnn/data 2>/dev/null
sudo cp -r ./dnn/*  $DATA_DIR/usr/bin/dnn/ 2>/dev/null
sudo cp ./build/voxl-tflite-server $DATA_DIR/usr/bin/ 2>/dev/null

################################################################################
# pack the control, data, and final ipk archives
################################################################################

cd $CONTROL_DIR/
tar --create --gzip -f ../control.tar.gz * 2>/dev/null
cd ../../ 2>/dev/null

cd $DATA_DIR/
tar --create --gzip -f ../data.tar.gz * 2>/dev/null
cd ../../ 2>/dev/null

ar -r $IPK_NAME ipk/control.tar.gz ipk/data.tar.gz ipk/debian-binary 2>/dev/null
echo ""
echo DONE
echo ""
