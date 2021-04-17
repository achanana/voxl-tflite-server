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
set -e

PACKAGE=$(cat ipk/control/control | grep "Package" | cut -d' ' -f 2)

# count ipk files in current directory
NUM_FILES=$(ls -1q $PACKAGE*.ipk | wc -l)

if [ $NUM_FILES -eq "0" ]; then
	echo "ERROR: no ipk file found"
	echo "run build.sh and make_package.sh first"
	exit 1
elif [ $NUM_FILES -gt "1" ]; then
	echo "ERROR: more than 1 ipk file found"
	echo "make sure there is only one ipk file in the current directory"
	exit 1
fi

echo "searching for ADB device"
adb wait-for-device
echo "adb device found"

# now we know only one ipk file exists
FILE=$(ls -1q $PACKAGE*.ipk)
echo "pushing $FILE to target"

adb push $FILE /home/root/ipk/$FILE
adb shell "opkg remove $PACKAGE"

adb shell "opkg install --force-reinstall --force-downgrade --force-depends /home/root/ipk/$FILE"

# adb shell "mv /usr/lib64/libstdc++.so.6.0.20 /usr/lib64/libstdc++.so.6.0.20.ORIGINAL 2>/dev/null"
# cd modalai
# rm -rf temporary 2>/dev/null
# mkdir temporary
# chmod 777 temporary
# cd temporary
# mkdir temp
# cd temp

# adb shell "sudo opkg update"

# DEPS="libmodal_pipe libmodal_json opencv"

# # install/update each dependency
# for i in ${DEPS}; do
#     # this will also update if already installed!
#     adb shell "sudo opkg install $i"
# done

# echo "Installing voxl-gpulibs"
# FILE=voxl-gpulibs-64bit.ipk
# cp ../../$FILE .
# adb push $FILE /home/root/ipk/$FILE
# adb shell "opkg install --force-reinstall --force-downgrade --force-depends /home/root/ipk/$FILE"

# adb shell "mv /usr/lib64/libstdc++.so.6.0.22 /usr/lib64/libstdc++.so.6.0.20 2>/dev/null"
