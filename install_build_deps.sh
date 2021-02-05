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
# * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OFe
# * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# * POSSIBILITY OF SUCH DAMAGE.
################################################################################

#!/bin/bash

REPO="http://voxl-packages.modalai.com/dev"
if [[ $# -eq 1 ]] ; then
    echo "[INFO] updating repo to pull from"
    REPO="http://voxl-packages.modalai.com/"$1
fi
echo "[INFO] Using repo: "$REPO

echo ""
mkdir /usr/lib64/ 2>/dev/null

cd modalai
rm -rf temporary 2>/dev/null
mkdir temporary
chmod 777 temporary
cd temporary
mkdir temp
cd temp
echo "Installing voxl-camera-server"
FILE=voxl-camera-server_0.2.7_202012150040.ipk
wget $REPO/$FILE 2>/dev/null
ar xvf $FILE > /dev/null
tar xvf data.tar.gz > /dev/null
cp -R usr/include/* /usr/include/ > /dev/null
rm -rf * > /dev/null

echo "Installing libmodal_pipe_1.6.2"
FILE=libmodal_pipe_1.6.2_202101301836.ipk   
wget $REPO/$FILE 2>/dev/null
ar xvf $FILE > /dev/null
tar xvf data.tar.gz > /dev/null
cp -R usr/include/* /usr/include/ > /dev/null
cp -R usr/lib64/* /usr/lib64/ > /dev/null
rm -rf * 2>/dev/null

echo "Installing opencv_4.3.0"
FILE=opencv_4.3.0.ipk
wget $REPO/$FILE 2>/dev/null
ar xvf $FILE > /dev/null
tar xvf data.tar.gz > /dev/null
cp -R usr/include/* /usr/include/ > /dev/null
cp -R usr/lib64/* /usr/lib64/ > /dev/null
rm -rf * > /dev/null

echo "Installing voxl_tflite_2_2_x"
FILE=voxl_tflite_2_2_x.ipk
cp ../../$FILE . > /dev/null
ar xvf $FILE > /dev/null
tar xvf data.tar.gz > /dev/null
cp -R usr/include/* /usr/include/ > /dev/null
cp -R usr/lib64/* /usr/lib64/ > /dev/null
rm -rf * > /dev/null

echo "Installing voxl-gpulibs"
FILE=voxl-gpulibs-64bit.ipk
cp ../../$FILE . > /dev/null
ar xvf $FILE > /dev/null
tar xvf data.tar.gz > /dev/null
mkdir -p /usr/include/CL/ > /dev/null
cp -R usr/CL/* /usr/include/CL/ > /dev/null
cp -R usr/lib64/* /usr/lib64/ > /dev/null
rm -rf * > /dev/null

cd .. 2>/dev/null
cd .. 2>/dev/null
rm -rf temporary 2>/dev/null

echo ""
echo "Done installing dependencies"
echo ""
exit 0
