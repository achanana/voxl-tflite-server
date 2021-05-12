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

# REPO="http://voxl-packages.modalai.com/stable"
# if [[ $# -eq 1 ]] ; then
#     echo "[INFO] updating repo to pull from"
#     REPO="http://voxl-packages.modalai.com/"$1
# fi
# echo "[INFO] Using repo: "$REPO

# echo ""
mkdir /usr/lib64/ 2>/dev/null

cd modalai
rm -rf temporary 2>/dev/null
mkdir temporary
chmod 777 temporary
cd temporary
mkdir temp
cd temp

# sudo opkg update


DEPS="libmodal_pipe libmodal_json opencv"

# # install/update each dependency
# for i in ${DEPS}; do
#     # this will also update if already installed!
#     sudo opkg install $i
# done

# variables
OPKG_CONF=/etc/opkg/opkg.conf
STABLE=http://voxl-packages.modalai.com/stable
DEV=http://voxl-packages.modalai.com/dev


# make sure opkg config file exists
if [ ! -f ${OPKG_CONF} ]; then
	echo "ERROR: missing ${OPKG_CONF}"
	echo "are you not running in voxl-emulator or voxl-cross?"
	exit 1
fi


# parse dev or stable option
if [ "$1" == "stable" ]; then
	echo "using stable repository"
	PKG_STRING="src/gz stable ${STABLE}"

elif [ "$1" == "dev" ]; then
	echo "using development repository"
	PKG_STRING="src/gz dev ${DEV}"

else
	echo ""
	echo "Please specify if the build dependencies should be pulled from"
	echo "the stable or development modalai opkg package repos."
	echo "If building the master branch you should specify stable."
	echo "For development branches please specify dev."
	echo ""
	echo "./install_build_deps.sh stable"
	echo "./install_build_deps.sh dev"
	echo ""
	exit 1
fi


# delete any existing repository entries
sudo sed -i '/voxl-packages.modalai.com/d' ${OPKG_CONF}

# write in the new entry
sudo echo ${PKG_STRING} >> ${OPKG_CONF}
sudo echo "" >> ${OPKG_CONF}

## make sure we have the latest package index
sudo opkg update


# install/update each dependency
for i in ${DEPS}; do
	# this will also update if already installed!
	sudo opkg install $i
done



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
