#!/bin/bash

TOOLCHAIN64="/opt/cross_toolchain/aarch64-gnu-4.9.toolchain.cmake"


mkdir -p build64
cd build64
cmake -DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN64} -DCMAKE_BUILD_TYPE=RELEASE -DENABLE_NEON=ON ../
make -j4 TARGET=aarch64 TARGET_TOOLCHAIN_PREFIX=aarch64-linux-gnu-
cd ../






