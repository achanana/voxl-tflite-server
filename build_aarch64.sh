#!/bin/bash
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

set -x
set -e

mkdir -p build
cd build
cmake -DCMAKE_TOOLCHAIN_FILE=/opt/data/workspace/modalai/voxl-cross64/aarch64-gnu-4.9.toolchain.cmake -DCMAKE_BUILD_TYPE=RELEASE -DENABLE_NEON=ON -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -std=c++11 -march=armv8-a" ../source

#SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#TENSORFLOW_DIR="${SCRIPT_DIR}/../../../.."
#
## Toolchain compatible names
#if ! [ -L /usr/bin/aarch64-linux-gnu-g++ ]; then
#  ln -s /usr/bin/aarch64-linux-gnu-g++-4.9 /usr/bin/aarch64-linux-gnu-g++
#fi
#
#if ! [ -L /usr/bin/aarch64-linux-gnu-gcc ]; then
#  ln -s /usr/bin/aarch64-linux-gnu-gcc-4.9 /usr/bin/aarch64-linux-gnu-gcc
#fi

make -j 4 TARGET=aarch64 TARGET_TOOLCHAIN_PREFIX=aarch64-linux-gnu-
