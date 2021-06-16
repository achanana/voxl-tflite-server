################################################################################
# * Copyright 2021 ModalAI Inc.
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
cmake_minimum_required(VERSION 3.3)

# James' standard list of cmake flags
set(CMAKE_C_FLAGS "-std=gnu99 -Wall -Wextra -Wuninitialized \
	-Wunused-variable -Wdouble-promotion -Wmissing-prototypes \
	-Wmissing-declarations -Werror=undef -Wno-unused-function ${CMAKE_C_FLAGS}")

set(CMAKE_CXX_FLAGS "-std=gnu++11 -Wall -Wextra -Wuninitialized -Wno-attributes \
    -Wunused-variable -Wno-double-promotion -Wno-missing-field-initializers \
    -Werror=undef -Wno-unused-function ${CMAKE_CXX_FLAGS}")

set(PROJECT_NAME voxl-tflite-server)

file(GLOB_RECURSE all_src_files "*.c*")
add_executable(${PROJECT_NAME} ${all_src_files})

include_directories(
  /usr/include/opencv4/
  /usr/include/tensorflow-2-2-x/
  /usr/include/tensorflow-2-2-x/tensorflow
  /usr/include/tensorflow-2-2-x/tensorflow/lite/tools/make/downloads/absl/memory/
  /usr/include/tensorflow-2-2-x/tensorflow/lite/delegates/nnapi/
  /usr/include/tensorflow-2-2-x/tensorflow/lite/examples/label_image/
  /usr/include/tensorflow-2-2-x/tensorflow/lite/
  /usr/include/tensorflow-2-2-x/tensorflow/lite/tools/evaluation/
  /usr/include/tensorflow-2-2-x/tensorflow/lite/tools/make/downloads/flatbuffers/include/
  /usr/include/tensorflow-2-2-x/tensorflow/lite/tools/make/downloads/absl
  /usr/include/tensorflow-2-2-x/tensorflow/lite/tools/make/downloads/eigen/
  /usr/include/tensorflow-2-2-x/tensorflow/cc/framework/
)

link_directories(/usr/lib64/)

target_link_libraries(${PROJECT_NAME}
  -Wl,-rpath-link,/usr/lib64/
  -L/usr/lib64/
  tensorflow-lite
  benchmark-lib
  absl_stacktrace
  absl_synchronization
  absl_symbolize
  absl_debugging_internal
  absl_demangle_internal
  absl_dynamic_annotations
  glib-2.0
  dl
  stdc++
  pthread
  z
  cutils
  log
  sync
  gthread-2.0
  pcre
  modal_pipe
  modal_json
  opencv_highgui
  opencv_imgproc
  opencv_core
  opencv_imgcodecs
  gsl
  llvm-qcom
  adreno_utils
  OpenCL
  CB
  EGL_adreno
  GLESv2_adreno
)

# Mark executables and/or libraries for installation
install(
    TARGETS ${PROJECT_NAME}
    LIBRARY			  DESTINATION /usr/lib
    RUNTIME			  DESTINATION /usr/bin
    PUBLIC_HEADER	  DESTINATION /usr/include
 )
