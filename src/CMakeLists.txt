cmake_minimum_required(VERSION 3.3)

SET(TARGET voxl-tflite-server)

option(BUILD_820 "Build the 820 binary" ON)
option(BUILD_865 "Build the 865 binary" OFF)

if(BUILD_865)
    add_definitions(-DBUILD_865)
    set(BUILD_820 OFF)
endif()

# Build from all source files
file(GLOB all_src_files *.c*)

add_executable(${TARGET}
	${all_src_files}
)

include_directories(
    ../include
    /usr/include/opencv4/

    # 820 SPECIFIC
    /usr/include/tensorflow/lite/tools/make/downloads/absl/
    /usr/include/tensorflow/lite/tools/make/downloads/flatbuffers/include/

    # 865 SPECIFIC
    /usr/include/flatbuffers/include/
    /usr/include/abseil-cpp/
    /usr/include/ruy/
)

set(ABSL_LIBS
    "absl_bad_any_cast_impl"
    "absl_bad_optional_access"
    "absl_bad_variant_access"
    "absl_base"
    "absl_city"
    "absl_civil_time"
    "absl_cord"
    "absl_debugging_internal"
    "absl_demangle_internal"
    "absl_exponential_biased"
    "absl_flags"
    "absl_flags_commandlineflag"
    "absl_flags_commandlineflag_internal"
    "absl_flags_config"
    "absl_flags_internal"
    "absl_flags_marshalling"
    "absl_flags_private_handle_accessor"
    "absl_flags_program_name"
    "absl_flags_reflection"
    "absl_graphcycles_internal"
    "absl_hash"
    "absl_hashtablez_sampler"
    "absl_int128"
    "absl_log_severity"
    "absl_malloc_internal"
    "absl_raw_hash_set"
    "absl_raw_logging_internal"
    "absl_spinlock_wait"
    "absl_stacktrace"
    "absl_status"
    "absl_str_format_internal"
    "absl_strings"
    "absl_strings_internal"
    "absl_symbolize"
    "absl_synchronization"
    "absl_throw_delegate"
    "absl_time"
    "absl_time_zone"
    "absl_wyhash"
)

if (BUILD_865)
set(LINK_LIBS
    "tensorflow-lite"
    ${ABSL_LIBS}
    ${ABSL_LIBS}
    ${ABSL_LIBS}
    "flatbuffers"
    "XNNPACK"
    "pthreadpool"
    "cpuinfo"
    "farmhash"
    "clog"
    "fft2d_fftsg2d"
    "fft2d_fftsg"
    "ruy"
    "glib-2.0"
    "dl"
    "stdc++"
    "pthread"
    "z"
    "cutils"
    "log"
    "sync"
    "gthread-2.0"
    "pcre"
    "modal_pipe"
    "modal_json"
    "opencv_core"
    "opencv_highgui"
    "opencv_imgproc"
    "opencv_imgcodecs"
    "gsl"
    "llvm-qcom"
    "adreno_utils"
    "OpenCL"
    "CB"
    "EGL_adreno"
    "GLESv2_adreno"
    "rt"
)
else()
set(LINK_LIBS
    "tensorflow-lite"
    "benchmark-lib"
    "absl_stacktrace"
    "absl_synchronization"
    "absl_symbolize"
    "absl_debugging_internal"
    "absl_demangle_internal"
    "absl_dynamic_annotations"
    "glib-2.0"
    "dl"
    "stdc++"
    "pthread"
    "z"
    "cutils"
    "log"
    "sync"
    "gthread-2.0"
    "pcre"
    "modal_pipe"
    "modal_json"
    "opencv_highgui"
    "opencv_imgproc"
    "opencv_core"
    "opencv_imgcodecs"
    "gsl"
    "llvm-qcom"
    "adreno_utils"
    "OpenCL"
    "CB"
    "EGL_adreno"
    "GLESv2_adreno"
    "rt"
)
endif()

target_link_libraries(${TARGET}
    -Wl,-rpath-link,/usr/lib64/
    -L/usr/lib64/
    ${LINK_LIBS}
)
# make sure everything is installed where we want
# LIB_INSTALL_DIR comes from the parent cmake file
install(
	TARGETS			${TARGET}
	LIBRARY			DESTINATION ${LIB_INSTALL_DIR}
	RUNTIME			DESTINATION /usr/bin
	PUBLIC_HEADER	DESTINATION /usr/include
)
