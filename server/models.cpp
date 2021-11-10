/*******************************************************************************
 * Copyright 2021 ModalAI Inc.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * 4. The Software is used solely in conjunction with devices provided by
 *    ModalAI Inc.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 ******************************************************************************/


#include <algorithm>
#include <fstream>
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/resource.h>
#include <unistd.h>
#include "threads.h"
#include "undistort.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
////////////////////////////////////////////////////////////////////////////////
//      ***********************WARNING************************
//      INCLUDE TENSORFLOW HEADERS *AFTER* LIBMODAL PIPE HEADERS
//      OTHERWISE WILL SET __ANDROID__ FLAG
//      *********************END WARNING**********************
////////////////////////////////////////////////////////////////////////////////
#include "memory.h"
#include "bitmap_helpers.h"
#include "optional_debug_tools.h"
#include "utils.h"

#define MPA_TFLITE_PATH (MODAL_PIPE_DEFAULT_BASE_DIR "tflite/")
#define MPA_TFLITE_DATA_PATH (MODAL_PIPE_DEFAULT_BASE_DIR "tflite_data/")

////////////////////////////////////////////////////////////////////////////////
// TFLITE DETECTION DATA FORMATS
// per frame, will send out one detections_array packet to /run/mpa/tflite_data
// can be filled with up to 64 detections per
////////////////////////////////////////////////////////////////////////////////
typedef struct object_detection_msg {
    int64_t timestamp_ns;
    uint32_t class_id;
    char class_name[64];
    float class_confidence;
    float detection_confidence;
    float x_min;
    float y_min;
    float x_max;
    float y_max;
} __attribute__((packed)) object_detection_msg;

typedef struct detections_array {
    int32_t num_detections;
    object_detection_msg detections[64];
} __attribute__((packed)) detections_array;

namespace tflite
{
namespace label_image
{

// generic template for TensorData*
template<typename T>
T* TensorData(TfLiteTensor* tensor, int batch_index);

////////////////////////////////////////////////////////////////////////////////
// Gets the float tensor data pointer
////////////////////////////////////////////////////////////////////////////////
template<>
float* TensorData(TfLiteTensor* tensor, int batch_index)
{
    int nelems = 1;

    for (int i = 1; i < tensor->dims->size; i++){
        nelems *= tensor->dims->data[i];
    }

    switch (tensor->type){
        case kTfLiteFloat32:
            return tensor->data.f + nelems * batch_index;
        default:
            fprintf(stderr, "Error in %s: should not reach here\n", __FUNCTION__);
    }

    return nullptr;
}

////////////////////////////////////////////////////////////////////////////////
// Gets the uint8_t tensor data pointer
////////////////////////////////////////////////////////////////////////////////
template<>
uint8_t* TensorData(TfLiteTensor* tensor, int batch_index)
{
    int nelems = 1;

    for (int i = 1; i < tensor->dims->size; i++){
        nelems *= tensor->dims->data[i];
    }

    switch (tensor->type){
        case kTfLiteUInt8:
            return tensor->data.uint8 + nelems * batch_index;
        default:
            fprintf(stderr, "Error in %s: should not reach here\n", __FUNCTION__);
    }

    return nullptr;
}

////////////////////////////////////////////////////////////////////////////////
// Timing helper functions
////////////////////////////////////////////////////////////////////////////////
uint64_t rc_nanos_thread_time()
{
	struct timespec ts;
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
	return ((uint64_t)ts.tv_sec*1000000000)+ts.tv_nsec;
}

uint64_t rc_nanos_monotonic_time()
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ((uint64_t)ts.tv_sec*1000000000)+ts.tv_nsec;
}


using TfLiteDelegatePtr    = tflite::Interpreter::TfLiteDelegatePtr;
using TfLiteDelegatePtrMap = std::map<std::string, TfLiteDelegatePtr>;

////////////////////////////////////////////////////////////////////////////////
// Creates a GPU delegate
////////////////////////////////////////////////////////////////////////////////
TfLiteDelegatePtr CreateGPUDelegate(Settings* s)
{
#if defined(__ANDROID__)
    TfLiteGpuDelegateOptionsV2 gpu_opts = TfLiteGpuDelegateOptionsV2Default();

    gpu_opts.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
    gpu_opts.inference_priority1  = s->allow_fp16 ? TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY
                                    : TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION;

    return evaluation::CreateGPUDelegate(&gpu_opts);
#else
    return evaluation::CreateGPUDelegate(s->model);
#endif
}

////////////////////////////////////////////////////////////////////////////////
// Gets all the available delegates, returns the delegate map*
////////////////////////////////////////////////////////////////////////////////
TfLiteDelegatePtrMap GetDelegates(Settings* s)
{
    TfLiteDelegatePtrMap delegates;

    if (s->gl_backend){
        auto delegate = CreateGPUDelegate(s);

        if (!delegate){
            fprintf(stderr, "GPU acceleration is unsupported on this platform.\n");
        }
        else{
            fprintf(stderr, "GPU acceleration is SUPPORTED on this platform\n");
            delegates.emplace("GPU", std::move(delegate));
        }
    }
    return delegates;
}

////////////////////////////////////////////////////////////////////////////////
// Takes a file name, and loads a list of labels from it, one per line, and
// returns a vector of the strings. It pads with empty strings so the length of
// the result is a multiple of 16, because our model expects that.
////////////////////////////////////////////////////////////////////////////////
TfLiteStatus ReadLabelsFile(const string& file_name, std::vector<string>* result, size_t* found_label_count)
{
    std::ifstream file(file_name);

    if (!file){
        fprintf(stderr, "Labels file %s not found\n", file_name.c_str());
        return kTfLiteError;
    }

    result->clear();
    string line;
    while (std::getline(file, line)){
        result->push_back(line);
    }

    *found_label_count = result->size();
    const int padding = 16;

    while (result->size() % padding){
        result->emplace_back();
    }
    return kTfLiteOk;
}

////////////////////////////////////////////////////////////////////////////////
// MobileNet object detection thread
////////////////////////////////////////////////////////////////////////////////
void TFliteMobileNet(void* data)
{
    int model_img_height;
    int model_img_width;
    int model_img_channels;
    std::unique_ptr<tflite::FlatBufferModel>            model;
    std::unique_ptr<tflite::Interpreter>                interpreter;
    tflite::ops::builtin::BuiltinOpResolver             resolver;
    tflite::label_image::Settings* tflite_settings      = NULL;
    TFliteThreadData* mobilenet_data                    = (TFliteThreadData*)data;
    pipe_info_t tflite_pipe                             = {"tflite", MPA_TFLITE_PATH, "camera_image_metadata_t", PROCESS_NAME, 16*1024*1024, 0};
    pipe_server_create(TFLITE_CH, tflite_pipe, 0);
    pipe_info_t data_pipe                               = {"tflite_data", MPA_TFLITE_DATA_PATH, "detections", PROCESS_NAME, 16*1024, 0};
    pipe_server_create(TFLITE_DATA_CH, data_pipe, 0);
    cv::Mat input_img, resized_img, output_img;
    static bool color = false;
    camera_image_metadata_t meta;
    uint64_t start_time, end_time;
    float total_resize_time, total_tensor_time, total_model_time;
    tflite_settings = new tflite::label_image::Settings;

    tflite_settings->model_name                   = mobilenet_data->model_file;
    tflite_settings->labels_file_name             = mobilenet_data->labels_file;
    tflite_settings->gl_backend                   = 1;
    tflite_settings->number_of_threads            = 4;
    tflite_settings->allow_fp16                   = 1;
    tflite_settings->input_mean                   = 127;
    tflite_settings->loop_count                   = 1;

    if (!tflite_settings->model_name.c_str()){
        fprintf(stderr, "FATAL: no model file name\n");
        exit(-1);
    }

    model = tflite::FlatBufferModel::BuildFromFile(tflite_settings->model_name.c_str());

    if (!model){
        fprintf(stderr, "FATAL: Failed to mmap model %s\n", tflite_settings->model_name.c_str());
        exit(-1);
    }

    tflite_settings->model = model.get();
    if (mobilenet_data->en_debug) printf("Loaded model %s\n", tflite_settings->model_name.c_str());
    model->error_reporter();
    if (mobilenet_data->en_debug) printf("Resolved reporter\n");

    tflite::InterpreterBuilder(*model, resolver)(&interpreter);

    if (!interpreter){
        fprintf(stderr, "Failed to construct interpreter\n");
        exit(-1);
    }

    interpreter->UseNNAPI(tflite_settings->old_accel);
    interpreter->SetAllowFp16PrecisionForFp32(tflite_settings->allow_fp16);

    if (tflite_settings->number_of_threads != -1){
        interpreter->SetNumThreads(tflite_settings->number_of_threads);
    }

    TfLiteDelegatePtrMap delegates_ = GetDelegates(tflite_settings);

    for (const auto& delegate : delegates_){
        if (interpreter->ModifyGraphWithDelegate(delegate.second.get()) != kTfLiteOk){
            printf("Failed to apply delegate\n");
            break;
        }
        else{
             if (mobilenet_data->en_debug) printf("Applied delegate \n");
            break;
        }
    }

    if (interpreter->AllocateTensors() != kTfLiteOk){
        fprintf(stderr, "Failed to allocate tensors!\n");
        exit(-1);
    }

    TfLiteIntArray* dims = interpreter->tensor(interpreter->inputs()[0])->dims;

    model_img_height   = dims->data[1];
    model_img_width    = dims->data[2];
    model_img_channels = dims->data[3];

    // Set thread priority
    pid_t tid = syscall(SYS_gettid);
    int which = PRIO_PROCESS;
    int nice  = -15;
    setpriority(which, tid, nice);

    // Inform the camera frames receiver that tflite processing is ready to receive frames and start processing
    mobilenet_data->thread_ready = true;
    fprintf(stderr, "\n------Setting TFLiteThread to ready!! W: %d H: %d C:%d",
            model_img_width, model_img_height, model_img_channels);

    int queue_process_idx          = 0;
    total_tensor_time              = 0;
    total_resize_time              = 0;
    total_model_time               = 0;
    int num_frames                 = 0;
    uint8_t resize_output[300*300] = {0};

    while (mobilenet_data->stop == false){
        if (queue_process_idx == mobilenet_data->camera_queue->insert_idx){
            std::unique_lock<std::mutex> lock(mobilenet_data->cond_mutex);
            mobilenet_data->cond_var.wait(lock);
            continue;
        }
        if (((!mobilenet_data->en_debug) && (!mobilenet_data->en_timing))){
            if (pipe_server_get_num_clients(TFLITE_CH) == 0 ){
                continue;
            }
        }
        // Coming here means we have a frame to run through the DNN model
        num_frames++;
        TFLiteMessage* new_frame = &mobilenet_data->camera_queue->queue[queue_process_idx];
        detections_array detection_output;
        detection_output.num_detections = 0;

        if (new_frame->metadata.format == IMAGE_FORMAT_NV12 || new_frame->metadata.format == IMAGE_FORMAT_NV21){
            color = true;
        }
        else if (new_frame->metadata.format != IMAGE_FORMAT_RAW8){
            fprintf(stderr, "Unexpected image format %d received! Exiting now.\n", new_frame->metadata.format);
            main_running = 0;
        }
        if (mobilenet_data->en_debug){
            fprintf(stderr, "\n------Popping index %d frame %d ...... Queue size: %d\n",
                queue_process_idx, new_frame->metadata.frame_id,
                abs(mobilenet_data->camera_queue->insert_idx - queue_process_idx));
        }

        meta = new_frame->metadata;
        int img_width    = meta.width;
        int img_height   = meta.height;
        int img_channels = 3;

        // bilinear resize struct + setup
        static undistort_map_t map;
        if (num_frames == 1){
            mcv_init_resize_map(img_width, img_height, model_img_width, model_img_height, &map);
        }

        input_img = cv::Mat(img_height, img_width, CV_8UC1, (uchar*)new_frame->image_pixels);

        if (color){
            cv::Mat yuv(img_height + img_height/2, img_width, CV_8UC1, (uchar*)new_frame->image_pixels);
            cv::cvtColor(yuv, output_img, CV_YUV2RGB_NV21);
        }
        else output_img = input_img.clone();

        start_time = rc_nanos_monotonic_time();
        mcv_resize_image(input_img.data, resize_output, &map);
        end_time = rc_nanos_monotonic_time();
        if (mobilenet_data->en_timing){
            printf("\nImage resize time:          %6.2fms\n", ((double)(end_time-start_time))/1000000.0);
            total_resize_time += (end_time-start_time);
        }
        cv::Mat resized(300, 300, CV_8UC1, (uchar*)resize_output);
        cv::Mat in[] = {resized, resized, resized};
        cv::merge(in, 3, resized_img);

        uint8_t*               pImageData = (uint8_t*)resized_img.data;

        const std::vector<int> inputs     = interpreter->inputs();
        const std::vector<int> outputs    = interpreter->outputs();

        // Get input dimension from the input tensor metadata assuming one input only
        int input = interpreter->inputs()[0];

        switch (interpreter->tensor(input)->type){
            case kTfLiteFloat32: {
                tflite_settings->input_floating = true;
                start_time = rc_nanos_monotonic_time();
                float* dst = TensorData<float>(interpreter->tensor(input), 0);
                const int row_elems = model_img_width * model_img_channels;
                for (int row = 0; row < model_img_height; row++) {
                    const uchar* row_ptr = resized_img.ptr(row);
                    for (int i = 0; i < row_elems; i++) {
                        dst[i] = (row_ptr[i] - 127.0f) / 127.0f;
                    }
                    dst += row_elems;
                }
                end_time = rc_nanos_monotonic_time();
                if (mobilenet_data->en_timing){
                    printf("Tflite tensor resize time:  %6.2fms\n", ((double)(end_time-start_time))/1000000.0);
                    total_tensor_time += (end_time-start_time);
                }
            }
                break;

            case kTfLiteUInt8:
                resize<uint8_t>(interpreter->typed_tensor<uint8_t>(input), pImageData,
                            model_img_height, model_img_width, img_channels, model_img_height,
                            model_img_width, model_img_channels, tflite_settings);
                break;

            default:
                exit(-1);
        }

        start_time = rc_nanos_monotonic_time();
        for (int i = 0; i < tflite_settings->loop_count; i++){
            if (interpreter->Invoke() != kTfLiteOk){
                fprintf(stderr, "Failed to invoke tflite!\n");
            }
        }
        end_time = rc_nanos_monotonic_time();
        if (mobilenet_data->en_timing){
            printf("Model execution time:       %6.2fms\n", ((double)(end_time-start_time))/1000000.0);
            total_model_time += (end_time-start_time);
        }

        // https://www.tensorflow.org/lite/models/object_detection/overview#starter_model
        TfLiteTensor* output_locations    = interpreter->tensor(interpreter->outputs()[0]);
        TfLiteTensor* output_classes      = interpreter->tensor(interpreter->outputs()[1]);
        TfLiteTensor* output_scores       = interpreter->tensor(interpreter->outputs()[2]);
        TfLiteTensor* output_detections   = interpreter->tensor(interpreter->outputs()[3]);
        const float*  detected_locations  = TensorData<float>(output_locations, 0);
        const float*  detected_classes    = TensorData<float>(output_classes, 0);
        const float*  detected_scores     = TensorData<float>(output_scores, 0);
        const int     detected_numclasses = (int)(*TensorData<float>(output_detections, 0));

        std::vector<string> labels;
        size_t label_count;

        if (ReadLabelsFile(tflite_settings->labels_file_name, &labels, &label_count) != kTfLiteOk){
            fprintf(stderr, "Unable to read labels file\n");
            exit(-1);
        }

        for (int i = 0; i < detected_numclasses; i++){
            const float score  = detected_scores[i];
            const int   top    = detected_locations[4 * i + 0] * img_height;
            const int   left   = detected_locations[4 * i + 1] * img_width;
            const int   bottom = detected_locations[4 * i + 2] * img_height;
            const int   right  = detected_locations[4 * i + 3] * img_width;

            // Check for object detection confidence of 60% or more
            if (score > 0.6f){
                if (mobilenet_data->en_debug){
                    printf("Detected: %s, Confidence: %6.2f\n", labels[detected_classes[i]].c_str(), (double)score);
                }
                int height = bottom - top;
                int width  = right - left;

                cv::Rect rect(left, top, width, height);
                cv::Point pt(left, top-10);

                cv::rectangle(output_img, rect, cv::Scalar(0), 7);
                cv::putText(output_img, labels[detected_classes[i]], pt, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0), 2);
                object_detection_msg curr_detection;
                curr_detection.timestamp_ns = rc_nanos_monotonic_time();
                curr_detection.class_id = detected_classes[i];
                strcpy(curr_detection.class_name, labels[detected_classes[i]].c_str());
                curr_detection.class_confidence = score;
                curr_detection.detection_confidence = -1; // UNKNOWN for ssd model architecture
                curr_detection.x_min = left;
                curr_detection.y_min = top;
                curr_detection.x_max = right;
                curr_detection.y_max = bottom;

                detection_output.detections[detection_output.num_detections++] = curr_detection;
            }
        }
        if (color){
            meta.format         = IMAGE_FORMAT_RGB;
            meta.size_bytes     = (img_height * img_width * 3);
            meta.stride         = (img_width * 3);
        }
        else {
            meta.format         = IMAGE_FORMAT_RAW8;
            meta.size_bytes     = (img_height * img_width);
            meta.stride         = (img_width);
        }

        if (output_img.data != NULL){
            pipe_server_write_camera_frame(TFLITE_CH, meta, (char*)output_img.data);
        }
        if (detection_output.num_detections > 0){
            int sz_bytes = (detection_output.num_detections * sizeof(object_detection_msg)) + sizeof(int32_t);
            pipe_server_write(TFLITE_DATA_CH, (char*)&detection_output, sz_bytes);
        }
        queue_process_idx = ((queue_process_idx + 1) % QUEUE_SIZE);
    }

    if (mobilenet_data->en_timing){
        printf("\n------------------------------------------\n");
        printf("TIMING STATS (on %d processed frames)\n", num_frames);
        printf("------------------------------------------\n");
        printf("Image resize time   -> Total: %6.2fms, Average: %6.2fms\n", (double)(total_resize_time/1000000), (double)((total_resize_time/(1000000* num_frames))));
        printf("Tensor resize time  -> Total: %6.2fms, Average: %6.2fms\n", (double)(total_tensor_time/1000000), (double)((total_tensor_time/(1000000* num_frames))));
        printf("Model GPU execution -> Total: %6.2fms, Average: %6.2fms\n", (double)(total_model_time/1000000), (double)((total_model_time/(1000000* num_frames))));
    }

    if (tflite_settings != NULL){
        delete tflite_settings;
    }
}

void TFliteMidas(void* data)
{
    int model_img_height;
    int model_img_width;
    int model_img_channels;
    std::unique_ptr<tflite::FlatBufferModel>            model;
    std::unique_ptr<tflite::Interpreter>                interpreter;
    tflite::ops::builtin::BuiltinOpResolver             resolver;
    tflite::label_image::Settings* tflite_settings      = NULL;
    TFliteThreadData* midas_data                        = (TFliteThreadData*)data;
    pipe_info_t tflite_pipe                             = {"tflite", MPA_TFLITE_PATH, "camera_image_metadata_t", PROCESS_NAME, 16*1024*1024, 0};
    pipe_server_create(TFLITE_CH, tflite_pipe, 0);
    cv::Mat input_img, resized_img, output_img;
    camera_image_metadata_t meta;
    uint64_t start_time, end_time;
    float total_resize_time, total_tensor_time, total_model_time;
    tflite_settings = new tflite::label_image::Settings;

    tflite_settings->model_name                   = midas_data->model_file;
    tflite_settings->labels_file_name             = midas_data->labels_file;
    tflite_settings->gl_backend                   = 1;
    tflite_settings->number_of_threads            = 4;
    tflite_settings->allow_fp16                   = 1;
    tflite_settings->input_mean                   = 127;
    tflite_settings->loop_count                   = 1;

    if (!tflite_settings->model_name.c_str()){
        fprintf(stderr, "FATAL: no model file name\n");
        exit(-1);
    }

    model = tflite::FlatBufferModel::BuildFromFile(tflite_settings->model_name.c_str());

    if (!model){
        fprintf(stderr, "FATAL: Failed to mmap model %s\n", tflite_settings->model_name.c_str());
        exit(-1);
    }

    tflite_settings->model = model.get();
    if (midas_data->en_debug) printf("Loaded model %s\n", tflite_settings->model_name.c_str());
    model->error_reporter();
    if (midas_data->en_debug) printf("Resolved reporter\n");

    tflite::InterpreterBuilder(*model, resolver)(&interpreter);

    if (!interpreter){
        fprintf(stderr, "Failed to construct interpreter\n");
        exit(-1);
    }

    interpreter->UseNNAPI(tflite_settings->old_accel);
    interpreter->SetAllowFp16PrecisionForFp32(tflite_settings->allow_fp16);

    if (tflite_settings->number_of_threads != -1){
        interpreter->SetNumThreads(tflite_settings->number_of_threads);
    }

    TfLiteDelegatePtrMap delegates_ = GetDelegates(tflite_settings);

    for (const auto& delegate : delegates_){
        if (interpreter->ModifyGraphWithDelegate(delegate.second.get()) != kTfLiteOk){
            printf("Failed to apply delegate\n");
            break;
        }
        else{
             if (midas_data->en_debug) printf("Applied delegate \n");
            break;
        }
    }

    if (interpreter->AllocateTensors() != kTfLiteOk){
        fprintf(stderr, "Failed to allocate tensors!\n");
        exit(-1);
    }

    TfLiteIntArray* dims = interpreter->tensor(interpreter->inputs()[0])->dims;

    model_img_height   = dims->data[1];
    model_img_width    = dims->data[2];
    model_img_channels = dims->data[3];

    // Set thread priority
    pid_t tid = syscall(SYS_gettid);
    int which = PRIO_PROCESS;
    int nice  = -15;
    setpriority(which, tid, nice);

    // Inform the camera frames receiver that tflite processing is ready to receive frames and start processing
    midas_data->thread_ready = true;
    fprintf(stderr, "\n------Setting TFLiteThread to ready!! W: %d H: %d C:%d",
            model_img_width, model_img_height, model_img_channels);

    int queue_process_idx          = 0;
    total_tensor_time              = 0;
    total_resize_time              = 0;
    total_model_time               = 0;
    int num_frames                 = 0;
    // uint8_t resize_output[model_img_width*model_img_height*3] = {0};

    while (midas_data->stop == false){
        if (queue_process_idx == midas_data->camera_queue->insert_idx){
            std::unique_lock<std::mutex> lock(midas_data->cond_mutex);
            midas_data->cond_var.wait(lock);
            continue;
        }
        if (((!midas_data->en_debug) && (!midas_data->en_timing))){
            if (pipe_server_get_num_clients(TFLITE_CH) == 0 ){
                continue;
            }
        }
        // Coming here means we have a frame to run through the DNN model
        num_frames++;
        TFLiteMessage* new_frame = &midas_data->camera_queue->queue[queue_process_idx];

        if (midas_data->en_debug){
            fprintf(stderr, "\n------Popping index %d frame %d ...... Queue size: %d\n",
                queue_process_idx, new_frame->metadata.frame_id,
                abs(midas_data->camera_queue->insert_idx - queue_process_idx));
        }

        meta = new_frame->metadata;
        int img_width    = meta.width;
        int img_height   = meta.height;
        int img_channels = 3;

        // bilinear resize struct + setup
        static undistort_map_t map;
        if (num_frames == 1){
            mcv_init_resize_map(img_width, img_height, model_img_width, model_img_height, &map);
        }

        cv::Mat yuv(img_height + img_height/2, img_width, CV_8UC1, (uchar*)new_frame->image_pixels);
        cv::cvtColor(yuv, output_img, CV_YUV2RGB_NV21);

        start_time = rc_nanos_monotonic_time();
        fprintf(stderr, "about to resize\n");
        uint8_t resize_output[model_img_width*model_img_height*3] = {0};

        mcv_resize_8uc3_image((uint8_t*)output_img.data, resize_output, &map);
        end_time = rc_nanos_monotonic_time();
        if (midas_data->en_timing){
            printf("\nImage resize time:          %6.2fms\n", ((double)(end_time-start_time))/1000000.0);
            total_resize_time += (end_time-start_time);
        }

        cv::Mat resized(model_img_height, model_img_width, CV_8UC3, (uchar*)resize_output);

        uint8_t* pImageData = (uint8_t*)resized.data;

        const std::vector<int> inputs     = interpreter->inputs();
        const std::vector<int> outputs    = interpreter->outputs();

        // Get input dimension from the input tensor metadata assuming one input only
        int input = interpreter->inputs()[0];

        switch (interpreter->tensor(input)->type){
            case kTfLiteFloat32: {
                tflite_settings->input_floating = true;
                start_time = rc_nanos_monotonic_time();
                float* dst = TensorData<float>(interpreter->tensor(input), 0);
                const int row_elems = model_img_width * model_img_channels;
                for (int row = 0; row < model_img_height; row++) {
                    const uchar* row_ptr = resized.ptr(row);
                    for (int i = 0; i < row_elems; i++) {
                        dst[i] = (row_ptr[i] - 127.0f) / 127.0f;
                    }
                    dst += row_elems;
                }
                end_time = rc_nanos_monotonic_time();
                if (midas_data->en_timing){
                    printf("Tflite tensor resize time:  %6.2fms\n", ((double)(end_time-start_time))/1000000.0);
                    total_tensor_time += (end_time-start_time);
                }
            }
                break;

            case kTfLiteUInt8:
                resize<uint8_t>(interpreter->typed_tensor<uint8_t>(input), pImageData,
                            model_img_height, model_img_width, img_channels, model_img_height,
                            model_img_width, model_img_channels, tflite_settings);
                break;

            default:
                exit(-1);
        }

        start_time = rc_nanos_monotonic_time();
        for (int i = 0; i < tflite_settings->loop_count; i++){
            if (interpreter->Invoke() != kTfLiteOk){
                fprintf(stderr, "Failed to invoke tflite!\n");
            }
        }
        end_time = rc_nanos_monotonic_time();
        if (midas_data->en_timing){
            printf("Model execution time:       %6.2fms\n", ((double)(end_time-start_time))/1000000.0);
            total_model_time += (end_time-start_time);
        }

        TfLiteTensor* output_locations    = interpreter->tensor(interpreter->outputs()[0]);
        float* depth  = TensorData<float>(output_locations, 0);
        cv::Mat depthImage(model_img_height, model_img_width, CV_32FC1, depth);
        double min_val, max_val;
        cv::Mat depthmap_visual;
        cv::minMaxLoc(depthImage, &min_val, &max_val);
        depthmap_visual = 255 * (depthImage - min_val) / (max_val - min_val); // * 255 for "scaled" disparity, 15 for midas default
        depthmap_visual.convertTo(depthmap_visual, CV_8U);
        cv::applyColorMap(depthmap_visual, output_img, 4); //COLORMAP_JET

        meta.format         = IMAGE_FORMAT_RGB;
        meta.size_bytes     = (output_img.rows * output_img.cols * 3);
        meta.stride         = (output_img.cols * 3);
        meta.width          = output_img.cols;
        meta.height         = output_img.rows;

        if (output_img.data != NULL){
                pipe_server_write_camera_frame(TFLITE_CH, meta, (char*)output_img.data);
        }
        queue_process_idx = ((queue_process_idx + 1) % QUEUE_SIZE);
    }

    if (midas_data->en_timing){
        printf("\n------------------------------------------\n");
        printf("TIMING STATS (on %d processed frames)\n", num_frames);
        printf("------------------------------------------\n");
        printf("Image resize time   -> Total: %6.2fms, Average: %6.2fms\n", (double)(total_resize_time/1000000), (double)((total_resize_time/(1000000* num_frames))));
        printf("Tensor resize time  -> Total: %6.2fms, Average: %6.2fms\n", (double)(total_tensor_time/1000000), (double)((total_tensor_time/(1000000* num_frames))));
        printf("Model GPU execution -> Total: %6.2fms, Average: %6.2fms\n", (double)(total_model_time/1000000), (double)((total_model_time/(1000000* num_frames))));
    }

    if (tflite_settings != NULL){
        delete tflite_settings;
    }
}

} //namespace
} //namespace

// -----------------------------------------------------------------------------------------------------------------------------
// This thread runs the mobilenet model
// -----------------------------------------------------------------------------------------------------------------------------
void* ThreadMobileNet(void* data)
{
    tflite::label_image::TFliteMobileNet(data);
    return NULL;
}

void* ThreadMidas(void* data)
{
    tflite::label_image::TFliteMidas(data);
    return NULL;
}
