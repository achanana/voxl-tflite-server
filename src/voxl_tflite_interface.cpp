/*==============================================================================================================================
Copyright 2017:

The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Copyright 2020:

Modified by ModalAI to run the object detection model on live camera frames
==============================================================================================================================*/

#include <getopt.h>     // NOLINT(build/include_order)
#include <sys/time.h>   // NOLINT(build/include_order)
#include <sys/types.h>  // NOLINT(build/include_order)
#include <sys/resource.h>
#include <sys/syscall.h>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <unistd.h>

//#include "voxl_tflite_gpu_object_detect.h"

#include "debug_log.h"
#include "memory.h"
#include <modal_pipe.h>
#include "bitmap_helpers.h"
#include "optional_debug_tools.h"
#include "utils.h"
#include "voxl_tflite_interface.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>

#define MPA_TFLITE_PATH (MODAL_PIPE_DEFAULT_BASE_DIR "tflite/")

void* ThreadMobileNet(void* data);
void* ThreadTflitePydnet(void* data);
void* ThreadSendImageData(void* data);
extern bool en_debug;
extern bool en_timing;

namespace tflite
{
namespace label_image
{

// Mutex / Condition Variables for sync between camera frame threads
std::mutex              g_condMutex;
std::condition_variable g_condVar;

template<typename T>
T* TensorData(TfLiteTensor* tensor, int batch_index);

// -----------------------------------------------------------------------------------------------------------------------------
// Gets the float* tensor data pointer
// -----------------------------------------------------------------------------------------------------------------------------
template<>
float* TensorData(TfLiteTensor* tensor, int batch_index)
{
    int nelems = 1;

    for (int i = 1; i < tensor->dims->size; i++)
    {
        nelems *= tensor->dims->data[i];
    }

    switch (tensor->type)
    {
        case kTfLiteFloat32:
            return tensor->data.f + nelems * batch_index;
        default:
            VOXL_LOG_FATAL("Should not reach here!\n");
    }

    return nullptr;
}

// -----------------------------------------------------------------------------------------------------------------------------
// Gets the uint8_t* tensor data pointer
// -----------------------------------------------------------------------------------------------------------------------------
template<>
uint8_t* TensorData(TfLiteTensor* tensor, int batch_index)
{
    int nelems = 1;

    for (int i = 1; i < tensor->dims->size; i++)
    {
        nelems *= tensor->dims->data[i];
    }

    switch (tensor->type)
    {
        case kTfLiteUInt8:
            return tensor->data.uint8 + nelems * batch_index;
        default:
            VOXL_LOG_FATAL("Should not reach here!\n");
    }

    return nullptr;
}

// -----------------------------------------------------------------------------------------------------------------------------
// Gets the time in microsecs
// -----------------------------------------------------------------------------------------------------------------------------
double get_us(struct timeval t)
{
    return (t.tv_sec * 1000000 + t.tv_usec);
}

using TfLiteDelegatePtr    = tflite::Interpreter::TfLiteDelegatePtr;
using TfLiteDelegatePtrMap = std::map<std::string, TfLiteDelegatePtr>;

// -----------------------------------------------------------------------------------------------------------------------------
// Creates a GPU delegate
// -----------------------------------------------------------------------------------------------------------------------------
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
    // return Interpreter::TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});;
}

// -----------------------------------------------------------------------------------------------------------------------------
// Gets all the available delegates
// -----------------------------------------------------------------------------------------------------------------------------
TfLiteDelegatePtrMap GetDelegates(Settings* s)
{
    TfLiteDelegatePtrMap delegates;

    if (s->gl_backend)
    {
        auto delegate = CreateGPUDelegate(s);

        if (!delegate)
        {
            VOXL_LOG_ERROR("GPU acceleration is unsupported on this platform.\n");
        }
        else
        {
            VOXL_LOG_INFO("GPU acceleration is SUPPORTED on this platform\n");
            delegates.emplace("GPU", std::move(delegate));
        }
    }

    return delegates;
}

// -----------------------------------------------------------------------------------------------------------------------------
// Takes a file name, and loads a list of labels from it, one per line, and returns a vector of the strings. It pads with empty
// strings so the length of the result is a multiple of 16, because our model expects that.
// -----------------------------------------------------------------------------------------------------------------------------
TfLiteStatus ReadLabelsFile(const string& file_name,
                            std::vector<string>* result,
                            size_t* found_label_count)
{
    std::ifstream file(file_name);

    if (!file)
    {
        VOXL_LOG_FATAL("Labels file %s not found\n", file_name.c_str());
        return kTfLiteError;
    }

    result->clear();
    string line;

    while (std::getline(file, line))
    {
        result->push_back(line);
    }

    *found_label_count = result->size();
    const int padding = 16;

    while (result->size() % padding)
    {
        result->emplace_back();
    }

    return kTfLiteOk;
}

// -----------------------------------------------------------------------------------------------------------------------------
// This function runs the object detection model on every live camera frame
// -----------------------------------------------------------------------------------------------------------------------------
void TFliteMobileNet(void* pData)
{
    int modelImageHeight;
    int modelImageWidth;
    int modelImageChannels;
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter>     interpreter;
    tflite::ops::builtin::BuiltinOpResolver  resolver;
    ///<@todo delete this
    tflite::label_image::Settings* s         = NULL;
    cv::Mat*  rgbImage                       = new cv::Mat();
    cv::Mat resizedImage                     = cv::Mat();
    TFliteThreadData* pThreadData            = (TFliteThreadData*)pData;
    pipe_info_t tflite_pipe                  = {"tflite", MPA_TFLITE_PATH, "camera_image_metadata_t", "voxl-tflite-server", 16*1024*1024, 0};

    pipe_server_create(OUTPUT_ID_RGB_IMAGE, tflite_pipe, 0);

    s = new tflite::label_image::Settings;

    s->model_name                   = pThreadData->pDnnModelFile;
    s->labels_file_name             = pThreadData->pLabelsFile;
    s->input_bmp_name               = "";
    s->gl_backend                   = 1; ///<@todo Is there a CL backend?
    s->number_of_threads            = 4;
    s->allow_fp16                   = 1;
    s->input_mean                   = 127;
    s->accel                        = 0;
    s->old_accel                    = 0;
    s->max_profiling_buffer_entries = 0;
    s->profiling                    = 0;
    s->verbose                      = 0;
    s->number_of_warmup_runs        = 0;
    s->loop_count                   = 1;

    if (!s->model_name.c_str())
    {
        VOXL_LOG_ERROR("FATAL: no model file name\n");
        exit(-1);
    }

    model = tflite::FlatBufferModel::BuildFromFile(s->model_name.c_str());

    if (!model)
    {
        VOXL_LOG_FATAL("FATAL: Failed to mmap model %s\n", s->model_name.c_str());
        exit(-1);
    }

    s->model = model.get();
    VOXL_LOG_INFO("Loaded model %s\n", s->model_name.c_str());
    model->error_reporter();
    VOXL_LOG_INFO("Resolved reporter\n");

    tflite::InterpreterBuilder(*model, resolver)(&interpreter);

    if (!interpreter)
    {
        VOXL_LOG_FATAL("Failed to construct interpreter\n");
        exit(-1);
    }

    interpreter->UseNNAPI(s->old_accel);
    interpreter->SetAllowFp16PrecisionForFp32(s->allow_fp16);

    if (s->number_of_threads != -1)
    {
        interpreter->SetNumThreads(s->number_of_threads);
    }

    TfLiteDelegatePtrMap delegates_ = GetDelegates(s);

    for (const auto& delegate : delegates_)
    {
        if (interpreter->ModifyGraphWithDelegate(delegate.second.get()) != kTfLiteOk)
        {
            VOXL_LOG_INFO("Failed to apply delegate\n");
            break;
        }
        else
        {
            VOXL_LOG_INFO("Applied delegate \n");
            break;
        }
    }

    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        VOXL_LOG_FATAL("Failed to allocate tensors!\n");
        exit(-1);
    }

    TfLiteIntArray* dims = interpreter->tensor(interpreter->inputs()[0])->dims;

    modelImageHeight   = dims->data[1];
    modelImageWidth    = dims->data[2];
    modelImageChannels = dims->data[3];

    // Set thread priority
    pid_t tid = syscall(SYS_gettid);
    int which = PRIO_PROCESS;
    int nice  = -15;

    struct timeval begin_time;
    struct timeval start_time, stop_time;
    struct timeval resize_start_time, resize_stop_time;
    struct timeval yuvrgb_start_time, yuvrgb_stop_time;
    uint32_t totalResizeTimemsecs = 0;
    uint32_t totalYuvRgbTimemsecs = 0;
    uint32_t totalGpuExecutionTimemsecs = 0;
    uint32_t numFrames = 0;
    camera_image_metadata_t meta;
    uint8_t* pImagePixels;

    gettimeofday(&begin_time, nullptr);

    setpriority(which, tid, nice);

    // Inform the camera frames receiver that tflite processing is ready to receive frames and start processing
    pThreadData->tfliteReady = true;
    fprintf(stderr, "\n------Setting TFLiteThread to ready!! W: %d H: %d C:%d\n",
            modelImageWidth, modelImageHeight, modelImageChannels);

    int queueProcessIdx = 0;

    while (pThreadData->stop == false)
    { //(!(en_debug) || !(en_timing) || pipe_server_get_num_clients(OUTPUT_ID_RGB_IMAGE) == 0) &&
        if (queueProcessIdx == pThreadData->pMsgQueue->queueInsertIdx)
        {
            std::unique_lock<std::mutex> lock(pThreadData->condMutex);
            pThreadData->condVar.wait(lock);
            continue;
        }
        if (((!en_debug) && (!en_timing))){
            if (pipe_server_get_num_clients(OUTPUT_ID_RGB_IMAGE) == 0 ){
                continue;
            }
        }


        // Coming here means we have a frame to run through the DNN model
        numFrames++;
        TFLiteMessage* pTFLiteMessage           = &pThreadData->pMsgQueue->queue[queueProcessIdx];
        if (en_debug){
            fprintf(stderr, "\n------Popping index %d frame %d ...... Queue size: %d\n",
                queueProcessIdx, pTFLiteMessage->metadata.frame_id,
                abs(pThreadData->pMsgQueue->queueInsertIdx - queueProcessIdx));
        }

        ///<@todo Create a wrapper for this structure
        meta = pTFLiteMessage->metadata;
        pImagePixels   = pTFLiteMessage->imagePixels;

        int imageWidth    = meta.width;
        int imageHeight   = meta.height;
        int imageChannels = 3;

        gettimeofday(&yuvrgb_start_time, nullptr);
        if (meta.format == IMAGE_FORMAT_NV12){
            cv::Mat yuv(imageHeight + imageHeight/2, imageWidth, CV_8UC1, (uchar*)pImagePixels);
            cv::cvtColor(yuv, *rgbImage, CV_YUV2RGB_NV21); // time + opencl
        }
        else {
            cv::Mat yuv(imageHeight, imageWidth, CV_8UC1, (uchar*)pImagePixels);
            cv::Mat in[] = {yuv, yuv, yuv};
            cv::merge(in, 3, *rgbImage);
        }

        // look into how it is cropped + timing
        cv::resize(*rgbImage,
               resizedImage,
               cv::Size(modelImageWidth, modelImageHeight),
               0,
               0,
               CV_INTER_LINEAR);

        gettimeofday(&yuvrgb_stop_time, nullptr);
        totalYuvRgbTimemsecs = (get_us(yuvrgb_stop_time) - get_us(yuvrgb_start_time)) / 1000;

        uint8_t*               pImageData = (uint8_t*)resizedImage.data;
        const std::vector<int> inputs     = interpreter->inputs();
        const std::vector<int> outputs    = interpreter->outputs();

        // Get input dimension from the input tensor metadata assuming one input only

        int input = interpreter->inputs()[0];

        if (s->verbose)
        {
            PrintInterpreterState(interpreter.get());
        }

        gettimeofday(&resize_start_time, nullptr);

        switch (interpreter->tensor(input)->type)
        {
            case kTfLiteFloat32:
                s->input_floating = true;
                resize<float>(interpreter->typed_tensor<float>(input), pImageData,
                                modelImageHeight, modelImageWidth, imageChannels, modelImageHeight,
                                modelImageWidth, modelImageChannels, s);
                break;

            case kTfLiteUInt8:
                resize<uint8_t>(interpreter->typed_tensor<uint8_t>(input), pImageData,
                                modelImageHeight, modelImageWidth, imageChannels, modelImageHeight,
                                modelImageWidth, modelImageChannels, s);
                break;

            default:
            exit(-1);
        }

        gettimeofday(&resize_stop_time, nullptr);

        totalResizeTimemsecs = (get_us(resize_stop_time) - get_us(resize_start_time)) / 1000;

        gettimeofday(&start_time, nullptr);

        for (int i = 0; i < s->loop_count; i++)
        {
            if (interpreter->Invoke() != kTfLiteOk)
            {
                VOXL_LOG_FATAL("Failed to invoke tflite!\n");
            }
        }

        gettimeofday(&stop_time, nullptr);

        totalGpuExecutionTimemsecs = (get_us(stop_time) - get_us(start_time)) / 1000;

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

        if (ReadLabelsFile(s->labels_file_name, &labels, &label_count) != kTfLiteOk)
        {
            exit(-1);
        }

        for (int i = 0; i < detected_numclasses; i++)
        {
            const float score  = detected_scores[i];
            const int   top    = detected_locations[4 * i + 0] * imageHeight;
            const int   left   = detected_locations[4 * i + 1] * imageWidth;
            const int   bottom = detected_locations[4 * i + 2] * imageHeight;
            const int   right  = detected_locations[4 * i + 3] * imageWidth;

            // Check for object detection confidence of 60% or more
            if (score > 0.6f)
            {
                if (en_debug){
                    std::cout << "Detected: " << labels[detected_classes[i]] <<  ", Confidence: " << score << std::endl;
                }
                int height = bottom - top;
                int width  = right - left;

                cv::Rect rect(left, top, width, height);
                cv::Point pt(left, top);

                cv::rectangle(*rgbImage, rect, cv::Scalar(0, 200, 0), 7);
                cv::putText(*rgbImage,
                            labels[detected_classes[i]], pt, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 2);

            }
        }

        if (en_timing){
            std::cout << std::endl <<  "Total resize time: " << totalResizeTimemsecs << "ms" << std::endl;
            std::cout << "Total YuvRGB time: " << totalYuvRgbTimemsecs << "ms" << std::endl;
            std::cout << "Total GPU time: " << totalGpuExecutionTimemsecs << "ms" << std::endl;
        }
        ///<@todo Handle different format types
        meta.format         = IMAGE_FORMAT_RGB;
        meta.size_bytes     = (imageHeight * imageWidth * 3);
        meta.stride         = (modelImageWidth * 3);
        if( rgbImage->data != NULL){
                pipe_server_write_camera_frame(OUTPUT_ID_RGB_IMAGE, meta, (char*)rgbImage->data);
        }

        queueProcessIdx = ((queueProcessIdx + 1) % MAX_MESSAGES);
    }

    if (s != NULL)
    {
        delete s;
    }

    if (rgbImage != NULL)
    {
        delete rgbImage;
        rgbImage = NULL;
    }
    pipe_server_close_all();
}

// -----------------------------------------------------------------------------------------------------------------------------
// This function runs the pydnet model
// -----------------------------------------------------------------------------------------------------------------------------
void TflitePydnet(void* pData)
{
    int modelImageHeight;
    int modelImageWidth;
    int modelImageChannels;
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter>     interpreter;
    tflite::ops::builtin::BuiltinOpResolver  resolver;
    tflite::label_image::Settings* s         = NULL;
    TFliteThreadData* pThreadData            = (TFliteThreadData*)pData;
    cv::Mat*  rgbImage                      = new cv::Mat();
    cv::Mat resizedImage                     = cv::Mat();
    pipe_info_t tflite_pipe                  = {"tflite", MPA_TFLITE_PATH, "camera_image_metadata_t", "voxl-tflite-server", 16*1024*1024, 0};

    pipe_server_create(OUTPUT_ID_RGB_IMAGE, tflite_pipe, 0);

    s = new tflite::label_image::Settings;


    s->model_name                   = pThreadData->pDnnModelFile;
    s->labels_file_name             = pThreadData->pLabelsFile;
    s->input_bmp_name               = "";
    s->gl_backend                   = 1;
    s->number_of_threads            = 8;
    s->allow_fp16                   = 1;
    s->input_mean                   = 127;
    s->accel                        = 1;
    s->old_accel                    = 0;
    s->max_profiling_buffer_entries = 0;
    s->profiling                    = 0;
    s->verbose                      = 0;
    s->number_of_warmup_runs        = 0;
    s->loop_count                   = 1;
    s->input_floating               = true;


    if (!s->model_name.c_str())
    {
        VOXL_LOG_ERROR("no model file name\n");
        exit(-1);
    }

    model = tflite::FlatBufferModel::BuildFromFile(s->model_name.c_str());

    if (!model)
    {
        VOXL_LOG_FATAL("Failed to mmap model %s\n", s->model_name.c_str());
        exit(-1);
    }
    s->model = model.get();
    VOXL_LOG_INFO("Loaded model %s\n", s->model_name.c_str());
    model->error_reporter();
    VOXL_LOG_INFO("Resolved reporter\n");
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);

    if (!interpreter)
    {
        VOXL_LOG_FATAL("FATAL: Failed to construct interpreter\n");
        exit(-1);
    }

    interpreter->UseNNAPI(s->old_accel);
    interpreter->SetAllowFp16PrecisionForFp32(s->allow_fp16);

    if (s->number_of_threads != -1)
    {
        interpreter->SetNumThreads(s->number_of_threads);
    }

    VOXL_LOG_INFO("WARNING: Pydnet requires ~1 minute to initialize the graph before processing frames.\n");

    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        VOXL_LOG_FATAL("Failed to allocate tensors!\n");
    }

    TfLiteIntArray* dims = interpreter->tensor(interpreter->inputs()[0])->dims;

    if (s->number_of_threads != -1)
    {
        interpreter->SetNumThreads(s->number_of_threads);
    }

    TfLiteDelegatePtrMap delegates_ = GetDelegates(s);

    for (const auto& delegate : delegates_)
    {
        interpreter->ModifyGraphWithDelegate(delegate.second.get());
    }

    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        VOXL_LOG_FATAL("Failed to allocate tensors!\n");
    }

    modelImageHeight   = dims->data[1];
    modelImageWidth    = dims->data[2];
    modelImageChannels = dims->data[3];

       // Set thread priority
    pid_t tid = syscall(SYS_gettid);
    int which = PRIO_PROCESS;
    int nice  = -15;

    struct timeval begin_time;
    struct timeval start_time, stop_time;
    struct timeval resize_start_time, resize_stop_time;
    struct timeval yuvrgb_start_time, yuvrgb_stop_time;
    struct timeval colormap_start_time, colormap_end_time;
    uint32_t totalResizeTimemsecs = 0;
    uint32_t totalYuvRgbTimemsecs = 0;
    uint32_t totalGpuExecutionTimemsecs = 0;
    uint32_t totalColorMapTimesecs = 0;
    uint32_t numFrames = 0;

    gettimeofday(&begin_time, nullptr);


    setpriority(which, tid, nice);

    // Inform the camera frames receiver that tflite processing is ready to receive frames and start processing
    pThreadData->tfliteReady = true;
    VOXL_LOG_ERROR("------Pydnet model required image width: %d height: %d channels:%d\n",
            modelImageWidth, modelImageHeight, modelImageChannels);

    cv::Mat _input;
    cv::Mat _output;
    cv::Mat _res_img;
    cv::Mat _resized_img;
    cv::Mat _resized_img_1;
    cv::Mat _mask;
    cv::Size _input_size;

    _input_size = cv::Size(modelImageWidth, modelImageHeight);
    cv::Mat masked_img = cv::Mat();

    int queueProcessIdx = 0;

    while (pThreadData->stop == false)
    {
        if ((!(en_debug) || !(en_timing) || pipe_server_get_num_clients(OUTPUT_ID_RGB_IMAGE) == 0) && queueProcessIdx == pThreadData->pMsgQueue->queueInsertIdx)
        {
            std::unique_lock<std::mutex> lock(pThreadData->condMutex);
            pThreadData->condVar.wait(lock);
            continue;
        }
        // Coming here means we have a frame to run through the DNN model
        numFrames++;

        TFLiteMessage* pTFLiteMessage           = &pThreadData->pMsgQueue->queue[queueProcessIdx];
        if (en_debug){
        VOXL_LOG_ERROR("------Popping index %d frame %d ...... Queue size: %d\n",
                queueProcessIdx, pTFLiteMessage->metadata.frame_id,
                abs(pThreadData->pMsgQueue->queueInsertIdx - queueProcessIdx));
        }

        ///<@todo Create a wrapper for this structure
        camera_image_metadata_t pImageMetadata =  pTFLiteMessage->metadata;
        uint8_t*                 pImagePixels   = pTFLiteMessage->imagePixels;

        int imageWidth    = pImageMetadata.width;
        int imageHeight   = pImageMetadata.height;
        int imageChannels = 3;
        cv::Mat colored_img;


        gettimeofday(&yuvrgb_start_time, nullptr);

        cv::Mat yuv(imageHeight + imageHeight/2, imageWidth, CV_8UC1, (uchar*)pImagePixels);
        cv::cvtColor(yuv, colored_img, CV_YUV2RGB_NV12);


        cv::resize(colored_img,
                   resizedImage,
                   cv::Size(modelImageWidth, modelImageHeight),
                   0,
                   0,
                   CV_INTER_LINEAR);
        gettimeofday(&yuvrgb_stop_time, nullptr);
        totalYuvRgbTimemsecs = (get_us(yuvrgb_stop_time) - get_us(yuvrgb_start_time)) / 1000;



        uint8_t*               pImageData = (uint8_t*)resizedImage.data;
        const std::vector<int> inputs     = interpreter->inputs();
        const std::vector<int> outputs    = interpreter->outputs();

        int input = interpreter->inputs()[0];

        gettimeofday(&resize_start_time, nullptr);

        s->input_floating = true;
        resize<float>(interpreter->typed_tensor<float>(input), pImageData,
                        modelImageHeight, modelImageWidth, imageChannels, modelImageHeight,
                        modelImageWidth, modelImageChannels, s);

        gettimeofday(&resize_stop_time, nullptr);
        totalResizeTimemsecs = (get_us(resize_stop_time) - get_us(resize_start_time)) / 1000;
        gettimeofday(&start_time, nullptr);

        interpreter->Invoke();

        gettimeofday(&stop_time, nullptr);

        totalGpuExecutionTimemsecs = (get_us(stop_time) - get_us(start_time)) / 1000;

        TfLiteTensor* output_locations    = interpreter->tensor(interpreter->outputs()[0]);
        float* depth  = TensorData<float>(output_locations, 0);
        cv::Mat depthImage(384, 640, CV_32FC1, depth);

        gettimeofday(&colormap_start_time, nullptr);

        /// Noralization
        double minVal;
        double maxVal;
        cv::minMaxLoc(depthImage, &minVal, &maxVal);
        depthImage = (depthImage - minVal) / (maxVal - minVal);
        depthImage = depthImage * 255.0;
        ///

        depthImage.convertTo(_mask, CV_8UC1);

        cv::Mat map_img;
        cv::Mat holder_img;
        applyColorMap(_mask, map_img, cv::COLORMAP_PLASMA);
        cv::resize(map_img, holder_img, cv::Size(640, 480), 0, 0, cv::INTER_NEAREST);
        cv::cvtColor(holder_img, *rgbImage, CV_BGR2RGB);
        gettimeofday(&colormap_end_time, nullptr);
        totalColorMapTimesecs += (get_us(colormap_end_time) - get_us(colormap_start_time)) / 1000;

        if (en_timing){
            std::cout << std::endl <<  "Total resize time: " << totalResizeTimemsecs << "ms" << std::endl;
            std::cout << "Total YuvRGB time: " << totalYuvRgbTimemsecs << "ms" << std::endl;
            std::cout << "Total GPU time: " << totalGpuExecutionTimemsecs << "ms" <<  std::endl;
            std::cout << "Total Colormap time: " << totalColorMapTimesecs << "ms" << std::endl;
        }

        pImageMetadata.format         = IMAGE_FORMAT_RGB;
        pImageMetadata.size_bytes     = (imageWidth * imageHeight * 3);
        pImageMetadata.stride         = (imageWidth * 3);
        pipe_server_write_camera_frame(OUTPUT_ID_RGB_IMAGE, pImageMetadata, (char*)rgbImage->data);;

        queueProcessIdx = ((queueProcessIdx + 1) % MAX_MESSAGES);
     }


    if (s != NULL)
    {
        delete s;
    }

    if (rgbImage != NULL)
    {
        delete rgbImage;
        rgbImage = NULL;
    }
    pipe_server_close_all();
}


} // namespace labelimage
} // namespace tflite
// -----------------------------------------------------------------------------------------------------------------------------
// This thread runs the pydnet model
// -----------------------------------------------------------------------------------------------------------------------------
void* ThreadTflitePydnet(void* pData)
{
    tflite::label_image::TflitePydnet(pData);
    return NULL;
}

// -----------------------------------------------------------------------------------------------------------------------------
// This thread runs the tensorflow model
// -----------------------------------------------------------------------------------------------------------------------------
void* ThreadMobileNet(void* pData)
{
    tflite::label_image::TFliteMobileNet(pData);
    return NULL;
}
