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

#include "debug_log.h"
#include "external_interface.h"
#include "memory.h"
#include "modal_camera_server_interface.h"
//#include "nnapi_delegate.h"
#include "bitmap_helpers.h"
#include "optional_debug_tools.h"
#include "utils.h"
#include "voxl_tflite_interface.h"
//#include <opencv2/core/core_c.h>
//#include <opencv2/core/types_c.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
//#include "opencv2/opencv.hpp"
//#include <opencv2/core/ocl.hpp>
#include "tcp_utils.hpp"

#define LOG(x) std::cout
// #define FRAME_DUMP
#define STATS_DUMP

void* ThreadMobileNet(void* data);
void* ThreadTflitePydnet(void* data);
void* ThreadSendImageData(void* data);

namespace tflite
{
namespace label_image
{

const int MAX_SENDTCP  = 128;
int g_sendTcpInsertdx  = 0;

const int     MAX_EXT_MESSAGES    = 128; ///<@todo Check this value for being too high
int           g_sendExtMessageIdx = 0;
TFLiteMessage g_sendExtMsg[MAX_EXT_MESSAGES];

// Mutex / Condition Variables for sync between camera frame threads
std::mutex              g_condMutex;
std::condition_variable g_condVar;

struct SendTcpData
{
    int      width;
    int      height;
    int      frameNumber;
    uint64_t timestampNsecs;
    uint8_t* pRgbData;
};

SendTcpData g_sendTcpData[MAX_SENDTCP];

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
            LOG(FATAL) << "Should not reach here!";
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
            LOG(FATAL) << "Should not reach here!";
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
            LOG(INFO) << "GPU acceleration is unsupported on this platform.";
        } 
        else 
        {
            LOG(INFO) << "GPU acceleration is SUPPORTED on this platform\n";
            delegates.emplace("GPU", std::move(delegate));
        }
    }

    if (s->accel) 
    {
        // auto delegate = evaluation::CreateNNAPIDelegate();

        // if (!delegate)
        // {
        //     LOG(INFO) << "NNAPI acceleration is unsupported on this platform.";
        // }
        // else
        // {
        //     delegates.emplace("NNAPI", evaluation::CreateNNAPIDelegate());
        // }
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
        LOG(FATAL) << "Labels file " << file_name << " not found\n";
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
// Send the frame over Tcp
// -----------------------------------------------------------------------------------------------------------------------------
void SendImageData(void* pData)
{
    // TFliteThreadData* pThreadData = (TFliteThreadData*)pData;
    // TcpServer*        pTcpServer  = pThreadData->pTcpServer;

    // int sendTcpProcessIdx = 0;

    // int opts[4] = { 0, 0, 0, 0 };

    // while (pThreadData->stop == false)
    // {
    //     fprintf(stderr, "\n------Sending Tcp server frame %d ... %d %d",
    //             g_sendTcpData[sendTcpProcessIdx].frameNumber, sendTcpProcessIdx, g_sendTcpInsertdx);

    //     if (sendTcpProcessIdx != g_sendTcpInsertdx)
    //     {
    //         fprintf(stderr, "\n------ Tcp sending frame %d ... pointer %p ... %d %d",
    //                 g_sendTcpData[sendTcpProcessIdx].frameNumber,
    //                 g_sendTcpData[sendTcpProcessIdx].pRgbData,
    //                 g_sendTcpData[sendTcpProcessIdx].width,
    //                 g_sendTcpData[sendTcpProcessIdx].height);

    //         pTcpServer->send_message(g_sendTcpData[sendTcpProcessIdx].pRgbData,
    //                                 g_sendTcpData[sendTcpProcessIdx].width*g_sendTcpData[sendTcpProcessIdx].height*3,
    //                                 2,
    //                                 g_sendTcpData[sendTcpProcessIdx].width,
    //                                 g_sendTcpData[sendTcpProcessIdx].height,
    //                                 &opts[0],
    //                                 g_sendTcpData[sendTcpProcessIdx].frameNumber,
    //                                 g_sendTcpData[sendTcpProcessIdx].timestampNsecs);

    //         sendTcpProcessIdx = ((sendTcpProcessIdx + 1) % MAX_SENDTCP);
    //     }
    //     else
    //     {
    //         std::unique_lock<std::mutex> lock(g_condMutex);
    //         g_condVar.wait(lock);
    //     }
    // }
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
    cv::Mat*  pRgbImage[MAX_EXT_MESSAGES];
    cv::Mat resizedImage                     = cv::Mat();
    TFliteThreadData* pThreadData            = (TFliteThreadData*)pData;
    TcpServer* pTcpServer                    = pThreadData->pTcpServer;
    ExternalInterface* pExternalInterface    = NULL;
    ExternalInterfaceData initData;

    if (pTcpServer != NULL)
    {
        for (int i = 0; i < MAX_EXT_MESSAGES; i++)
        {
            pRgbImage[i] = new cv::Mat();
        }
    }
    else
    {
        pRgbImage[0] = new cv::Mat();

        memset(&initData, 0, sizeof(ExternalInterfaceData));
        ///<@todo get this from the thread data
        initData.outputMask = RgbOutputMask;

        pExternalInterface = ExternalInterface::Create(&initData);
    }

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
        LOG(ERROR) << "no model file name\n";
        exit(-1);
    }

    model = tflite::FlatBufferModel::BuildFromFile(s->model_name.c_str());

    if (!model)
    {
        LOG(FATAL) << "\nFailed to mmap model " << s->model_name << "\n";
        exit(-1);
    }

    s->model = model.get();
    LOG(INFO) << "\nLoaded model " << s->model_name << "\n";
    model->error_reporter();
    LOG(INFO) << "Resolved reporter\n";

    tflite::InterpreterBuilder(*model, resolver)(&interpreter);

    if (!interpreter)
    {
        LOG(FATAL) << "Failed to construct interpreter\n";
        exit(-1);
    }

    interpreter->UseNNAPI(s->old_accel);
    interpreter->SetAllowFp16PrecisionForFp32(s->allow_fp16);

    if (s->verbose)
    {
        LOG(INFO) << "tensors size: " << interpreter->tensors_size() << "\n";
        LOG(INFO) << "nodes size: " << interpreter->nodes_size() << "\n";
        LOG(INFO) << "inputs: " << interpreter->inputs().size() << "\n";
        LOG(INFO) << "input(0) name: " << interpreter->GetInputName(0) << "\n";

        int t_size = interpreter->tensors_size();

        for (int i = 0; i < t_size; i++)
        {
            if (interpreter->tensor(i)->name)
              LOG(INFO) << i << ": " << interpreter->tensor(i)->name << ", "
                        << interpreter->tensor(i)->bytes << ", "
                        << interpreter->tensor(i)->type << ", "
                        << interpreter->tensor(i)->params.scale << ", "
                        << interpreter->tensor(i)->params.zero_point << "\n";
        }
    }

    if (s->number_of_threads != -1)
    {
        interpreter->SetNumThreads(s->number_of_threads);
    }

    TfLiteDelegatePtrMap delegates_ = GetDelegates(s);

    for (const auto& delegate : delegates_)
    {
        if (interpreter->ModifyGraphWithDelegate(delegate.second.get()) != kTfLiteOk)
        {
            LOG(FATAL) << "Failed to apply " << delegate.first << " delegate\n";
            break;
        }
        else
        {
            LOG(INFO) << "Applied " << delegate.first << " delegate ";
            break;
        }
    }

    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        LOG(FATAL) << "Failed to allocate tensors!";
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

    gettimeofday(&begin_time, nullptr);

    setpriority(which, tid, nice);

    // Inform the camera frames receiver that tflite processing is ready to receive frames and start processing
    pThreadData->tfliteReady = true;
    fprintf(stderr, "\n------Setting TFLiteThread to ready!! W: %d H: %d C:%d",
            modelImageWidth, modelImageHeight, modelImageChannels);

    int queueProcessIdx = 0;

    while (pThreadData->stop == false)
    {
        if (queueProcessIdx == pThreadData->pMsgQueue->queueInsertIdx)
        {
            std::unique_lock<std::mutex> lock(pThreadData->condMutex);
            pThreadData->condVar.wait(lock);
            continue;
        }

        // Coming here means we have a frame to run through the DNN model
        numFrames++;

        TFLiteMessage* pTFLiteMessage           = &pThreadData->pMsgQueue->queue[queueProcessIdx];
        fprintf(stderr, "\n------Popping index %d frame %d ...... Queue size: %d",
                queueProcessIdx, pTFLiteMessage->pMetadata->frame_id,
                abs(pThreadData->pMsgQueue->queueInsertIdx - queueProcessIdx));

        ///<@todo Create a wrapper for this structure
        camera_image_metadata_t* pImageMetadata = pTFLiteMessage->pMetadata;
        uint8_t*                 pImagePixels   = pTFLiteMessage->pImagePixels;

        int imageWidth    = pImageMetadata->width;
        int imageHeight   = pImageMetadata->height;
        int imageChannels = 3;
        int frameNumber   = pImageMetadata->frame_id;

        gettimeofday(&yuvrgb_start_time, nullptr);
        ///<@todo camera server needs to send packed frames
        // memcpy(pTempYuv, (uint8_t*)pBufferInfo->vaddr, imageWidth*imageHeight);
        // memcpy(pTempYuv, pImagePixels, pImageMetadata->size_bytes);
        // memcpy(pTempYuv+(imageWidth*imageHeight), (uint8_t*)pBufferInfo->craddr, imageWidth*imageHeight/2);

        // RotateNV21(pRotatedYuv, (uint8_t*)pTempYuv, imageWidth, imageHeight, rotation);
        // cv::Mat yuv(imageHeight + imageHeight/2, imageWidth, CV_8UC1, (uchar*)pRotatedYuv);
        cv::Mat yuv(imageHeight + imageHeight/2, imageWidth, CV_8UC1, (uchar*)pImagePixels);
        cv::cvtColor(yuv, *pRgbImage[g_sendTcpInsertdx], CV_YUV2RGB_NV21);
        cv::resize(*pRgbImage[g_sendTcpInsertdx],
                   resizedImage,
                   cv::Size(modelImageWidth, modelImageHeight),
                   0,
                   0,
                   CV_INTER_LINEAR);
        gettimeofday(&yuvrgb_stop_time, nullptr);

        totalYuvRgbTimemsecs += (get_us(yuvrgb_stop_time) - get_us(yuvrgb_start_time)) / 1000;

        uint8_t*               pImageData = (uint8_t*)resizedImage.data;
        const std::vector<int> inputs     = interpreter->inputs();
        const std::vector<int> outputs    = interpreter->outputs();

        // Get input dimension from the input tensor metadata assuming one input only

        int input = interpreter->inputs()[0];

        if (s->verbose)
        {
            // PrintInterpreterState(interpreter.get());
        }

        gettimeofday(&resize_start_time, nullptr);

        switch (interpreter->tensor(input)->type)
        {
            case kTfLiteFloat32:
            fprintf(stderr, "\n------kTfLiteFloat32!!");
              s->input_floating = true;
              resize<float>(interpreter->typed_tensor<float>(input), pImageData,
                            modelImageHeight, modelImageWidth, imageChannels, modelImageHeight,
                            modelImageWidth, modelImageChannels, s);
                          
              break;

            case kTfLiteUInt8:
            fprintf(stderr, "\n------kTfLiteUInt8!!");
              resize<uint8_t>(interpreter->typed_tensor<uint8_t>(input), pImageData,
                              modelImageHeight, modelImageWidth, imageChannels, modelImageHeight,
                              modelImageWidth, modelImageChannels, s);
              break;

            default:
              LOG(FATAL) << "cannot handle input type "
                        << interpreter->tensor(input)->type << " yet";
              exit(-1);
        }

        gettimeofday(&resize_stop_time, nullptr);

        totalResizeTimemsecs += (get_us(resize_stop_time) - get_us(resize_start_time)) / 1000;

        gettimeofday(&start_time, nullptr);

        for (int i = 0; i < s->loop_count; i++)
        {
            if (interpreter->Invoke() != kTfLiteOk)
            {
                LOG(FATAL) << "Failed to invoke tflite!\n";
            }
        }

        gettimeofday(&stop_time, nullptr);
        LOG(INFO) << "GPU invoked \n";
        LOG(INFO) << "average GPU model execution time: "
                  << (get_us(stop_time) - get_us(start_time)) / (s->loop_count * 1000)
                  << " ms \n";

        totalGpuExecutionTimemsecs += (get_us(stop_time) - get_us(start_time)) / 1000;

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

        LOG(INFO) << "Frame: " << frameNumber << ".... Detected Num Classes is: " << detected_numclasses << "\n";

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
                LOG(INFO) << score * 100.0 << "\t" << " Class id:  " << labels[detected_classes[i]]
                          << "\t" << "[ " << left << ", " << top << ", " << right-left << ", " << bottom-top << " ]" << "\n";

                int height = bottom - top;
                int width  = right - left;

                cv::Rect rect(left, top, width, height);
                cv::Point pt(left, top);

                cv::rectangle(*pRgbImage[g_sendTcpInsertdx], rect, cv::Scalar(0, 200, 0), 7);
                cv::putText(*pRgbImage[g_sendTcpInsertdx],
                            labels[detected_classes[i]], pt, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 0, 0), 2);

#ifdef FRAME_DUMP
                char filename[128];
                {
                    sprintf(filename, "/data/misc/camera/frame_%d.bmp", frameNumber);
                    cv::imwrite(filename, *pRgbImage[g_sendTcpInsertdx]);
                }
#endif // FRAME_DUMP
            }
        }

        LOG(INFO) << "\n\n";

        if (pTcpServer == NULL)
        {
            ///<@todo Handle different format types
            // pImageMetadata->bits_per_pixel = 24;
            pImageMetadata->format         = IMAGE_FORMAT_RGB; ///<@todo Fix this to the correct format
            pImageMetadata->size_bytes     = (imageWidth * imageHeight * 3);
            pImageMetadata->stride         = (imageWidth * 3);

            pExternalInterface->BroadcastFrame(OUTPUT_ID_RGB_IMAGE, (char*)pImageMetadata, sizeof(camera_image_metadata_t));
            pExternalInterface->BroadcastFrame(OUTPUT_ID_RGB_IMAGE,
                                               (char*)pRgbImage[g_sendTcpInsertdx]->data,
                                               pImageMetadata->size_bytes);
        }

        queueProcessIdx = ((queueProcessIdx + 1) % MAX_MESSAGES);
    }

#ifdef STATS_DUMP
    struct timeval end_time;
    gettimeofday(&end_time, nullptr);
    LOG(INFO) << "\n\nAverage execution time per frame in msecs: ";
    LOG(INFO) << ((get_us(end_time) - get_us(begin_time)) / 1000) / numFrames << " ms \n";
    LOG(INFO) << "Average resize time per frame msecs: " << totalResizeTimemsecs / numFrames << "\n";
    LOG(INFO) << "Average yuv-->rgb time per frame in msecs: " << totalYuvRgbTimemsecs / numFrames << "\n";
    LOG(INFO) << "\n\n ==== Average GPU model execution time per frame: " << totalGpuExecutionTimemsecs / numFrames << " msecs\n";
#endif // STATS_DUMP

    if (s != NULL)
    {
        delete s;
    }

    if (pTcpServer == NULL)
    {
        if (pRgbImage[0] != NULL)
        {
            delete pRgbImage[0];
            pRgbImage[0] = NULL;
        }
    }
}
  // namespace label_image
  // namespace tflite

// -----------------------------------------------------------------------------------------------------------------------------
// This function runs the tflite model
// -----------------------------------------------------------------------------------------------------------------------------
void TflitePydnet(void* pData)
{
    //STOPS BEFORE ANY OF THIS...
    int modelImageHeight;
    int modelImageWidth;
    int modelImageChannels;
    std::unique_ptr<tflite::FlatBufferModel> model; 
    std::unique_ptr<tflite::Interpreter>     interpreter;
    tflite::ops::builtin::BuiltinOpResolver  resolver;
    tflite::label_image::Settings* s         = NULL;
    TFliteThreadData* pThreadData            = (TFliteThreadData*)pData;
    cv::Mat*  pRgbImage[MAX_EXT_MESSAGES];
    cv::Mat resizedImage                     = cv::Mat();
    TcpServer* pTcpServer                    = pThreadData->pTcpServer;
    ExternalInterface* pExternalInterface    = NULL;
    ExternalInterfaceData initData;

    if (pTcpServer != NULL)
    {
        for (int i = 0; i < MAX_EXT_MESSAGES; i++)
        {
            pRgbImage[i] = new cv::Mat();
        }
    }
    else
    {
        pRgbImage[0] = new cv::Mat();
        memset(&initData, 0, sizeof(ExternalInterfaceData));
        ///<@todo get this from the thread data
        initData.outputMask = RgbOutputMask;
        pExternalInterface = ExternalInterface::Create(&initData);
    }

    s = new tflite::label_image::Settings;


    s->model_name                   = pThreadData->pDnnModelFile;
    s->labels_file_name             = pThreadData->pLabelsFile;
    s->input_bmp_name               = "";
    s->gl_backend                   = 1;
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
    s->input_floating               = true;
    //uint32_t numFrames = 0;

    if (!s->model_name.c_str())
    {
        LOG(ERROR) << "no model file name\n";
        exit(-1);
    }

    model = tflite::FlatBufferModel::BuildFromFile(s->model_name.c_str());

    if (!model)
    {
        LOG(FATAL) << "\nFailed to mmap model " << s->model_name << "\n";
        exit(-1);
    }
    s->model = model.get();
    LOG(INFO) << "\nLoaded model " << s->model_name << "\n";
    model->error_reporter();
    LOG(INFO) << "Resolved reporter\n";
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);

    if (!interpreter)
    {
        LOG(FATAL) << "Failed to construct interpreter\n";
        exit(-1);
    }

    interpreter->UseNNAPI(s->old_accel);
    interpreter->SetAllowFp16PrecisionForFp32(s->allow_fp16);

    if (s->verbose)
    {
        LOG(INFO) << "tensors sizee: " << interpreter->tensors_size() << "\n";
        LOG(INFO) << "nodes size: " << interpreter->nodes_size() << "\n";
        LOG(INFO) << "inputs: " << interpreter->inputs().size() << "\n";
        LOG(INFO) << "input(0) name: " << interpreter->GetInputName(0) << "\n";

        int t_size = interpreter->tensors_size();

        for (int i = 0; i < t_size; i++)
        {
            if (interpreter->tensor(i)->name)
              LOG(INFO) << i << ": " << interpreter->tensor(i)->name << ", "
                        << interpreter->tensor(i)->bytes << ", "
                        << interpreter->tensor(i)->type << ", "
                        << interpreter->tensor(i)->params.scale << ", "
                        << interpreter->tensor(i)->params.zero_point << "\n";
        }
    }

    if (s->number_of_threads != -1)
    {
        interpreter->SetNumThreads(s->number_of_threads);
    }

    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        LOG(FATAL) << "Failed to allocate tensors!";
    }

    TfLiteIntArray* dims = interpreter->tensor(interpreter->inputs()[0])->dims;

    if (s->number_of_threads != -1)
    {
        interpreter->SetNumThreads(s->number_of_threads);
    }
        
    TfLiteDelegatePtrMap delegates_ = GetDelegates(s);

    for (const auto& delegate : delegates_)
    {
        if (interpreter->ModifyGraphWithDelegate(delegate.second.get()) != kTfLiteOk)
        {
            LOG(FATAL) << "Failed to apply " << delegate.first << " delegate\n";
            break;
        }
        else
        {
            LOG(INFO) << "Applied " << delegate.first << " delegate ";
            break;
        }
    }

    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        LOG(FATAL) << "Failed to allocate tensors!";
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
    uint32_t totalResizeTimemsecs = 0;
    uint32_t totalYuvRgbTimemsecs = 0;
    uint32_t totalGpuExecutionTimemsecs = 0;
    uint32_t numFrames = 0;

    gettimeofday(&begin_time, nullptr);


    setpriority(which, tid, nice);


    // Inform the camera frames receiver that tflite processing is ready to receive frames and start processing
    pThreadData->tfliteReady = true;
    fprintf(stderr, "\n------Pydnet model required image width: %d height: %d channels:%d",
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
        if (queueProcessIdx == pThreadData->pMsgQueue->queueInsertIdx)
        {
            std::unique_lock<std::mutex> lock(pThreadData->condMutex);
            pThreadData->condVar.wait(lock);
            continue;
        }
        // Coming here means we have a frame to run through the DNN model
        numFrames++;

        TFLiteMessage* pTFLiteMessage           = &pThreadData->pMsgQueue->queue[queueProcessIdx];
        fprintf(stderr, "\n------Popping index %d frame %d ...... Queue size: %d",
                queueProcessIdx, pTFLiteMessage->pMetadata->frame_id,
                abs(pThreadData->pMsgQueue->queueInsertIdx - queueProcessIdx));

        ///<@todo Create a wrapper for this structure
        camera_image_metadata_t* pImageMetadata = pTFLiteMessage->pMetadata;
        uint8_t*                 pImagePixels   = pTFLiteMessage->pImagePixels;

        int imageWidth    = pImageMetadata->width;
        int imageHeight   = pImageMetadata->height;
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
        totalYuvRgbTimemsecs += (get_us(yuvrgb_stop_time) - get_us(yuvrgb_start_time)) / 1000;

        

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
        totalResizeTimemsecs += (get_us(resize_stop_time) - get_us(resize_start_time)) / 1000;
        gettimeofday(&start_time, nullptr);

        interpreter->Invoke();

        gettimeofday(&stop_time, nullptr);
        LOG(INFO) << "GPU invoked \n";
        LOG(INFO) << "average GPU model execution time: "
                  << (get_us(stop_time) - get_us(start_time)) / (s->loop_count * 1000)
                  << " ms \n";

        totalGpuExecutionTimemsecs += (get_us(stop_time) - get_us(start_time)) / 1000;

        TfLiteTensor* output_locations    = interpreter->tensor(interpreter->outputs()[0]);
        float* depth  = TensorData<float>(output_locations, 0);
        cv::Mat depthImage(384, 640, CV_32FC1, depth);

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
        cv::cvtColor(holder_img, *pRgbImage[g_sendTcpInsertdx], CV_BGR2RGB);      

        

#ifdef FRAME_DUMP
                char filename[128];
                {
                    sprintf(filename, "/data/misc/camera/frame_%d.bmp", frameNumber);
                    cv::imwrite(filename, *pRgbImage[g_sendTcpInsertdx]);
                }
#endif // FRAME_DUMP      

        LOG(INFO) << "\n\n";

        if (pTcpServer == NULL) 
        {
            pImageMetadata->format         = IMAGE_FORMAT_RGB;   
            pImageMetadata->size_bytes     = (imageWidth * imageHeight * 3); 
            pImageMetadata->stride         = (imageWidth * 3); 
            pExternalInterface->BroadcastFrame(OUTPUT_ID_RGB_IMAGE, (char*)pImageMetadata, sizeof(camera_image_metadata_t));
            pExternalInterface->BroadcastFrame(OUTPUT_ID_RGB_IMAGE,
                                        (char*)pRgbImage[g_sendTcpInsertdx]->data,
                                        pImageMetadata->size_bytes);
        }

           queueProcessIdx = ((queueProcessIdx + 1) % MAX_MESSAGES);
        
     }


#ifdef STATS_DUMP
    struct timeval end_time;
    gettimeofday(&end_time, nullptr);
    LOG(INFO) << "\n\nAverage execution time per frame in msecs: ";
    LOG(INFO) << ((get_us(end_time) - get_us(begin_time)) / 1000) / numFrames << " ms \n";
    LOG(INFO) << "Average resize time per frame msecs: " << totalResizeTimemsecs / numFrames << "\n";
    LOG(INFO) << "Average yuv-->rgb time per frame in msecs: " << totalYuvRgbTimemsecs / numFrames << "\n";
    LOG(INFO) << "\n\n ==== Average GPU model execution time per frame: " << totalGpuExecutionTimemsecs / numFrames << " msecs\n";
#endif // STATS_DUMP

    if (s != NULL)
    {
        delete s;
    }

    if (pTcpServer == NULL)
    {
        if (pRgbImage[0] != NULL)
        {
            delete pRgbImage[0];
            pRgbImage[0] = NULL;
        }
    }
}  // namespace tflite
}
}
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

// -----------------------------------------------------------------------------------------------------------------------------
// This thread send the image data
// -----------------------------------------------------------------------------------------------------------------------------
void* ThreadSendImageData(void* pData)
{
    tflite::label_image::SendImageData(pData);
    return NULL;
}
