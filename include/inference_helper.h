#ifndef INFERENCE_HELPER_H
#define INFERENCE_HELPER_H

#include <stdint.h>
#include <modal_pipe.h>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <mutex>
#include <condition_variable>

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/examples/label_image/bitmap_helpers.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/string_util.h"

#ifdef BUILD_865
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#endif

#include "ai_detection.h"
#include "resize.h"

#define MAX_IMAGE_SIZE  12441600  // 4k YUV image size
#define QUEUE_SIZE      32        // max messages to be stored in queue

// TFlite message data
struct TFLiteMessage {
    camera_image_metadata_t metadata;      // image metadata information
    uint8_t image_pixels[MAX_IMAGE_SIZE];  // image pixels
};

// TFlite message queue
struct TFLiteCamQueue {
    TFLiteMessage queue[QUEUE_SIZE];    // camera frame queue
    int insert_idx = 0;                 // next element insert location (between 0 - QUEUE_SIZE)
};

// delegate enum, for code readability
enum DelegateOpt { XNNPACK, GPU, NNAPI };

class InferenceHelper
{
    public:
        // Constructor
        InferenceHelper(char* model_file, char* labels_file, DelegateOpt delegate_choice, bool _en_debug, bool _en_timing);
        // Destructor
        ~InferenceHelper();

        // pre-processing funcs, gets necessary params from loaded model
        bool preprocess_image(camera_image_metadata_t &meta, char* frame, cv::Mat &preprocessed_image, cv::Mat &output_image);

        // generic run_inference, requires input from preprocess_image() above
        bool run_inference(cv::Mat preprocessed_image);

        // post-processing funcs, specific to model type (output tensor format)
        bool postprocess_object_detect(cv::Mat &output_image, std::vector<ai_detection_t>& detections_vector);
        bool postprocess_mono_depth(camera_image_metadata_t &meta, cv::Mat* output_image);

        // summary timing stats
        void print_summary_stats();

        pthread_t               thread;         // model thread handle
        std::mutex              cond_mutex;     // mutex
        std::condition_variable cond_var;       // condition variable

        TFLiteCamQueue       camera_queue;      // camera message queue for the thread

        std::string cam_name;

    private:
        
        // holders for model specific data
        int model_width;
        int model_height;
        int model_channels;

        // cam properties
        int input_width;
        int input_height;

        // delegate ptrs
        DelegateOpt hardware_selection;
        TfLiteDelegate* gpu_delegate;
        #ifdef BUILD_865
        TfLiteDelegate* xnnpack_delegate;
        tflite::StatefulNnApiDelegate* nnapi_delegate;
        #endif

        // only used if running an object detection model
        ai_detection_t detection_data;
        char* labels_location;

        bool en_debug;
        bool en_timing;

        // timing variables
        float total_preprocess_time = 0;
        float total_inference_time = 0;
        float total_postprocess_time = 0;
        uint64_t start_time = 0;
        int num_frames_processed = 0;

        // tflite
        std::unique_ptr<tflite::FlatBufferModel> model;
        std::unique_ptr<tflite::Interpreter> interpreter;
        tflite::ops::builtin::BuiltinOpResolver resolver;

        // mcv resize vars
        uint8_t* resize_output;
        undistort_map_t map;
};

#endif // INFERENCE_HELPER_H
