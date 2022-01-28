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


#include <getopt.h>
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <thread>
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/resource.h>
#include "config_file.h"
#include "inference_helper.h"


#define PROCESS_NAME          "voxl-tflite-server"
#define IMAGE_CH              0
#define DETECTION_CH          1
#define HIRES_PIPE            "/run/mpa/hires/"
#define TFLITE_IMAGE_PATH     (MODAL_PIPE_DEFAULT_BASE_DIR "tflite/")
#define TFLITE_DETECTION_PATH (MODAL_PIPE_DEFAULT_BASE_DIR "tflite_data/")

char* coco_labels  = (char*)"/usr/bin/dnn/coco_labels.txt";
bool en_debug = false;
bool en_timing = false;

InferenceHelper* inf_helper;

enum PostProcessType { OBJECT_DETECT, MONO_DEPTH };
PostProcessType post_type;


void _print_usage()
{
    printf("\nCommand line arguments are as follows:\n\n");
    printf("-c, --config    : load the config file only, for use by the config wizard\n");
    printf("-d, --debug     : enable verbose debug output (Default: Off)\n");
    printf("-t, --timing    : enable timing output for model operations (Default: Off)\n");
    printf("-h              : Print this help message\n");
}

static bool _parse_opts(int argc, char* argv[])
{
    static struct option long_options[] =
    {
        {"config",     no_argument, 0, 'm'},
        {"debug",      no_argument, 0, 'c'},
        {"timing",     no_argument, 0, 't'},
        {"help",       no_argument, 0, 'h'},
        {0, 0, 0}
    };

    while (1){
		int option_index = 0;
		int c = getopt_long(argc, argv, "cdtph", long_options, &option_index);

		if (c == -1) break; // Detect the end of the options.

		switch (c){
		case 0:
			// for long args without short equivalent that just set a flag
            // nothing left to do so just break.
			if (long_options[option_index].flag != 0) break;
			break;

		case 'c':
			config_file_read();
            exit(0);

		case 'd':
			printf("Enabling debug mode\n");
			en_debug = true;
			break;

        case 't':
			printf("Enabling timing mode\n");
			en_timing = true;
			break;

		case 'h':
			_print_usage();
			return true;

		default:
			// Print the usage if there is an incorrect command line option
			_print_usage();
			return true;
		}
	}
	return false;
}

// timing helper
uint64_t rc_nanos_monotonic_time()
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ((uint64_t)ts.tv_sec*1000000000)+ts.tv_nsec;
}

static void* inference_worker(void* data)
{
    // Set thread priority
    pid_t tid = syscall(SYS_gettid);
    int which = PRIO_PROCESS;
    int nice  = -15;
    setpriority(which, tid, nice);

    // keep track of where we are in terms of processing the tflite camera queue
    int queue_index = 0;

    while(main_running){
        // first, check if anything new has been added to the queue
        if (queue_index == inf_helper->camera_queue.insert_idx){
            std::unique_lock<std::mutex> lock(inf_helper->cond_mutex);
            inf_helper->cond_var.wait(lock);
            continue;
        }

        // grab the frame and bump our queue index, making sure its within queue size
        TFLiteMessage* new_frame = &inf_helper->camera_queue.queue[queue_index];
        queue_index = ((queue_index + 1) % QUEUE_SIZE);

        cv::Mat preprocessed_image, output_image;

        if (!inf_helper->preprocess_image(new_frame->metadata, (char*)new_frame->image_pixels, preprocessed_image, output_image)) continue;

        if (!inf_helper->run_inference(preprocessed_image)) continue;

        if (post_type == OBJECT_DETECT){
            std::vector<ai_detection_t> detections;
            if (!inf_helper->postprocess_object_detect(output_image, detections)) continue;

            if (!detections.empty()){
                for (unsigned int i = 0; i < detections.size(); i++){
                    pipe_server_write(DETECTION_CH, (char*)&detections[i], sizeof(ai_detection_t));
                }
            } 
            pipe_server_write_camera_frame(IMAGE_CH, new_frame->metadata, (char*)output_image.data);
        }
        else if (post_type == MONO_DEPTH){
            if (!inf_helper->postprocess_mono_depth(new_frame->metadata, &output_image)) continue;
            pipe_server_write_camera_frame(IMAGE_CH, new_frame->metadata, (char*)output_image.data);
        }
    }
    
    return NULL;
}

static void _camera_connect_cb(__attribute__((unused)) int ch, __attribute__((unused)) void *context){
    printf("Connected to camera server\n");
}

static void _camera_disconnect_cb(__attribute__((unused)) int ch, __attribute__((unused)) void *context){
    fprintf(stderr, "Disonnected from camera server\n");
}

static void _camera_helper_cb(__attribute__((unused))int ch, camera_image_metadata_t meta, char* frame, void* context){
    static int n_skipped = 0;
    if (n_skipped < skip_n_frames){
		n_skipped++;
		return;
	}
    if (pipe_client_bytes_in_pipe(ch)>0){
        n_skipped++;
		if(en_debug) fprintf(stderr, "WARNING, skipping frame on channel %d due to frame backup\n", ch);
		return;
    }
    if (!en_debug && !en_timing){
        if (!pipe_server_get_num_clients(IMAGE_CH) && !pipe_server_get_num_clients(DETECTION_CH))
            return;
    }

    if (meta.size_bytes > MAX_IMAGE_SIZE){
        fprintf(stderr, "Model cannot process an image with %d bytes\n", meta.size_bytes);
        return;
    }

    int queue_ind = inf_helper->camera_queue.insert_idx;

    TFLiteMessage* camera_message = &inf_helper->camera_queue.queue[queue_ind];

    camera_message->metadata     = meta;
    memcpy(camera_message->image_pixels, (uint8_t*)frame, meta.size_bytes);

    inf_helper->camera_queue.insert_idx = ((queue_ind + 1) % QUEUE_SIZE);
    inf_helper->cond_var.notify_all();

    return;
}

int main(int argc, char *argv[])
{
    if (_parse_opts(argc, argv)){
		return -1;
	}

    ////////////////////////////////////////////////////////////////////////////////
	// gracefully handle an existing instance of the process and associated PID file
	////////////////////////////////////////////////////////////////////////////////

    // make sure another instance isn't running
	// if return value is -3 then a background process is running with
	// higher privaledges and we couldn't kill it, in which case we should
	// not continue or there may be hardware conflicts. If it returned -4
	// then there was an invalid argument that needs to be fixed.
    if(kill_existing_process(PROCESS_NAME, 2.0)<-2) return -1;

    // start signal handler so we can exit cleanly
	if(enable_signal_handler()==-1){
		fprintf(stderr,"ERROR: failed to start signal handler\n");
		return(-1);
	}

    // make PID file to indicate your project is running
	// due to the check made on the call to rc_kill_existing_process() above
	// we can be fairly confident there is no PID file already and we can
	// make our own safely.
	make_pid_file(PROCESS_NAME);

    // disable garbage multithreading
    cv::setNumThreads(1);

    ////////////////////////////////////////////////////////////////////////////////
	// load config
	////////////////////////////////////////////////////////////////////////////////
	if(config_file_read()){
		return -1;
	}
	config_file_print();

    ////////////////////////////////////////////////////////////////////////////////
	// initialize InferenceHelper
	////////////////////////////////////////////////////////////////////////////////
    DelegateOpt opt_ = GPU;     // default for MAI models
    if (!strcmp(delegate, "xnnpack")) opt_ = XNNPACK;
    else if (!strcmp(delegate, "nnapi")) opt_ = NNAPI;

    // create our inference helper!
    inf_helper = new InferenceHelper(model, coco_labels, opt_, en_debug, en_timing);

    // store cam name
    std::string full_path(input_pipe);
    std::string cam_name(full_path.substr(full_path.rfind("/", full_path.size() - 2) + 1));
    cam_name.pop_back();

    inf_helper->cam_name = cam_name;

    // set postprocess type
    if (!strcmp(model, "/usr/bin/dnn/ssdlite_mobilenet_v2_coco.tflite")) post_type = OBJECT_DETECT;
    else if (!strcmp(model, "/usr/bin/dnn/midas_v2.tflite")){
        fprintf(stderr,"\n\nWARNING: Selected Midas V2 model which requires hires input. Overriding input pipe to hires\n\n");
        snprintf(input_pipe, sizeof(input_pipe), "%s", HIRES_PIPE);
        post_type = MONO_DEPTH;
    }
    else{
        fprintf(stderr, "WARNING: Unknown model type provided! Defaulting post-process to object detection.\n");
        post_type = OBJECT_DETECT;
    }

    main_running = 1;

    fprintf(stderr, "\n------VOXL TFLite Server------\n\n");

    // Start the thread that will run the tensorflow lite model on live camera frames.
    pthread_attr_t thread_attributes;
    pthread_attr_init(&thread_attributes);
    pthread_attr_setdetachstate(&thread_attributes, PTHREAD_CREATE_JOINABLE);

    pthread_create(&(inf_helper->thread), &thread_attributes, inference_worker, NULL);

    // fire up our camera server connection
    int ch = pipe_client_get_next_available_channel();

    pipe_client_set_connect_cb(ch, _camera_connect_cb, NULL);
    pipe_client_set_disconnect_cb(ch, _camera_disconnect_cb, NULL);
    pipe_client_set_camera_helper_cb(ch, _camera_helper_cb, NULL);

    if(pipe_client_open(ch, input_pipe, PROCESS_NAME, CLIENT_FLAG_EN_CAMERA_HELPER, 0)){
        fprintf(stderr, "Failed to open pipe: %s\n", input_pipe);
        return -1;
    }

    // open our output pipes
    pipe_info_t image_pipe = {"tflite", TFLITE_IMAGE_PATH, "camera_image_metadata_t", PROCESS_NAME, 16*1024*1024, 0};
    pipe_server_create(IMAGE_CH, image_pipe, 0);
    pipe_info_t detection_pipe = {"tflite_data", TFLITE_DETECTION_PATH, "ai_detection_t", PROCESS_NAME, 16*1024, 0};
    pipe_server_create(DETECTION_CH, detection_pipe, 0);

    while(main_running){
        usleep(5000000);
    }

    pipe_client_close_all();
    pipe_server_close_all();
    
    fprintf(stderr, "\nStopping the application\n");

    inf_helper->cond_var.notify_all();
    pthread_join(inf_helper->thread, NULL);

    delete(inf_helper);
    return 0;
}
