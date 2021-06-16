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


#include <string.h>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <list>
#include "threads.h"


// model thread
extern void* ThreadMobileNet(void* data);

// forward declaration
static void _cam_helper_cb(__attribute__((unused))int ch,
                                                  camera_image_metadata_t meta,
                                                  char* frame,
                                                  void* context);

////////////////////////////////////////////////////////////////////////////////
// Constructor: starts thread and connects to camera server
////////////////////////////////////////////////////////////////////////////////
TFliteModelExecute::TFliteModelExecute(struct TFliteThreadData* init_data)
{
    model_thread_data.frame_skip  = init_data->frame_skip;
    model_thread_data.input_pipe  = init_data->input_pipe;
    model_thread_data.en_debug    = init_data->en_debug;
    model_thread_data.en_timing   = init_data->en_timing;
    model_thread_data.labels_file = init_data->labels_file;
    model_thread_data.model_file  = init_data->model_file;
    model_cam_queue.insert_idx = 0;
    model_thread_data.camera_queue = &model_cam_queue;
    model_thread_data.stop = false;

    // Start the thread that will run the tensorflow lite model on live camera frames.
    pthread_attr_t thread_attributes;
    pthread_attr_init(&thread_attributes);
    pthread_attr_setdetachstate(&thread_attributes, PTHREAD_CREATE_JOINABLE);

    if (!strcmp(model_thread_data.model_file, "/usr/bin/dnn/ssdlite_mobilenet_v2_coco.tflite")){
        pthread_create(&(model_thread_data.thread), &thread_attributes, ThreadMobileNet, &model_thread_data);
    }
    else{
        fprintf(stderr, "------voxl-mpa-tflite: FATAL: Unsupported model provided!!\n");
        exit(-1);
    }
    pipe_client_set_camera_helper_cb(0, _cam_helper_cb, this);
    pipe_client_open(TFLITE_CH,
                    model_thread_data.input_pipe,
                    "voxl-tflite-server",
                    CLIENT_FLAG_EN_CAMERA_HELPER,
                    0);
}

////////////////////////////////////////////////////////////////////////////////
// Destructor
////////////////////////////////////////////////////////////////////////////////
TFliteModelExecute::~TFliteModelExecute()
{
    pipe_client_close_all();
    pipe_server_close_all();
    model_thread_data.stop          = true;
    model_thread_data.thread_ready  = false;
    model_thread_data.cond_var.notify_all();
    pthread_join(model_thread_data.thread, NULL);
}

////////////////////////////////////////////////////////////////////////////////
// Camera helper: inserts frames into cam queue
////////////////////////////////////////////////////////////////////////////////
static void _cam_helper_cb(__attribute__((unused))int ch, camera_image_metadata_t meta, char* frame, void* context)
{
    static TFliteModelExecute *model_context = (TFliteModelExecute*)context;
    static int n_skipped = 0;
    //Skip some frames
    if (n_skipped < model_context->model_thread_data.frame_skip){
		n_skipped++;
		return;
	}
    if (pipe_client_bytes_in_pipe(ch)>0){
		n_skipped++;
		if(model_context->model_thread_data.en_debug){
			fprintf(stderr, "WARNING, skipping frame on channel %d due to frame backup\n", ch);
		}
		return;
	}
    if (((!model_context->model_thread_data.en_debug) && (!model_context->model_thread_data.en_timing))){
        if (pipe_server_get_num_clients(TFLITE_CH) == 0){
            return;
        }
    }
    n_skipped = 0;
    // Now, pump the frame into our queue
    if (model_context->model_thread_data.thread_ready){
        if (meta.size_bytes > MAX_IMAGE_SIZE){
            fprintf(stderr, "Model cannot process an image with %d bytes\n", meta.size_bytes);
            return;
        }
        int queue_index = model_context->model_cam_queue.insert_idx;

        TFLiteMessage* camera_message = &model_context->model_cam_queue.queue[queue_index];

        camera_message->metadata     = meta;
        memcpy(camera_message->image_pixels, (uint8_t*)frame, meta.size_bytes);

        model_context->model_cam_queue.insert_idx = ((queue_index + 1) % QUEUE_SIZE);
        model_context->model_thread_data.cond_var.notify_all();
    }
    return;
}
