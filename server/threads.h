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


#ifndef VOXL_TFLITE_THREADS_HEADER
#define VOXL_TFLITE_THREADS_HEADER

#include <mutex>
#include <condition_variable>
#include <modal_pipe.h>

#define PROCESS_NAME    "voxl-tflite-server"
#define MAX_IMAGE_SIZE  12441600  // 4k YUV image size
#define QUEUE_SIZE      256       // max messages to be stored in queue
#define TFLITE_CH       0         // output MPA channel

////////////////////////////////////////////////////////////////////////////////
// TFlite message data
////////////////////////////////////////////////////////////////////////////////
struct TFLiteMessage
{
    camera_image_metadata_t  metadata;                     // image metadata information
    uint8_t                  image_pixels[MAX_IMAGE_SIZE];  // image pixels
};

////////////////////////////////////////////////////////////////////////////////
// TFlite message queue
////////////////////////////////////////////////////////////////////////////////
struct TFLiteCamQueue
{
    TFLiteMessage queue[QUEUE_SIZE];    // camera frame queue
    int           insert_idx;       // next element insert location (between 0 - QUEUE_SIZE)
};

////////////////////////////////////////////////////////////////////////////////
// TFlite thread data
////////////////////////////////////////////////////////////////////////////////
struct TFliteThreadData
{
    TFLiteCamQueue*         camera_queue;   // points to camera message queue
    char*                   model_file;     // model filename
    char*                   labels_file;    // labels filename
    char*                   input_pipe;     // mpa pipe name
    int                     frame_skip;     // number of frames to skip
    bool                    en_debug;       // flag for debug messages
    bool                    en_timing;      // flag for timing messages
    volatile bool           stop;           // flag for the thread to terminate
    volatile bool           thread_ready;   // flag for showing model thread is ready to start processing frames
    pthread_t               thread;         // model thread handle
    pthread_t               image_thread;   // image thread handle
    std::mutex              cond_mutex;     // mutex
    std::condition_variable cond_var;       // condition variable
};

////////////////////////////////////////////////////////////////////////////////
// TFlite execution class
////////////////////////////////////////////////////////////////////////////////
class TFliteModelExecute
{
public:
    TFliteModelExecute(struct TFliteThreadData* init_data);
    ~TFliteModelExecute();

    TFliteThreadData     model_thread_data;    // model specific thread data
    TFLiteCamQueue       model_cam_queue;      // camera message queue for the thread
};
#endif