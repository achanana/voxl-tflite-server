// /*******************************************************************************
//  * Copyright 2020 ModalAI Inc.
//  *
//  * Redistribution and use in source and binary forms, with or without
//  * modification, are permitted provided that the following conditions are met:
//  *
//  * 1. Redistributions of source code must retain the above copyright notice,
//  *    this list of conditions and the following disclaimer.
//  *
//  * 2. Redistributions in binary form must reproduce the above copyright notice,
//  *    this list of conditions and the following disclaimer in the documentation
//  *    and/or other materials provided with the distribution.
//  *
//  * 3. Neither the name of the copyright holder nor the names of its contributors
//  *    may be used to endorse or promote products derived from this software
//  *    without specific prior written permission.
//  *
//  * 4. The Software is used solely in conjunction with devices provided by
//  *    ModalAI Inc.
//  *
//  * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//  * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//  * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
//  * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
//  * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
//  * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
//  * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
//  * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
//  * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
//  * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  * POSSIBILITY OF SUCH DAMAGE.
//  ******************************************************************************/


#include <modal_pipe.h>
#include <getopt.h>     // NOLINT(build/include_order)
#include <sys/time.h>   // NOLINT(build/include_order)
#include <sys/types.h>  // NOLINT(build/include_order)
#include <sys/resource.h>
#include <string.h>
#include <sys/syscall.h>
#include <string.h>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#include <list>
#include <unistd.h>

#include "debug_log.h"
#include "voxl_tflite_gpu_object_detect.h"
#include "voxl_tflite_interface.h"

// Tensorflow thread
extern void* ThreadMobileNet(void* data);
extern void* ThreadTflitePydnet(void* data); //from pData to data MJT
extern void* ThreadSendImageData(void* data);

extern bool en_debug;
char * pipeName;
int frame_skip;

static void _cam_helper_cb(__attribute__((unused))int ch,
                                                  camera_image_metadata_t meta,
                                                  char* frame,
                                                  void* context);

//------------------------------------------------------------------------------------------------------------------------------
// Initialize the object instance.
//------------------------------------------------------------------------------------------------------------------------------
TFliteModelExecute::TFliteModelExecute(TFLiteInitData* pInitData)
{
    Status             status       = S_OK;

    m_tfliteThreadData.pDnnModelFile = pInitData->pDnnModelFile;
    m_tfliteThreadData.pLabelsFile   = pInitData->pLabelsFile;
    m_tfliteThreadData.stop          = false;
    m_tfliteThreadData.pMsgQueue     = &m_tfliteMsgQueue;
    m_tfliteMsgQueue.queueInsertIdx  = 0;
    frame_skip                       = pInitData->frame_skip;

    // Start the thread that will run the tensorflow lite model to detect objects in the camera frames. This thread wont
    // stop issuing requests to the camera module until we terminate the program with Ctrl+C
    pthread_attr_t tfliteAttr;
    pthread_attr_init(&tfliteAttr);
    pthread_attr_setdetachstate(&tfliteAttr, PTHREAD_CREATE_JOINABLE);

    if (!strcmp(pInitData->pDnnModelFile, "/usr/bin/dnn/tflite_pydnet.tflite"))
    {
        pthread_create(&(m_tfliteThreadData.thread), &tfliteAttr, ThreadTflitePydnet, &m_tfliteThreadData);
    }
    else if (!strcmp(pInitData->pDnnModelFile, "/usr/bin/dnn/mobilenet_v1_ssd_coco_labels.tflite"))
    {
        pthread_create(&(m_tfliteThreadData.thread), &tfliteAttr, ThreadMobileNet, &m_tfliteThreadData);
    }
    else
    {
        VOXL_LOG_FATAL("------voxl-mpa-tflite: FATAL: Unsupported model provided!!\n");
        status = S_ERROR;
    }
    // makes it here...
    if (status == S_OK)
    {
        VOXL_LOG_FATAL("------waiting for camera server------\n");
        pipe_client_set_camera_helper_cb(0, _cam_helper_cb, this);
        pipe_client_open(0,
                     pInitData->pInputPipe,
                     "voxl-tflite-server",
                     CLIENT_FLAG_EN_CAMERA_HELPER,
                     0);
        pipeName = pInitData->pInputPipe;
    }
}

//------------------------------------------------------------------------------------------------------------------------------
// Destructor
//------------------------------------------------------------------------------------------------------------------------------
TFliteModelExecute::~TFliteModelExecute()
{


    pipe_client_close_all();

    m_tfliteThreadData.stop        = true;
    m_tfliteThreadData.tfliteReady = false;

    m_tfliteThreadData.condVar.notify_all();

    pthread_join(m_tfliteThreadData.thread, NULL);

}

//------------------------------------------------------------------------------------------------------------------------------
// Process the image data
//------------------------------------------------------------------------------------------------------------------------------
void TFliteModelExecute::PipeImageData(camera_image_metadata_t meta, uint8_t* pImagePixels)
{
    if (m_tfliteThreadData.tfliteReady == true)
    {

        if(meta.size_bytes > MAX_IMAGE_SIZE){
            VOXL_LOG_ERROR("Model Received too many bytes: %d\n", meta.size_bytes);
            return;
        }
        int queueInsertIdx = m_tfliteMsgQueue.queueInsertIdx;

        TFLiteMessage* pTFLiteMessage = &m_tfliteMsgQueue.queue[queueInsertIdx];

        pTFLiteMessage->metadata     = meta;
        memcpy(pTFLiteMessage->imagePixels, pImagePixels, meta.size_bytes);

        m_tfliteMsgQueue.queueInsertIdx = ((queueInsertIdx + 1) % MAX_MESSAGES);
        m_tfliteThreadData.condVar.notify_all();

    }
}

void TFliteModelExecute::pause(){

    pipe_client_close_all();

}

void TFliteModelExecute::resume(){

    pipe_client_set_camera_helper_cb(0, _cam_helper_cb, this);
    pipe_client_open(0,
                     (char*) pipeName,
                     "voxl-tflite-server",
                     CLIENT_FLAG_EN_CAMERA_HELPER,
                     0);

}

// camera helper callback whenever a frame arrives
static void _cam_helper_cb(__attribute__((unused))int ch,
                                                  camera_image_metadata_t meta,
                                                  char* frame,
                                                  void* context)
{
    static int n_skipped = 0;
    //Skip some frames
    if(n_skipped < frame_skip){
		if(en_debug){
			printf("\nskipping required frame\n");
		}
		n_skipped++;
		return;
	}
    if(pipe_client_bytes_in_pipe(ch)>0){
		n_skipped++;
		if(en_debug){
			fprintf(stderr, "WARNING, skipping frame on channel %d due to frame backup\n", ch);
		}
		return;
	}

    n_skipped = 0;
    // add check bytes in pipe
    ((TFliteModelExecute*) context)->PipeImageData(meta, (uint8_t*) frame);
    return;
}
