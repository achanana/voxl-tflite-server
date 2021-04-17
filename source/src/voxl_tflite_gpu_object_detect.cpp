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
extern char* PydnetModel;
extern char* MobileNetModel;

// Global variables to this file only
static TFliteModelExecute* g_pTFliteModelExecute = NULL;

// Global function prototypes
void PipeImageDataCb(camera_image_metadata_t* pImageMetadata, uint8_t* pImagePixels);

//------------------------------------------------------------------------------------------------------------------------------
// Callback to get the image data
//------------------------------------------------------------------------------------------------------------------------------
void PipeImageDataCb(camera_image_metadata_t* pImageMetadata, uint8_t* pImagePixels)
{
    g_pTFliteModelExecute->PipeImageData(pImageMetadata, pImagePixels);
}

//------------------------------------------------------------------------------------------------------------------------------
// Perform any necessary clean up actions before the object gets destroyed
//------------------------------------------------------------------------------------------------------------------------------
void TFliteModelExecute::Cleanup()
{
    if (m_pInputPipeInterface != NULL)
    {
        m_pInputPipeInterface->Destroy();
        m_pInputPipeInterface = NULL;
    }
}

//------------------------------------------------------------------------------------------------------------------------------
// Destructor
//------------------------------------------------------------------------------------------------------------------------------
void TFliteModelExecute::Destroy()
{
    m_tfliteThreadData.stop        = true;
    m_tfliteThreadData.tfliteReady = false;

    m_tfliteThreadData.condVar.notify_all();

    pthread_join(m_tfliteThreadData.thread, NULL);

    if (m_tfliteThreadData.pTcpServer != NULL)
    {
        pthread_join(m_tfliteThreadData.threadSendImg, NULL);
    }

    Cleanup();
    delete this;
}

//------------------------------------------------------------------------------------------------------------------------------
// Process the image data
//------------------------------------------------------------------------------------------------------------------------------
void TFliteModelExecute::PipeImageData(camera_image_metadata_t* pImageMetadata, uint8_t* pImagePixels)
{
    if (m_tfliteThreadData.tfliteReady == true)
    {
        // char name[128];

        // sprintf(&name[0], "/data/misc/camera/temp/myapp_frame_preview_%d.nv12", pImageMetadata->frame_id);
        // FILE* fd = fopen(&name[0], "wb");
        // fwrite(pImagePixels, pImageMetadata->size_bytes, 1, fd);
        // fclose(fd);

        int queueInsertIdx = m_tfliteMsgQueue.queueInsertIdx;

        fprintf(stderr, "\n------voxl-mpa-tflite-gpu INFO: Received hires frame-%d: %d %d ... Index: %d",
                pImageMetadata->frame_id, pImageMetadata->width, pImageMetadata->height, queueInsertIdx);

        TFLiteMessage* pTFLiteMessage = &m_tfliteMsgQueue.queue[queueInsertIdx];

        pTFLiteMessage->pImagePixels = pImagePixels;
        pTFLiteMessage->pMetadata    = pImageMetadata;

        m_tfliteMsgQueue.queueInsertIdx = ((queueInsertIdx + 1) % MAX_MESSAGES);
        m_tfliteThreadData.condVar.notify_all();

        // // Mutex is required for msgQueue access from here and from within the thread wherein it will be de-queued
        // pthread_mutex_lock(&m_tfliteThreadData.mutex);
        // // Queue up work for the result thread "TensorflowThread"
        // m_tfliteThreadData.msgQueue.push((void*)pTFLiteMessage);

        // TensorflowMessage* pTemp = (TensorflowMessage*)m_tfliteThreadData.msgQueue.front();
        // fprintf(stderr, "\n------Queue size: %d ... Front frame: %d", m_tfliteThreadData.msgQueue.size(),
        //         pTemp->pMetadata->frame_id);
        // pthread_cond_signal(&m_tfliteThreadData.cond);
        // pthread_mutex_unlock(&m_tfliteThreadData.mutex);
    }
}

//------------------------------------------------------------------------------------------------------------------------------
// Anything to do prior to running
//------------------------------------------------------------------------------------------------------------------------------
void TFliteModelExecute::Run()
{
}

//------------------------------------------------------------------------------------------------------------------------------
// Create an instance of the class, initialize and return the object instance. If there are any problems during initialization
// will be result in the object not being instantiated
//------------------------------------------------------------------------------------------------------------------------------
TFliteModelExecute* TFliteModelExecute::Create(TFLiteInitData* pInitData)
{
    g_pTFliteModelExecute = new TFliteModelExecute;

    if (g_pTFliteModelExecute != NULL)
    {
        if (g_pTFliteModelExecute->Initialize(pInitData) != S_OK)
        {
            VOXL_LOG_FATAL("\n------voxl-mpa-tflite-gpu: Failed to initialize");
            g_pTFliteModelExecute->Destroy();
            g_pTFliteModelExecute = NULL;
        }
    }

    return g_pTFliteModelExecute;
}

//------------------------------------------------------------------------------------------------------------------------------
// Initialize the object instance. Return S_ERROR on any errors.
//------------------------------------------------------------------------------------------------------------------------------
Status TFliteModelExecute::Initialize(TFLiteInitData* pInitData)
{
    Status             status       = S_OK;
    InputInterfaceData inputData    = { 0 };
    bool               isPydnet     = false;

    m_tfliteThreadData.pTcpServer    = pInitData->pTcpServer;
    m_tfliteThreadData.pDnnModelFile = pInitData->pDnnModelFile;
    m_tfliteThreadData.pLabelsFile   = pInitData->pLabelsFile;
    m_tfliteThreadData.stop          = false;
    m_tfliteThreadData.pMsgQueue     = &m_tfliteMsgQueue;
    m_tfliteMsgQueue.queueInsertIdx  = 0;
    m_pInputPipeInterface            = NULL;

    // Start the thread that will run the tensorflow lite model to detect objects in the camera frames. This thread wont
    // stop issuing requests to the camera module until we terminate the program with Ctrl+C
    pthread_attr_t tfliteAttr;
    pthread_attr_init(&tfliteAttr);
    pthread_attr_setdetachstate(&tfliteAttr, PTHREAD_CREATE_JOINABLE);

    if (!strcmp(pInitData->pDnnModelFile, PydnetModel))
    {
        pthread_create(&(m_tfliteThreadData.thread), &tfliteAttr, ThreadTflitePydnet, &m_tfliteThreadData);
        //isPydnet = false; changed for live frame access with pydnet
    }
    else if (!strcmp(pInitData->pDnnModelFile, MobileNetModel))
    {
        pthread_create(&(m_tfliteThreadData.thread), &tfliteAttr, ThreadMobileNet, &m_tfliteThreadData);
    }
    else
    {
        VOXL_LOG_FATAL("\n------voxl-mpa-tflite: FATAL: Unsupported model provided!!");
        status = S_ERROR;
    }

    if (status == S_OK)
    {
        if (isPydnet == false)
        {
            if (pInitData->pTcpServer != NULL)
            {
                pthread_create(&(m_tfliteThreadData.threadSendImg), &tfliteAttr, ThreadSendImageData, &m_tfliteThreadData);
            }

            pthread_attr_destroy(&tfliteAttr);

            usleep(2000000);

            // TensorflowMessage* pTensorflowMessage = &m_tfliteMsgQueue[pImageMetadata->frame_id % MAX_MESSAGES];

            // pTensorflowMessage->pImagePixels = pImagePixels;
            // pTensorflowMessage->pMetadata    = pImageMetadata;

            // // Mutex is required for msgQueue access from here and from within the thread wherein it will be de-queued
            // pthread_mutex_lock(&m_tfliteThreadData.mutex);
            // // Queue up work for the result thread "TensorflowThread"
            // m_tfliteThreadData.msgQueue.push_back((void*)pTensorflowMessage);
            // fprintf(stderr, "\n------Queue size: %d", m_tfliteThreadData.msgQueue.size());
            // pthread_cond_signal(&m_tfliteThreadData.cond);
            // pthread_mutex_unlock(&m_tfliteThreadData.mutex);

            inputData.ImageReceivedCallback = PipeImageDataCb;
            inputData.pipeName              = "/run/mpa/tracking/"; ///@todo should not be hardcoded

            m_pInputPipeInterface = CameraNamedPipe::Create();

            if (m_pInputPipeInterface != NULL)
            {
                status = m_pInputPipeInterface->Initialize(&inputData);
            }
            else
            {
                VOXL_LOG_FATAL("\n------voxl-mpa-tflite-gpu: FATAL: Cannot open camera pipe!");
                status = S_ERROR;
            }
        }
    }

    return status;
}