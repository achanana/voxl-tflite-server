/*******************************************************************************
 * Copyright 2020 ModalAI Inc.
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
#ifndef VOXL_TFLITE_HEADER
#define VOXL_TFLITE_HEADER

#include <mutex>
#include <condition_variable>
#include "common_defs.h"
#include <modal_pipe.h>

//4k YUV
#define MAX_IMAGE_SIZE (12441600)

class  TcpServer;

// -----------------------------------------------------------------------------------------------------------------------------
// TFlite thread message data
// -----------------------------------------------------------------------------------------------------------------------------
struct TFLiteMessage
{
    camera_image_metadata_t  metadata;                     ///< Image metadata information
    uint8_t                  imagePixels[MAX_IMAGE_SIZE];  ///< Image pixels
};

// -----------------------------------------------------------------------------------------------------------------------------
// Tensorflow message queue
// -----------------------------------------------------------------------------------------------------------------------------
struct TFLiteMsgQueue
{
    TFLiteMessage queue[MAX_MESSAGES];      ///< Image metadata information
    int           queueInsertIdx;           ///< Next element insert location (between 0 - MAX_MESSAGES)
};

// -----------------------------------------------------------------------------------------------------------------------------
// Thread Data for camera request and result thread
// -----------------------------------------------------------------------------------------------------------------------------
typedef struct TFliteThreadData
{
    TFLiteMsgQueue*         pMsgQueue;      ///< Points to circular message queue
    char*                   pDnnModelFile;  ///< Dnn Model filename
    char*                   pLabelsFile;    ///< DNN Model labels filename
    volatile bool           stop;           ///< Indication for the thread to terminate
    volatile bool           tfliteReady;    ///< Indication that the tflite thread is ready to start processing frames
    pthread_t               thread;         ///< Thread handle
    pthread_t               threadSendImg;  ///< Send Image Thread handle
    std::mutex              condMutex;      ///< Mutex
    std::condition_variable condVar;        ///< Condition variable
    int                     camera;         ///< Camera indicator - 0 for hires, 1 for tracking
    bool                    verbose;        ///< Verbose debug output
} TFliteThreadData;

#endif // VOXL_TFLITE_HEADER