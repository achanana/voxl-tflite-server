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

///<@todo Clean up the headers
#include <fcntl.h>
#include <modal_pipe.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <thread>
#include <unistd.h>
#include "common_defs.h"
#include "debug_log.h"
#include "input_interface_named_pipe.h"

void ProcessCameraServerData(ThreadData* pThreadData);

CameraNamedPipe* g_pCameraNamedPipe = NULL;

//------------------------------------------------------------------------------------------------------------------------------
// Create an object of type CameraNamedPipe and initialize it
//------------------------------------------------------------------------------------------------------------------------------
CameraNamedPipe* CameraNamedPipe::Create()
{
    return new CameraNamedPipe;
}

//------------------------------------------------------------------------------------------------------------------------------
// Destroy the object
//------------------------------------------------------------------------------------------------------------------------------
void CameraNamedPipe::Cleanup()
{
    VOXL_LOG_INFO("\n------ Shutting down pipe");
    pipe_client_close_all();
    m_threadData.readerThreadStop = 1;
}

//------------------------------------------------------------------------------------------------------------------------------
// Destroy the object
//------------------------------------------------------------------------------------------------------------------------------
void CameraNamedPipe::Destroy()
{
    Cleanup();
    delete this;
}

//------------------------------------------------------------------------------------------------------------------------------
// Initialize the object
//------------------------------------------------------------------------------------------------------------------------------
Status CameraNamedPipe::Initialize(InputInterfaceData* pInputIntfData)
{
    Status status = S_OK;
    char   clientName[MAX_NAME_LENGTH];

    m_threadData.frameFifoFD      = -1;
    m_threadData.readerThreadStop = 0;
    m_pReaderThread               = NULL;

    strcpy(&clientName[0], "voxl-tflite-gpu");

    if (pInputIntfData != NULL)
    {
        memcpy(&m_threadData.interfaceData, pInputIntfData, sizeof(InputInterfaceData));

        int result = pipe_client_init_channel(ChannelId,
                                              (char*) pInputIntfData->pipeName,
                                              &clientName[0],
                                              ReaderThreadDisabled,
                                              (640*480)+1000); // Shouldn't matter because we are going to read from the fd

        if (result == 0)
    	{
            m_threadData.frameFifoFD = pipe_client_get_fd(ChannelId);

            if (m_threadData.frameFifoFD == -1)
            {
                VOXL_LOG_FATAL("\n------ FATAL: Camera pipe interface initialization failed because we cannot get pipe fd!");
                status = S_ERROR;
            }
    	}
        else
        {
            VOXL_LOG_FATAL("\n------ FATAL: pipe_client_init_channel(..) call failed!");
            status = S_ERROR;
        }
    }
    else
    {
        status = S_ERROR;
    }

    if (status == S_OK)
    {
        g_pCameraNamedPipe = this;
        m_threadData.readerThreadStop = 0;

        m_pReaderThread = new std::thread(ProcessCameraServerData, &m_threadData);
    }
    
    return status;
}

//------------------------------------------------------------------------------------------------------------------------------
// Thread that continuously receives data from the camera server pipe
//------------------------------------------------------------------------------------------------------------------------------
void ProcessCameraServerData(ThreadData* pThreadData)
{
    int frameFifoFD = pThreadData->frameFifoFD;

    if (frameFifoFD != -1)
    {
        camera_image_metadata_t imageInfo[MAX_MESSAGES]    = { 0 };
        volatile int            bytes                      = 0;
        pid_t                   tid                        = syscall(SYS_gettid);
        int                     which                      = PRIO_PROCESS;
        int                     nice                       = -5;
        int                     imageSizeBytes             = 0;
        uint8_t*                pImageBuffer[MAX_MESSAGES] = { NULL };
        int                     frameIndex                 = 0;

        setpriority(which, tid, nice);

        while (pThreadData->readerThreadStop == 0)
        {
            bytes = read(frameFifoFD, &imageInfo[frameIndex], sizeof(camera_image_metadata_t));
            imageSizeBytes = imageInfo[frameIndex].size_bytes;

            ///<@todo Is there any way to pull this out
            if (pImageBuffer[0] == NULL)
            {
                for (int i = 0; i < MAX_MESSAGES; i++)
                {
                    pImageBuffer[i] = new uint8_t[imageSizeBytes];

                    if (pImageBuffer[i] == NULL)
                    {
                        break;
                    }
                }
            }

            if (bytes == sizeof(camera_image_metadata_t))
            {
                while (bytes != imageSizeBytes)
                {
                    bytes = read(frameFifoFD, pImageBuffer[frameIndex], imageSizeBytes);
                }

                if ((imageInfo[frameIndex].frame_id % 2) == 0)
                {
                    pThreadData->interfaceData.ImageReceivedCallback(&imageInfo[frameIndex], pImageBuffer[frameIndex]);
                }
                else
                {
                    continue;
                }
            }
            else if (bytes > 0)
            {
                VOXL_LOG_FATAL("\n------ FATAL: Need to handle looking for magic number");
                break;
            }

            frameIndex = ((frameIndex + 1) % MAX_MESSAGES);
        }
    }
}