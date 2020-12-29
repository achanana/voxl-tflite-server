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

///<@todo Clean up the headers
#include "common_defs.h"
#include "debug_log.h"
#include "voxl_tflite_server.h"
#include "voxl_mpa_tflite_server_pipe.h"

///<@todo Add logic to stop writing if no clients are listening

// Initialize static pointer variable to NULL
NamedPipe* NamedPipe::m_spNamedPipe = NULL;

//------------------------------------------------------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------------------------------------------------------
NamedPipe::NamedPipe()
{
    // Named pipe directories for different channel types
    m_pChannelPipeNames[OUTPUT_ID_RGB_IMAGE] = (char*) TFLITE_CHANNEL_RGB_IMAGE;
    m_hasChannelStarted[OUTPUT_ID_RGB_IMAGE] = false;
}

//------------------------------------------------------------------------------------------------------------------------------
// Create a singleton instance of this class
//------------------------------------------------------------------------------------------------------------------------------
ExternalInterface* NamedPipe::Create()
{
    if (m_spNamedPipe == NULL)
    {
        m_spNamedPipe = new NamedPipe;
    }

    return m_spNamedPipe;
}

//------------------------------------------------------------------------------------------------------------------------------
// Destroy the instance
//------------------------------------------------------------------------------------------------------------------------------
void NamedPipe::Destroy()
{
    if (m_spNamedPipe != NULL)
    {
        pipe_server_close_all_channels();
        delete m_spNamedPipe;
        m_spNamedPipe = NULL;
    }
}

//------------------------------------------------------------------------------------------------------------------------------
// libmodal_pipe calls into this callback when a client subscribes to the channel
//------------------------------------------------------------------------------------------------------------------------------
void NamedPipe::RequestPipeHandler(int channel, char* pClientName, int bytes, int clientid, void* pContext)
{
    if (m_spNamedPipe != NULL)
    {
        // Start the requested channel if it hasn't already been started
        m_spNamedPipe->HandleClientRequest(channel, pClientName, bytes, clientid);
    }
}

//------------------------------------------------------------------------------------------------------------------------------
// Call back into the voxl-tflite-server callback to start the requested camera
//------------------------------------------------------------------------------------------------------------------------------
void NamedPipe::HandleClientRequest(int channel, char* pClientName, int bytes, int clientid)
{
    switch (channel)
    {
        case OUTPUT_ID_RGB_IMAGE:
            if (m_hasChannelStarted[OUTPUT_ID_RGB_IMAGE] == false)
            {
                VOXL_LOG_INFO("\n------voxl-tflite-server INFO: Connected to Rgb image channel %d, client name: %s\n",
                              channel, pClientName);

                m_hasChannelStarted[OUTPUT_ID_RGB_IMAGE] = true;
            }

            break;

        default:
            VOXL_LOG_WARNING("\n------voxl-tflite-server WARNING: Bad channel type: %d", channel);
            break;
    }

	return;
}

//------------------------------------------------------------------------------------------------------------------------------
// Do any one time initialization in this function
//------------------------------------------------------------------------------------------------------------------------------
Status NamedPipe::Initialize(ExternalInterfaceData* pExtIntfData)   ///< Data provided by server core to external interface
{
    Status status = S_OK;

    if (pExtIntfData->outputMask & RgbOutputMask)
    {
        // 0 means success
        if (0 == pipe_server_init_channel(OUTPUT_ID_RGB_IMAGE,
                                          m_pChannelPipeNames[OUTPUT_ID_RGB_IMAGE],
                                          ControlPipeDisabled))
        {
            pipe_server_set_request_cb(OUTPUT_ID_RGB_IMAGE, RequestPipeHandler, NULL);
        }
        else
        {
            status = S_ERROR;
        }
    }

    if (status == S_OK)
    {
        m_interfaceData = *pExtIntfData;
    }

    return status;
}

//------------------------------------------------------------------------------------------------------------------------------
// Broadcast the camera frame to all clients
//------------------------------------------------------------------------------------------------------------------------------
Status NamedPipe::BroadcastFrame(TFliteOutputs outputChannel,   ///< Output type to send the data to
                                 char*         pData,           ///< Data to send
                                 uint32_t      sizeBytes)       ///< Size of data to send
{
    if (m_hasChannelStarted[outputChannel] == true)
    {
        pipe_server_send_to_channel(outputChannel, pData, sizeBytes);
    }

    return S_OK;
}