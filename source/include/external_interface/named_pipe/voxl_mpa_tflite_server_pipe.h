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

#ifndef VOXL_TFLITE_SERVER_PIPE_H
#define VOXL_TFLITE_SERVER_PIPE_H

#include <stdint.h>
#include <modal_pipe_server.h>
#include "common_defs.h"
#include "external_interface.h"

//------------------------------------------------------------------------------------------------------------------------------
// Class that uses the named pipe interface to send the camera frame data
//------------------------------------------------------------------------------------------------------------------------------
class NamedPipe : public ExternalInterface
{
public:
    // Create a singleton instance of this class
    static ExternalInterface* Create();

private:
    // Disable direct instantiation of this class
    NamedPipe();
    virtual ~NamedPipe() {}

    // Destroy the instance
    virtual void Destroy();
    virtual Status BroadcastFrame(TFliteOutputs outputChannel, char* pFrameData, uint32_t sizeBytes);

    // Do any one time initialization in this function
    virtual Status Initialize(ExternalInterfaceData* pExtIntfData);
    static  void   RequestPipeHandler(int channel, char* pClientName, int bytes, int clientid, void* pContext);
    void           HandleClientRequest(int channel, char* pClientName, int bytes, int clientid);

    // Disable the control pipe
    static const int ControlPipeDisabled = 0;

    static NamedPipe* m_spNamedPipe;                                ///< Single instance pointer of the class
    bool              m_hasChannelStarted[OUTPUT_ID_MAX_TYPES];    ///< Has the channel been started
    char*             m_pChannelPipeNames[OUTPUT_ID_MAX_TYPES];    ///< Channel pipe names indexed with enums
};

#endif // #ifndef VOXL_TFLITE_SERVER_PIPE_H