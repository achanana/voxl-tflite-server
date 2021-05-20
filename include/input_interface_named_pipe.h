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

#ifndef INPUT_NAMED_PIPE_H
#define INPUT_NAMED_PIPE_H

#include <stdint.h>
#include <modal_pipe_client.h>
#include <thread>
#include "common_defs.h"

// Forward decl
struct camera_image_metadata_t;

// Function prototypes for callbacks by the input interface
typedef void (*ImageReceivedCb)(camera_image_metadata_t* pImageMetadata, uint8_t* pImagePixels);

// -----------------------------------------------------------------------------------------------------------------------------
// Input interface required data
// -----------------------------------------------------------------------------------------------------------------------------
struct InputInterfaceData
{
    const char*     pipeName;               ///< Pipe name
    ImageReceivedCb ImageReceivedCallback;  ///< Image received callback
};

// -----------------------------------------------------------------------------------------------------------------------------
// Reader thread data
// -----------------------------------------------------------------------------------------------------------------------------
struct ThreadData
{
    InputInterfaceData interfaceData;       ///< Interface data
    int                frameFifoFD;         ///< Frame fifo id
    volatile int       readerThreadStop;    ///< Reader thread stop
};

//------------------------------------------------------------------------------------------------------------------------------
// Class that uses the named pipe interface to send the camera frame data
//------------------------------------------------------------------------------------------------------------------------------
class CameraNamedPipe
{
public:
    // Create a singleton instance of this class
    static CameraNamedPipe* Create();
    // Destroy the instance
    void Destroy();
    // Initialize the instance
    Status Initialize(InputInterfaceData* pInputIntfData);

private:
    // Disable direct instantiation of this class
    CameraNamedPipe()  { }
    ~CameraNamedPipe() { }

    void Cleanup();

    static const int32_t ReaderThreadDisabled = 0;
    static const int32_t ChannelId            = 0;

    std::thread* m_pReaderThread;       ///< Reader thread
    ThreadData   m_threadData;          ///< Thread data
};

#endif // #ifndef INPUT_NAMED_PIPE_H
