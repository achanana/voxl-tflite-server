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

#ifndef TFLITE_GPU_OBJECT_DETECT
#define TFLITE_GPU_OBJECT_DETECT

#include "common_defs.h"
#include "voxl_tflite_interface.h"

//------------------------------------------------------------------------------------------------------------------------------
// Initialization data
//------------------------------------------------------------------------------------------------------------------------------
struct TFLiteInitData
{
    char*      pDnnModelFile;   ///< Dnn model
    char*      pLabelsFile;     ///< Dnn labels file
    int        camera;          ///< Camera we want to use - 0 for hires, 1 for tracking
    bool       verbose;         ///< Verbose debug output - default off
};

//------------------------------------------------------------------------------------------------------------------------------
// Tensorflow Lite Gpu Object Detect class
//------------------------------------------------------------------------------------------------------------------------------
class TFliteModelExecute
{
public:
    TFliteModelExecute(TFLiteInitData* pInitData);
    ~TFliteModelExecute();

    void pause();
    void resume();

    // Callback called by the input interface when it receives the image data
    void PipeImageData(camera_image_metadata_t meta, uint8_t* pImagePixels);

private:

    TFliteThreadData     m_tfliteThreadData;    ///< Tflite thread data
    TFLiteMsgQueue       m_tfliteMsgQueue;      ///< Message queue to the TFLite thread
};

#endif // end #define TFLITE_GPU_OBJECT_DETECT
