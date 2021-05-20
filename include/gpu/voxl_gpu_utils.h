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
#ifndef VOXL_GPU_UTILS
#define VOXL_GPU_UTILS

#include "common_defs.h"
#include "debug_log.h"

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

// -----------------------------------------------------------------------------------------------------------------------------
// Rotation Api
// -----------------------------------------------------------------------------------------------------------------------------
enum ImageToTensorApi
{
    GPU_OPENCL,
};

// -----------------------------------------------------------------------------------------------------------------------------
// Initialization structure
// -----------------------------------------------------------------------------------------------------------------------------
struct ImageToTensorCreateData
{
    ImageToTensorApi api;           ///< Api to use
    unsigned int     srcFormat;     ///< Source format
    unsigned int     srcBpp;        ///< Bpp
    unsigned int     srcWidth;      ///< Source buffer width
    unsigned int     srcHeight;     ///< Source buffer height
    unsigned int     srcStride;     ///< Source buffer stride
};

// -----------------------------------------------------------------------------------------------------------------------------
// Execute structure
// -----------------------------------------------------------------------------------------------------------------------------
struct ImageToTensorExecData
{
    unsigned int* pSrcBuffer;       ///< Source buffer
    float*        pDstHBuffer;      ///< Dest H buffer
    float*        pDstWBuffer;      ///< Dest W buffer
    float*        pDstCBuffer;      ///< Dest C buffer
};

// -----------------------------------------------------------------------------------------------------------------------------
// Class that performs image to tensor conversion
// @todo Optimize for performance, supports only RGB to HWC
// -----------------------------------------------------------------------------------------------------------------------------
class ImageToTensor
{
public:
    // Create an instance of the class
    static ImageToTensor* Create(ImageToTensorCreateData* pCreateData);
    // Main function that implements the funcionality
    virtual Status ConvertToTensor(ImageToTensorExecData* pExecData) = 0;

protected:
    ImageToTensor() { }
    virtual ~ImageToTensor() { }

    unsigned int m_srcFormat;   ///< Source format
    unsigned int m_srcBpp;      ///< Bpp
    unsigned int m_srcWidth;    ///< Source buffer width
    unsigned int m_srcHeight;   ///< Source buffer height
    unsigned int m_srcStride;   ///< Source buffer stride
    unsigned int m_rowBytes;    ///< Number of bytes per row
    float        m_mean;        ///< Mean value
    float        m_stddev;      ///< Stddev value

private:
    // One time initialization
    virtual Status Initialize(ImageToTensorCreateData* pCreateData) = 0;
};

// -----------------------------------------------------------------------------------------------------------------------------
// Gpu class that performs image to tensor conversion
// -----------------------------------------------------------------------------------------------------------------------------
class GpuImageToTensor : public ImageToTensor
{
public:
    GpuImageToTensor() { }
    ~GpuImageToTensor() { }
    // Main function that implements the functionality
    Status ConvertToTensor(ImageToTensorExecData* pExecData);

private:
    // One time initialization
    Status Initialize(ImageToTensorCreateData* pInitData);

    static const char ImageToTensorCLKernel[];      ///< Kernel code

    std::vector<cl::Platform> m_platform;           ///< Platform
    std::vector<cl::Device>   m_devices;            ///< Devices
    cl::Context*              m_pContext;           ///< CL Context
    cl::CommandQueue*         m_pQueue;             ///< Queue
    cl::Program*              m_pProgram;           ///< Program
    cl::Kernel*               m_pKernel;            ///< CL Kernel
    cl::Buffer*               m_pBufferSrcRGB;      ///< Src RGB image
    cl::Buffer*               m_pBufferDstH;        ///< Dst H data
    cl::Buffer*               m_pBufferDstW;        ///< Dst W data
    cl::Buffer*               m_pBufferDstC;        ///< Dst C data
};

#endif // #ifndef VOXL_GPU_UTILS