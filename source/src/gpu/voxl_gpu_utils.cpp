#include <iostream>
#include <vector>
#include <string>
#include <tgmath.h>
#include "voxl_gpu_utils.h"

const char GpuImageToTensor::ImageToTensorCLKernel[] =
    "kernel void image_to_tensor(global unsigned char*  pSrcBuffer,\n"
    "                            global unsigned float* pDestHBuffer,\n"
    "                            global unsigned float* pDestWBuffer,\n"
    "                            global unsigned float* pDestCBuffer,\n"
    "                            float mean,\n"
    "                            float stddev,\n"
    "                            unsigned int bpp,\n"
    "                            unsigned int width,\n"
    "                            unsigned int stride)\n"
    "{\n"
    "    //Work-item gets its index within index space\n"
    "    int threadId            = get_global_id(0);\n"
    "    unsigned int  rowBytes  = (stride*bpp);\n"
    "    unsigned int  copyRGB   = (threadId % 3);\n"
    "    unsigned int  copyRow   = (threadId / 3);\n"
    "    ///<@todo Assuming 4 bytes integer\n"
    "    unsigned char*  pSrcAddr = (pSrcBuffer  + (copyRow*rowBytes)) + copyRGB;\n"
    "    unsigned float* pDstAddr = pDestHBuffer;\n"
    "\n"
    "    if (copyRGB == 0)\n"
    "    {\n"
    "        pDstAddr = pDstHBuffer;\n"
    "\n"
    "    }\n"
    "    else if (copyRGB == 1)\n"
    "    {\n"
    "        pDstAddr = pDstWBuffer;\n"
    "\n"
    "    }\n"
    "    else\n"
    "    {\n"
    "        pDstAddr = pDstCBuffer;\n"
    "\n"
    "    }\n"
    "\n"
    "    pDstAddr = pDstAddr + (copyRow * width * sizeof(float));\n"
    "\n"
    "    ///<@todo Optimize based on buffer access pattern\n"
    "\n"
    "    ///<@todo Assumes unsigned int is 4 bytes\n"
    "    for (int i=0; i < width; i++)\n"
    "    {\n"
    "        pDstAddr[i] = (((float)(pSrcAddr[i*3] * 1.0) - mean) / stdev);\n"
    "    }\n"
    "}\n";

// -----------------------------------------------------------------------------------------------------------------------------
// Create an instance of the requested object
// -----------------------------------------------------------------------------------------------------------------------------
ImageToTensor* ImageToTensor::Create(ImageToTensorCreateData* pCreateData)
{
    ImageToTensor* pImageToTensor = NULL;

    switch (pCreateData->api)
    {
        case GPU_OPENCL:
            pImageToTensor = new GpuImageToTensor;
            pImageToTensor->Initialize(pCreateData);
            break;
        default:
            VOXL_LOG_FATAL("\n------voxl-tflite-server FATAL: Cannot create image to tensor object");
            break;
    }

    return pImageToTensor;
}

// -----------------------------------------------------------------------------------------------------------------------------
// Perform any one time initialization
// -----------------------------------------------------------------------------------------------------------------------------
Status GpuImageToTensor::Initialize(ImageToTensorCreateData* pCreateData)
{
    Status status = S_OK;

    m_srcFormat = pCreateData->srcFormat;
    m_srcBpp    = pCreateData->srcBpp;
    m_srcWidth  = pCreateData->srcWidth;
    m_srcHeight = pCreateData->srcHeight;
    m_srcStride = pCreateData->srcStride;
    m_rowBytes  = (m_srcStride * 3); ///<@todo 3 because of RGB - make it generic by looking at format

    cl::Platform::get(&m_platform);

	if (m_platform.empty())
    {
	    VOXL_LOG_FATAL("\n------voxl-tflite-server FATAL: OpenCL platforms not found");
        status = S_ERROR;
	}

    if (status == S_OK)
    {
        m_platform[0].getDevices(CL_DEVICE_TYPE_GPU, &m_devices);
        m_pContext = new cl::Context(m_devices[0]);

        // std::cout << m_devices[0].getInfo<CL_DEVICE_NAME>() << std::endl;
        // std::cout << m_devices[0].getInfo<CL_DRIVER_VERSION>() << std::endl;

        // Create command queue.
        m_pQueue = new cl::CommandQueue(*m_pContext, m_devices[0]);

        // Compile OpenCL program
        m_pProgram = new cl::Program(*m_pContext,
                                     cl::Program::Sources(1,
                                                          std::make_pair(ImageToTensorCLKernel,
                                                          strlen(ImageToTensorCLKernel))));

        try
        {
            m_pProgram->build(m_devices);
        }
        catch (const cl::Error&)
        {
            std::cerr
            << "OpenCL compilation error" << std::endl
            << m_pProgram->getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_devices[0])
            << std::endl;

            return S_ERROR;
        }

        m_pKernel = new cl::Kernel(*m_pProgram, "image_to_tensor");
    }

    return status;
}

// -----------------------------------------------------------------------------------------------------------------------------
// Function that performs image split (into left and right images)
// -----------------------------------------------------------------------------------------------------------------------------
Status GpuImageToTensor::ConvertToTensor(ImageToTensorExecData* pExecData)
{
    Status status = S_OK;

    unsigned int* pSrcBuffer   = pExecData->pSrcBuffer;
    float*        pDstHBuffer  = pExecData->pDstHBuffer;
    float*        pDstWBuffer  = pExecData->pDstWBuffer;
    float*        pDstCBuffer  = pExecData->pDstCBuffer;
    unsigned int  hwcSizeBytes = (m_srcWidth * m_srcHeight * sizeof(float));

    // Allocate device buffers and transfer input data to device.
    cl::Buffer clBufferSrc(*m_pContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (m_rowBytes * m_srcHeight), pSrcBuffer);
    cl::Buffer clBufferDstH(*m_pContext, CL_MEM_READ_WRITE, hwcSizeBytes);
    cl::Buffer clBufferDstW(*m_pContext, CL_MEM_READ_WRITE, hwcSizeBytes);
    cl::Buffer clBufferDstC(*m_pContext, CL_MEM_READ_WRITE, hwcSizeBytes);

    // Set kernel parameters (dest, src, width, height, sin(theta), cos(theta))
    m_pKernel->setArg(0, clBufferSrc);
    m_pKernel->setArg(1, clBufferDstH);
    m_pKernel->setArg(2, clBufferDstW);
    m_pKernel->setArg(3, clBufferDstC);
    m_pKernel->setArg(4, m_mean); ///<@todo Assumes RGB-888
    m_pKernel->setArg(5, m_stddev);
    m_pKernel->setArg(6, 3); ///<@todo Assumes RGB-888
    m_pKernel->setArg(7, m_srcWidth);
    m_pKernel->setArg(8, m_srcStride);

    // Launch kernel on the compute device
    ///<@todo Do the global, local workgroup size. In each one of the instance of the kernel copy one entire row of R or G or B
    m_pQueue->enqueueNDRangeKernel(*m_pKernel, cl::NullRange, m_srcHeight * 3, cl::NullRange);
    // Get result back to host
    m_pQueue->enqueueReadBuffer(clBufferDstH, CL_TRUE, 0, hwcSizeBytes, pDstHBuffer);
    m_pQueue->enqueueReadBuffer(clBufferDstW, CL_TRUE, 0, hwcSizeBytes, pDstWBuffer);
    m_pQueue->enqueueReadBuffer(clBufferDstC, CL_TRUE, 0, hwcSizeBytes, pDstCBuffer);

    return status;
}