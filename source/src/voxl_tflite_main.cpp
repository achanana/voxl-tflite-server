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

///<@todo clean up unwanted headers
#include <errno.h>
#include <getopt.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <mutex>
#include <condition_variable>
#include "common_defs.h"
#include "debug_log.h"
#include "tcp_utils.hpp"
#include "voxl_tflite_gpu_object_detect.h"

volatile int            g_keepRunning = 1;
std::mutex              g_exitCondMutex;
std::condition_variable g_exitCondVar;
char* PydnetModel    = (char*)"/usr/bin/dnn/pydnet_model.tflite";
char* MobileNetModel = (char*)"/usr/bin/dnn/mobilenet_v1_ssd_coco_labels.tflite";


// -----------------------------------------------------------------------------------------------------------------------------
// Function prototypes
// -----------------------------------------------------------------------------------------------------------------------------
void CtrlCHandler(int dummy);

// -----------------------------------------------------------------------------------------------------------------------------
// Ctrl+C handler that will stop the camera and exit the program gracefully
// -----------------------------------------------------------------------------------------------------------------------------
void CtrlCHandler(int dummy)
{
    g_keepRunning = 0;
    g_exitCondVar.notify_all();
}

// -----------------------------------------------------------------------------------------------------------------------------
// Check for error in parsing the arguments
// -----------------------------------------------------------------------------------------------------------------------------
int ErrorCheck(int numInputsScanned, const char* pOptionName)
{
    int error = 0;

    if (numInputsScanned != 1)
    {
        fprintf(stderr, "ERROR: Invalid argument for %s option\n", pOptionName);
        error = -1;
    }

    return error;
}

// -----------------------------------------------------------------------------------------------------------------------------
// Parses the command line arguments to the main function
// -----------------------------------------------------------------------------------------------------------------------------
int ParseArgs(int         argc,
              char* const pArgv[],
              int*        pDumpPreviewFrames,
              char*       pIPAddress,
              char*       pDnnModelFile,
              char*       pLabelsFile)
{
    static struct option LongOptions[] =
    {
        {"dumppreview",  required_argument, 0, 'd'},
        {"ipaddress",    required_argument, 0, 'i'},
        {"dnnmodel",     required_argument, 0, 'm'},
        {"labels",       required_argument, 0, 'l'},
        {"help",         no_argument,       0, 'h'},
        {0, 0, 0, 0                               }
    };

    int numInputsScanned = 0;
    int optionIndex      = 0;
    int status           = 0;
    int option;

    while ((status == 0) && (option = getopt_long_only (argc, pArgv, ":d:i:m:l:h", &LongOptions[0], &optionIndex)) != -1)
    {
        switch(option)
        {
            case 'd':
                numInputsScanned = sscanf(optarg, "%d", pDumpPreviewFrames);

                if (ErrorCheck(numInputsScanned, LongOptions[optionIndex].name) != 0)
                {
                    printf("\nNo preview dump frames specified");
                    status = -EINVAL;
                }

                break;

            case 'i':
                numInputsScanned = sscanf(optarg, "%s", pIPAddress);

                if (ErrorCheck(numInputsScanned, LongOptions[optionIndex].name) != 0)
                {
                    printf("\nNo IP address specified");
                    status = -EINVAL;
                }

                break;

            case 'm':
                numInputsScanned = sscanf(optarg, "%s", pDnnModelFile);

                if (ErrorCheck(numInputsScanned, LongOptions[optionIndex].name) != 0)
                {
                    printf("\nNo DNN model filename specified");
                    status = -EINVAL;
                }

                if (!strcmp(pDnnModelFile, "pydnet"))
                {
                    strcpy(pDnnModelFile, PydnetModel);
                    g_keepRunning = 0;
                }
                else if (!strcmp(pDnnModelFile, "mobilenet"))
                {
                    strcpy(pDnnModelFile, MobileNetModel);
                }

                fprintf(stderr, "------Slected model: %s", pDnnModelFile);

                break;

            case 'l':
                numInputsScanned = sscanf(optarg, "%s", pLabelsFile);

                if (ErrorCheck(numInputsScanned, LongOptions[optionIndex].name) != 0)
                {
                    printf("\nNo DNN model labels filename specified");
                    status = -EINVAL;
                }

                break;

            case 'h':
                status = -EINVAL; // This will have the effect of printing the help message and exiting the program
                break;

            // Unknown argument
            case '?':
            default:
                printf("\nInvalid argument passed!");
                status = -EINVAL;
                break;
        }
    }

    return status;
}

// -----------------------------------------------------------------------------------------------------------------------------
// Print the help message
// -----------------------------------------------------------------------------------------------------------------------------
void PrintHelpMessage()
{
    printf("\n\nCommand line arguments are as follows:\n");
    printf("\n-i : IP address of the VOXL to stream the object detected RGB image");
    printf("\n\t : -i 0 to disable streaming");
    printf("\n-m : Deep learning model filename (Default: /bin/dnn/mobilenet_v1_ssd_coco_labels.tflite)");
    printf("\n-l : Class labels filename (Default: /bin/dnn/mobilenet_v1_ssd_coco_labels.txt)");
    printf("\n-d : Dump 'n' preview frames (Default is 0)");
    printf("\n-h : Print this help message");
    printf("\n\nFor example: voxl-mpa-tflite-gpu -i 192.168.1.159\n\n");
}

//------------------------------------------------------------------------------------------------------------------------------
// Main entry point for the application
//------------------------------------------------------------------------------------------------------------------------------
int main(int argc, char **argv)
{
    TcpServer*  pTcpServer         = NULL;
    int         status             = 0;
    char        ipAddress[256]     = "192.168.1.159"; //<@todo Get ip address programatically
    char        dnnModelFile[256]  = "/usr/bin/dnn/mobilenet_v1_ssd_coco_labels.tflite";
    char        dnnLabelsFile[256] = "/usr/bin/dnn/mobilenet_v1_ssd_coco_labels.txt";
    int         numFramesDump      = 0;

    TFLiteInitData initData;

    status = ParseArgs(argc, argv, &numFramesDump, &ipAddress[0], &dnnModelFile[0], &dnnLabelsFile[0]);

    if (status != 0)
    {
        PrintHelpMessage();
    }
    else
    {
        ipAddress[0] = '0'; ///<@todo Remove this

        initData.numFramesDump = numFramesDump;
        initData.pDnnModelFile = &dnnModelFile[0];
        initData.pLabelsFile   = &dnnLabelsFile[0];
        initData.pIPAddress    = &ipAddress[0];

        if (ipAddress[0] == '0')
        {
            initData.pTcpServer = NULL;
        }
        else
        {
            const int portNum = 5556;

            pTcpServer          = new TcpServer;
            initData.pTcpServer = pTcpServer;

            initData.pTcpServer->create_socket(initData.pIPAddress, portNum);
            initData.pTcpServer->bind_socket();
            initData.pTcpServer->connect_client();
        }

        signal(SIGINT, CtrlCHandler);

        VOXL_LOG_INFO("\n------ Hello Tensorflow-Lite-Gpu!\n\n");

        TFLiteObjectDetect* pTFLiteObjectDetect = TFLiteObjectDetect::Create(&initData);

        if (pTFLiteObjectDetect != NULL)
        {
            pTFLiteObjectDetect->Run();

            // The apps keeps running till Ctrl+C is pressed to terminate the program
            while (g_keepRunning)
            {
                std::unique_lock<std::mutex> lock(g_exitCondMutex);
                g_exitCondVar.wait(lock);
            }

            VOXL_LOG_INFO("\n------ Stopping the application");
            pTFLiteObjectDetect->Destroy();

            if (pTcpServer != NULL)
            {
                delete pTcpServer;
            }

            VOXL_LOG_INFO("\n\n------ Done: application exited gracefully\n");
        }
        else
        {
            status = -1;
        }
    }

    return status;
}