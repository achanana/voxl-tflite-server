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
#include "voxl_tflite_gpu_object_detect.h"

volatile int            g_keepRunning = 1;
std::mutex              g_exitCondMutex;
std::condition_variable g_exitCondVar;
char* PydnetModel    = (char*)"/usr/bin/dnn/tflite_pydnet.tflite";
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
              char*       pDnnModelFile,
              char*       pLabelsFile,
              int*        pCamera,
              bool*       verbose)
{
    static struct option LongOptions[] =
    {
        {"dnnmodel",     required_argument, 0, 'm'},
        {"camera",       required_argument, 0, 'c'},
        {"labels",       required_argument, 0, 'l'},
        {"help",         no_argument,       0, 'h'},
        {0, 0, 0, 0                               }
    };

    int numInputsScanned = 0;
    int optionIndex      = 0;
    int status           = 0;
    int option;

    while ((status == 0) && (option = getopt_long_only (argc, pArgv, ":c:m:l:hv", &LongOptions[0], &optionIndex)) != -1)
    {
        switch(option)
        {
            case 'v':
                *verbose = true;
                break;
            case 'c':
                numInputsScanned = sscanf(optarg, "%d", pCamera);
                if (!strcmp(optarg, "tracking"))
                {
                    *pCamera = 1;
                }
                else if (!strcmp(optarg, "hires"))
                {
                    *pCamera = 0;
                }
                else
                {
                    printf("Invalid camera option specified: %s\n", optarg);
                    status = -EINVAL;
                }
                break;

            case 'm':
                numInputsScanned = sscanf(optarg, "%s", pDnnModelFile);

                if (ErrorCheck(numInputsScanned, LongOptions[optionIndex].name) != 0)
                {
                    printf("No DNN model filename specified\n");
                    status = -EINVAL;
                }

                if (!strcmp(pDnnModelFile, "pydnet"))
                {
                    strcpy(pDnnModelFile, PydnetModel);
                }
                else if (!strcmp(pDnnModelFile, "mobilenet"))
                {
                    strcpy(pDnnModelFile, MobileNetModel);
                }

                fprintf(stderr, "------Selected model: %s\n", pDnnModelFile);

                break;

            case 'l':
                numInputsScanned = sscanf(optarg, "%s", pLabelsFile);

                if (ErrorCheck(numInputsScanned, LongOptions[optionIndex].name) != 0)
                {
                    printf("No DNN model labels filename specified\n");
                    status = -EINVAL;
                }

                break;

            case 'h':
                status = -EINVAL; // This will have the effect of printing the help message and exiting the program
                break;

            // Unknown argument
            case '?':
            default:
                printf("Invalid argument passed!\n");
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
    printf("\nCommand line arguments are as follows:\n\n");
    printf("-c <camera>     : Camera to use for object detection: hires or tracking. (Default: tracking)\n");
    printf("-v              : Verbose debug output (Default: Off)\n");
    printf("-m <file>       : Deep learning model filename (Default: /bin/dnn/mobilenet_v1_ssd_coco_labels.tflite)\n");
    printf("-l <file>       : Class labels filename (Default: /bin/dnn/mobilenet_v1_ssd_coco_labels.txt)\n");
    printf("-h              : Print this help message\n");
}

static void _server_connect_cb(int ch, int client_id, char* name, void* context){

    if(pipe_server_get_num_clients(0) == 1){
        ((TFliteModelExecute *)context)->resume();
    }

}


static void _server_disconnect_cb(int ch, int client_id, char* name, void* context){

    if(pipe_server_get_num_clients(0) == 0){
        ((TFliteModelExecute *)context)->pause();
    }

}


//------------------------------------------------------------------------------------------------------------------------------
// Main entry point for the application
//------------------------------------------------------------------------------------------------------------------------------
int main(int argc, char **argv)
{

    int         status             = 0;
    char        dnnModelFile[256]  = "/usr/bin/dnn/mobilenet_v1_ssd_coco_labels.tflite";
    char        dnnLabelsFile[256] = "/usr/bin/dnn/mobilenet_v1_ssd_coco_labels.txt";
    int         camera             = 1;
    bool        verbose            = 0;


    TFLiteInitData initData;

    status = ParseArgs(argc, argv, &dnnModelFile[0], &dnnLabelsFile[0], &camera, &verbose);


    Debug::SetDebugLevel(verbose ? DebugLevel::ALL : DebugLevel::ERROR);


    if (status != 0)
    {
        PrintHelpMessage();
    }
    else
    {

        initData.pDnnModelFile = &dnnModelFile[0];
        initData.pLabelsFile   = &dnnLabelsFile[0];
        initData.camera        = camera;
        initData.verbose       = verbose;

        signal(SIGINT, CtrlCHandler);

        VOXL_LOG_FATAL("\n------VOXL TFLite Server------\n\n");

        TFliteModelExecute* pTFliteModelExecute = new TFliteModelExecute(&initData);

        if (pTFliteModelExecute != NULL)
        {


            pipe_server_set_disconnect_cb(0, _server_disconnect_cb, pTFliteModelExecute);
            pipe_server_set_connect_cb(0, _server_connect_cb, pTFliteModelExecute);

            // The apps keeps running till Ctrl+C is pressed to terminate the program
            while (g_keepRunning)
            {
                std::unique_lock<std::mutex> lock(g_exitCondMutex);
                // g_exitCondVar.wait(lock);
                ///<@todo Enable convition variable
                sleep(2);
            }

            VOXL_LOG_FATAL("------ Stopping the application\n");
            delete pTFliteModelExecute;

            VOXL_LOG_FATAL("\n------ Done: application exited gracefully\n\n");
        }
        else
        {
            status = -1;
        }
    }

    return status;
}