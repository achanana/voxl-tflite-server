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
#include <modal_pipe.h>
#include "config_file.h"

volatile int            g_keepRunning = 1;
std::mutex              g_exitCondMutex;
std::condition_variable g_exitCondVar;
char* PydnetModel     = (char*)"/usr/bin/dnn/tflite_pydnet.tflite";
char* MobileNetModel  = (char*)"/usr/bin/dnn/mobilenet_v1_ssd_coco_labels.tflite";
char* MobileNetLabels = (char*)"/usr/bin/dnn/mobilenet_v1_ssd_coco_labels.txt";
bool en_debug = false;

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
// Print the help message
// -----------------------------------------------------------------------------------------------------------------------------
void _print_usage()
{
    printf("\nCommand line arguments are as follows:\n\n");
    printf("-c, --config    :  load the config file only, for use by the config wizard\n");
    printf("-d, --debug     : Verbose debug output (Default: Off)\n");
    printf("-h              : Print this help message\n");
}

// -----------------------------------------------------------------------------------------------------------------------------
// Parses the command line arguments to the main function
// -----------------------------------------------------------------------------------------------------------------------------
static bool _parse_opts(int argc, char* argv[])
{
    static struct option long_options[] =
    {
        {"config",     required_argument, 0, 'm'},
        {"debug",      required_argument, 0, 'c'},
        {"help",       no_argument,       0, 'h'},
        {0, 0, 0}
    };

    while (1)
	{
		int option_index = 0;
		int c = getopt_long(argc, argv, "cdh", long_options, &option_index);

		// Detect the end of the options.
		if (c == -1)
		{
			break;
		}

		switch (c)
		{
		case 0:
			// for long args without short equivalent that just set a flag nothing left to do so just break.
			if (long_options[option_index].flag != 0) break;
			break;

		case 'c':
			config_file_read();
            exit(0);

		case 'd':
			VOXL_LOG_INFO("Enabling debug mode");
			en_debug = true;
			break;

		case 'h':
			_print_usage();
			return true;

		default:
			// Print the usage if there is an incorrect command line option
			_print_usage();
			return true;
		}
	}

	return false;
}

static void _server_connect_cb(int ch, int client_id, char* name, void* context){
        VOXL_LOG_INFO("\nClient Connected\n");
        // ((TFliteModelExecute *)context)->resume()
}


static void _server_disconnect_cb(int ch, int client_id, char* name, void* context){
        VOXL_LOG_INFO("\nClient Disconnected\n");
    // if(!en_debug && pipe_server_get_num_clients(0) == 0){
    //     ((TFliteModelExecute *)context)->pause();
    // }

}


//------------------------------------------------------------------------------------------------------------------------------
// Main entry point for the application
//------------------------------------------------------------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    TFLiteInitData initData;

    if (_parse_opts(argc, argv)){
		return -1;
	}

    // load and print config file
	if(config_file_read()){
		return -1;
	}
	config_file_print();

    Debug::SetDebugLevel(en_debug ? DebugLevel::ALL : DebugLevel::ERROR);

    initData.pDnnModelFile = model;
    initData.pLabelsFile   = MobileNetLabels;
    initData.pInputPipe    = input_pipe;
    initData.frame_skip    = skip_n_frames;

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
        return -1;
    }
    return 0;
}