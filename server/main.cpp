/*******************************************************************************
 * Copyright 2021 ModalAI Inc.
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


#include <getopt.h>
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "threads.h"
#include "config_file.h"


char* coco_labels  = (char*)"/usr/bin/dnn/coco_labels.txt";
bool en_debug = false;
bool en_timing = false;


void _print_usage()
{
    printf("\nCommand line arguments are as follows:\n\n");
    printf("-c, --config    : load the config file only, for use by the config wizard\n");
    printf("-d, --debug     : enable verbose debug output (Default: Off)\n");
    printf("-t, --timing    : enable timing output for model operations (Default: Off)\n");
    printf("-h              : Print this help message\n");
}

static bool _parse_opts(int argc, char* argv[])
{
    static struct option long_options[] =
    {
        {"config",     no_argument, 0, 'm'},
        {"debug",      no_argument, 0, 'c'},
        {"timing",     no_argument, 0, 't'},
        {"help",       no_argument,       0, 'h'},
        {0, 0, 0}
    };

    while (1){
		int option_index = 0;
		int c = getopt_long(argc, argv, "cdth", long_options, &option_index);

		if (c == -1) break; // Detect the end of the options.

		switch (c){
		case 0:
			// for long args without short equivalent that just set a flag
            // nothing left to do so just break.
			if (long_options[option_index].flag != 0) break;
			break;

		case 'c':
			config_file_read();
            exit(0);

		case 'd':
			printf("Enabling debug mode\n");
			en_debug = true;
			break;

        case 't':
			printf("Enabling timing mode\n");
			en_timing = true;
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

int main(int argc, char *argv[])
{
    if (_parse_opts(argc, argv)){
		return -1;
	}

    ////////////////////////////////////////////////////////////////////////////////
	// gracefully handle an existing instance of the process and associated PID file
	////////////////////////////////////////////////////////////////////////////////

    // make sure another instance isn't running
	// if return value is -3 then a background process is running with
	// higher privaledges and we couldn't kill it, in which case we should
	// not continue or there may be hardware conflicts. If it returned -4
	// then there was an invalid argument that needs to be fixed.
    if(kill_existing_process(PROCESS_NAME, 2.0)<-2) return -1;

    // start signal handler so we can exit cleanly
	if(enable_signal_handler()==-1){
		fprintf(stderr,"ERROR: failed to start signal handler\n");
		return(-1);
	}

    // make PID file to indicate your project is running
	// due to the check made on the call to rc_kill_existing_process() above
	// we can be fairly confident there is no PID file already and we can
	// make our own safely.
	make_pid_file(PROCESS_NAME);

    ////////////////////////////////////////////////////////////////////////////////
	// load config
	////////////////////////////////////////////////////////////////////////////////
	if(config_file_read()){
		return -1;
	}
	config_file_print();

    ////////////////////////////////////////////////////////////////////////////////
	// initialize tflite data
	////////////////////////////////////////////////////////////////////////////////
    struct TFliteThreadData data;
    data.frame_skip  = skip_n_frames;
    data.input_pipe  = input_pipe;
    data.en_debug    = en_debug;
    data.en_timing   = en_timing;
    data.labels_file = coco_labels;
    data.model_file  = model;

    fprintf(stderr, "\n------VOXL TFLite Server------\n\n");

    main_running = 1;

    TFliteModelExecute* model_thread = new TFliteModelExecute(&data);

    if (model_thread != NULL)
    {
    	while(main_running){
	    	usleep(5000000);
	    }

        fprintf(stderr, "------ Stopping the application\n");

        delete model_thread;

        fprintf(stderr, "\n------ Done: application exited gracefully\n\n");
    }
    else
    {
        return -1;
    }
    return 0;
}
