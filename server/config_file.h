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


#ifndef CONFIG_FILE_H
#define CONFIG_FILE_H

#include <modal_json.h>

#define buf_len 64
// remove labels,
#define CONFIG_FILE "/etc/modalai/voxl-tflite-server.conf"

#define CONFIG_FILE_HEADER "\
/**\n\
 * This file contains configuration that's specific to voxl-tflite-server.\n\
 *\n\
 * skip_n_frames - how many frames to skip between processed frames. For 30hz\n\
 *                   input frame rate, we recommend skipping 5 frame resulting\n\
 *                   in 5hz model output.\n\
 * model         - which model to use. Currently only support mobilenet for\n\
 *                    object detection.\n\
 * input_pipe    - which camera to use (tracking or hires).\n\
 */\n"


static int skip_n_frames;
static char model[buf_len];
static char input_pipe[buf_len];



static inline void config_file_print(void)
{
	printf("=================================================================\n");
	printf("skip_n_frames:                    %d\n",    skip_n_frames);
	printf("=================================================================\n");
    printf("=================================================================\n");
	printf("model:                            %s\n",    model);
	printf("=================================================================\n");
    printf("=================================================================\n");
	printf("input_pipe:                       %s\n",    input_pipe);
	printf("=================================================================\n");
	return;
}


static inline int config_file_read(void)
{
	int ret = json_make_empty_file_with_header_if_missing(CONFIG_FILE, CONFIG_FILE_HEADER);
	if(ret < 0) return -1;
	else if(ret>0) fprintf(stderr, "Creating new config file: %s\n", CONFIG_FILE);

	cJSON* parent = json_read_file(CONFIG_FILE);
	if(parent==NULL) return -1;

	// actually parse values
	json_fetch_int_with_default(parent, "skip_n_frames", &skip_n_frames, 5);
    json_fetch_string_with_default(parent, "model", model, buf_len, "/usr/bin/dnn/ssdlite_mobilenet_v2_coco.tflite");
	json_fetch_string_with_default(parent, "input_pipe", input_pipe, buf_len, "/run/mpa/hires_preview/");

	if(json_get_parse_error_flag()){
		fprintf(stderr, "failed to parse config file %s\n", CONFIG_FILE);
		cJSON_Delete(parent);
		return -1;
	}


	// write modified data to disk if neccessary
	if(json_get_modified_flag()){
		printf("The config file was modified during parsing, saving the changes to disk\n");
		json_write_to_file_with_header(CONFIG_FILE, parent, CONFIG_FILE_HEADER);
	}
	cJSON_Delete(parent);
	return 0;
}






#endif // end CONFIG_FILE_H