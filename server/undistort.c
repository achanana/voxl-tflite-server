#include <stdio.h>
#include <string.h>
#include <math.h>
#include "undistort.h"


// AVG 1.4ms min 1.20 ms for vga image on VOXL1 fastest core
int mcv_resize_image(const uint8_t* input, uint8_t* output, undistort_map_t* map)
{
	// shortcut variables to make code cleaner
	int  height = map->h_out;
	int   width = map->w_out;
	int n_pix = width*height;
	bilinear_lookup_t* L = map->L;

	// go through every pixel in output image
	for(int pix=0; pix<n_pix; pix++){

		// check for invalid (blank) pixels
		if(L[pix].I[0]<0){
			output[pix] = 0;
            printf("INVALID PIXEL\n");
			continue;
		}

		// get indices from index lookup I
		uint16_t x1 = L[pix].I[0];
		uint16_t y1 = L[pix].I[1];

		// don't worry about all the index algebra, the compiler optimizes this
		uint16_t p0 = input[map->w_in*y1 + x1];
		uint16_t p1 = input[map->w_in*y1 + x1 + 1];
		uint16_t p2 = input[map->w_in*(y1+1) + x1];
		uint16_t p3 = input[map->w_in*(y1+1) + x1 + 1];

		// multiply add each pixel with weighting
		output[pix] = (	p0*L[pix].F[0] +
						p1*L[pix].F[1] +
						p2*L[pix].F[2] +
						p3*L[pix].F[3]) /256;
	}
	return 0;
}

int mcv_init_resize_map(int w_in, int h_in, int w_out, int h_out, undistort_map_t* map)
{
	map->h_out = h_out;
	map->w_out = w_out;
    map->h_in = h_in;
	map->w_in = w_in;

	// allocate new map
	// TODO some sanity and error checking here
	map->L = (bilinear_lookup_t*)malloc(w_out*h_out*sizeof(bilinear_lookup_t));
	if(map->L==NULL){
		perror("failed to allocate memory for lookup table");
		return -1;
	}
	bilinear_lookup_t* L = map->L;

    float x_r = ((float)(w_in - 1)/(float)(w_out - 1));
    float y_r = ((float)(h_in - 1)/(float)(h_out - 1));

	for(int v=0; v<h_out; ++v){
		for(int u=0; u<w_out; ++u){
            int x_l = floor(x_r * u);
            int y_l = floor(y_r * v);
            // x and y difference for top left point
            float x_w = (x_r * u) - x_l;
            float y_w = (y_r * v) - y_l;

			int pix = w_out*v + u;

			// populate lookup table with top left corner pixel
			L[pix].I[0] = x_l;
			L[pix].I[1] = y_l;

			// integer weightings for 4 pixels. Due to truncation, these 4 ints
			// should sum to no more than 255
			L[pix].F[0] = (1-x_w)*(1-y_w)*256;
			L[pix].F[1] = (x_w)*(1-y_w)*256;
			L[pix].F[2] = (y_w)*(1-x_w)*256;
			L[pix].F[3] = (x_w)*(y_w)*256;
		}
	}
	return 0;
}