#pragma once

#include "tpeg_common.h"
#include "tpeg_cuda.h"

/**
*  big endian:
*	level (1 bit) : position 16
*	run_length (6 bit) : position 5 ~ 0
*	level (1 bit) : position 8
*
*  little endian:
*	level (8 bit) : position 7 ~ 0
*/

#define BIG_ENDIAN_IDX 0	// level's lower 8 bit
#define LITTLE_ENDIAN_IDX 1	// level's upper 2 bit (8 , 16) + run's lower 6 bit (0 ~ 5)

#define RUN_LENGTH_OFFSET 1
#define LEVEL_LITTLE_OFFSET 8

__global__ void EntropyForward(short* dct_result_buffer, char* encoded_frame_buffer, unsigned short* block_diff_sum_buffer) {

	unsigned int block_dispatch_idx = blockIdx.y * gridDim.x + blockIdx.x;

	unsigned int color_channel_idx = threadIdx.x;

	char* encoded_frame_buffer_ptr = encoded_frame_buffer + (size_t)block_dispatch_idx * (BLOCK_HEDDER_SIZE + (BLOCK_SIZE * ENDIAN_SIZE) * DST_COLOR_SIZE);

	// Check this block needs to send as packet.
	if (block_diff_sum_buffer[block_dispatch_idx] == 0) {

		// Reset block diff sum buffer.
		block_diff_sum_buffer[block_dispatch_idx] = 0;

		// Set flag "no need to encode".
		encoded_frame_buffer_ptr[BLOCK_BIT_SIZE_B + color_channel_idx] = (char)NO_NEED_TO_ENCODE;

		return;
	}

	// This pixel color's data grame pointer.
	char* pixel_data_ptr = encoded_frame_buffer_ptr + BLOCK_HEDDER_SIZE + ((size_t)color_channel_idx << (BLOCK_SIZE_LOG2 + ENDIAN_SIZE_LOG2));

	// DCTFrame's data gram index.
	short* dct_result_buffer_ptr = dct_result_buffer + (size_t)block_dispatch_idx * (BLOCK_SIZE * DST_COLOR_SIZE) + ((size_t)color_channel_idx << BLOCK_SIZE_LOG2);

	short level = 0;
	unsigned char sum = 0, run = 0;

#pragma unroll

	// 0 ~ 62 (63 ELEM)
	for (int i = 0; i < BLOCK_SIZE - 1; i++, dct_result_buffer_ptr++) {
		level = *(dct_result_buffer_ptr);
		if (level == 0 || level == (short)(1 << 15)) { // +0, -0

			run++;	// count up run value

			continue;
		}

		// if DCT result buffer's level is validity, set level and run value to buffer

		pixel_data_ptr[BIG_ENDIAN_IDX] = (char)((unsigned short)level);

		/*
		 10進     2の補数表現
		------------------------
		 127  →  01111111
		………
		   3  →  00000011
		   2  →  00000010
		   1  →  00000001
		   0  →  00000000
		  -1  →  11111111
		  -2  →  11111110
		  -3  →  11111101
		………
		-128  →  10000000
		*/

		pixel_data_ptr[LITTLE_ENDIAN_IDX] = (char)(((unsigned short)run << RUN_LENGTH_OFFSET) | ((unsigned short)level >> LEVEL_LITTLE_OFFSET));

		sum++;	// count buffer sum

		run = 0;	// reset run value

		pixel_data_ptr += ENDIAN_SIZE;	// advance the pointer
	}

	// 63 (1ELEM)
	level = *(dct_result_buffer_ptr);
	pixel_data_ptr[BIG_ENDIAN_IDX] = (char)((unsigned short)level);
	pixel_data_ptr[LITTLE_ENDIAN_IDX] = (char)(((unsigned short)run << RUN_LENGTH_OFFSET) | ((unsigned short)level >> LEVEL_LITTLE_OFFSET));

	// Reset save level's count.
	encoded_frame_buffer_ptr[BLOCK_BIT_SIZE_B + (int)color_channel_idx] = (char)++sum;
}

__global__ void EntropyInvert(char* encoded_frame_buffer, short* dct_result_buffer) {

	unsigned int block_dispatch_idx = blockIdx.y * gridDim.x + blockIdx.x;

	unsigned int color_channel_idx = threadIdx.x;

	char* encoded_frame_buffer_ptr = encoded_frame_buffer + (size_t)block_dispatch_idx * (BLOCK_HEDDER_SIZE + (BLOCK_SIZE * ENDIAN_SIZE) * DST_COLOR_SIZE);

	// this pixel color's data grame pointer.
	char* pixel_data_ptr = encoded_frame_buffer_ptr + BLOCK_HEDDER_SIZE + ((size_t)color_channel_idx << (BLOCK_SIZE_LOG2 + ENDIAN_SIZE_LOG2));

	// DiffFrame's data gram index.
	short* dct_result_buffer_ptr = dct_result_buffer + (size_t)block_dispatch_idx * (BLOCK_SIZE * DST_COLOR_SIZE) + ((size_t)color_channel_idx << BLOCK_SIZE_LOG2);

	unsigned char big_endian, little_endian, run, level;

	for (int i = 0; i < (BLOCK_SIZE - 1); i++, pixel_data_ptr += ENDIAN_SIZE) {
		big_endian = (unsigned char)pixel_data_ptr[BIG_ENDIAN_IDX];
		little_endian = (unsigned char)pixel_data_ptr[LITTLE_ENDIAN_IDX];

		/**
		*  extract only bits corresponding to run length
		*	1 1 1 1 1 1 1 1
		*          &
		*	0 1 1 1 1 1 1 0 (= 126)
		*
		*	0 1 1 1 1 1 1 0 >> 1 = 0 0 1 1 1 1 1 1
		*/
		run = (little_endian & (unsigned char)126) >> RUN_LENGTH_OFFSET;
		i += run;

		/**
		*  Combine the parts corresponding to the levels of the little endian and big endian, respectively.
		*	1 1 1 1 1 1 1 1
		*          &
		*	1 0 0 0 0 0 0 1 (= 129)
		*
		*	(1 0 0 0 0 0 0 1 << 8) | 1 1 1 1 1 1 1 1
		*/
		dct_result_buffer_ptr[i] = (short)((unsigned short)(little_endian & (unsigned char)129) << LEVEL_LITTLE_OFFSET) | (short)((unsigned short)big_endian);
	}
}