#pragma once

#include "tpeg_common.h"
#include "tpeg_cuda.h"

__shared__ unsigned short FrameRowDiffSumBuffer[BLOCK_AXIS_SIZE];

__global__ void GetDiffSum(
	unsigned char* current_frame_buffer,
	unsigned char* prev_frame_buffer,
	unsigned short* block_diff_sum_buffer
) {
	unsigned short sum, current_y, prev_y;
	short diff_y;

	const int block_dispatch_idx = blockIdx.y * gridDim.x + blockIdx.x;

	unsigned int frame_pixel_offset =
		(blockIdx.y * gridDim.x << BLOCK_SIZE_LOG2) +
		(threadIdx.x * gridDim.x << BLOCK_AXIS_SIZE_LOG2) + (blockIdx.x << BLOCK_AXIS_SIZE_LOG2);

	unsigned char* prev_frame_buffer_ptr = prev_frame_buffer + frame_pixel_offset * DST_COLOR_SIZE;

	unsigned char* current_frame_buffer_ptr = current_frame_buffer + frame_pixel_offset * SRC_COLOR_SIZE;

	sum = 0;

#pragma unroll

	for (int i = 0; i < BLOCK_AXIS_SIZE; i++) {
		diff_y =
			(short)Conv2Y(
				current_frame_buffer_ptr[R],
				current_frame_buffer_ptr[G],
				current_frame_buffer_ptr[B])
			- (short)Conv2Y(
				prev_frame_buffer_ptr[R],
				prev_frame_buffer_ptr[G],
				prev_frame_buffer_ptr[B]);

		sum += (unsigned short)(diff_y * (1 - (diff_y < 0) * 2));

		prev_frame_buffer_ptr[R] = current_frame_buffer_ptr[R];
		prev_frame_buffer_ptr[G] = current_frame_buffer_ptr[G];
		prev_frame_buffer_ptr[B] = current_frame_buffer_ptr[B];

		prev_frame_buffer_ptr += DST_COLOR_SIZE;
		current_frame_buffer_ptr += SRC_COLOR_SIZE;
	}

	// set block's row sum to buffer.
	FrameRowDiffSumBuffer[threadIdx.x] = sum;

	__syncthreads();

	if (threadIdx.x != 0) return;

	sum = 0;

#pragma unroll

	for (int i = 0; i < BLOCK_AXIS_SIZE; i++)
		sum += FrameRowDiffSumBuffer[i];

	block_diff_sum_buffer[block_dispatch_idx] += sum;
}