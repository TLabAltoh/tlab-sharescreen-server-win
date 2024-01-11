#pragma once

#include "tpeg_common.h"
#include "tpeg_cuda.h"

__global__ void SetBlockIdx(char* encoded_frame_buffer) {

	const short block_dispatch_idx = blockIdx.y * gridDim.x + blockIdx.x;

	char* encoded_frame_buffer_ptr = encoded_frame_buffer + (size_t)block_dispatch_idx * (BLOCK_HEDDER_SIZE + (BLOCK_SIZE * ENDIAN_SIZE) * DST_COLOR_SIZE);

	encoded_frame_buffer_ptr[BLOCK_INDEX_BE] = (char)((unsigned short)block_dispatch_idx >> 8);
	encoded_frame_buffer_ptr[BLOCK_INDEX_LE] = (char)(block_dispatch_idx);

	return;
}