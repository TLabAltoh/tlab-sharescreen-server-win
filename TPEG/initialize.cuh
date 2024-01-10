#pragma

#include "cuda_common.h"

__global__ void SetBlockIdx(char* encFrameBuffer) {

	const short bIdx = blockIdx.y * gridDim.x + blockIdx.x;

	char* hedder = encFrameBuffer + (size_t)bIdx * (BLOCK_HEDDER_SIZE + ENC_BUFFER_BLOCK_SIZE * DST_COLOR_SIZE);

	hedder[BLOCK_IDX_UPPER_IDX] = (char)((unsigned short)bIdx >> 8);
	hedder[BLOCK_IDX_LOWER_IDX] = (char)(bIdx);

	return;
}