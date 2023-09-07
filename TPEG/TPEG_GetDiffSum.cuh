#pragma once

#include "TPEG_Common.h"

__shared__ unsigned short FrameRowDiffSumBuffer[BLOCK_AXIS_SIZE];

__global__ void GetDiffSum(
	unsigned char* currentFrameBuffer,
	unsigned char* prevFrameBuffer,
	unsigned short* blockDiffSumBuffer
) {
	// tmporaly sum buffer.
	unsigned short sum;

	//////////////////////////////////////////////////////////////////////////
	// Calc frame's row tmp sum.
	//

	// Get block index.
	const int bIdx = blockIdx.y * gridDim.x + blockIdx.x;

	// Get a pointer to each buffer.
	unsigned char* prevFrameBufferPt =
		prevFrameBuffer +
		(
			((size_t)bIdx << BLOCK_SIZE_LOG2) +
			((size_t)threadIdx.x << BLOCK_AXIS_SIZE_LOG2)
		) * DST_COLOR_SIZE;

	const unsigned char* currentFrameBufferPt =
		currentFrameBuffer +
		(
			((size_t)blockIdx.y * gridDim.x << BLOCK_SIZE_LOG2) +
			((size_t)threadIdx.x * gridDim.x << BLOCK_AXIS_SIZE_LOG2) + 
			((size_t)blockIdx.x << BLOCK_AXIS_SIZE_LOG2)
		) * ORG_COLOR_SIZE;

	sum = 0;
	unsigned short currentY;
	unsigned short prevY;
	short diffY;

#pragma unroll

	for (int i = 0; i < BLOCK_AXIS_SIZE; i++) {
		// Get luminance's diff value.
		diffY =
			(short)RGB2yCbCr_ForwardConvertor::Y(
				currentFrameBufferPt[R_IDX],
				currentFrameBufferPt[G_IDX],
				currentFrameBufferPt[B_IDX]
			) -
			(short)RGB2yCbCr_ForwardConvertor::Y(
				prevFrameBufferPt[R_IDX],
				prevFrameBufferPt[G_IDX],
				prevFrameBufferPt[B_IDX]
			);

		// Count diff sum.
		sum += (unsigned short)(diffY * (1 - (diffY < 0) * 2));

		prevFrameBufferPt[R_IDX] = currentFrameBufferPt[R_IDX];
		prevFrameBufferPt[G_IDX] = currentFrameBufferPt[G_IDX];
		prevFrameBufferPt[B_IDX] = currentFrameBufferPt[B_IDX];

		// Increment the pinter.
		currentFrameBufferPt += ORG_COLOR_SIZE;
		prevFrameBufferPt += DST_COLOR_SIZE;
	}

	// set block's row sum to buffer.
	FrameRowDiffSumBuffer[threadIdx.x] = sum;

	//////////////////////////////////////////////////////////////////////////
	// Calc frame's tmp sum.
	//

	__syncthreads();

	if (threadIdx.x != 0) return;

	sum = 0;

#pragma unroll

	for (int i = 0; i < BLOCK_AXIS_SIZE; i++) sum += FrameRowDiffSumBuffer[i];

	blockDiffSumBuffer[bIdx] += sum;
}