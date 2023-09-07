#pragma once

#include "TPEG_Common.h"

// NO USE --------------
// RUN_BIT_SIZE 6
// LEVEL_BIT_SIZE 12
// BYTE_BIT_SIZE 8
// ---------------------

// CONSTITUTION ---------------------
// BIG_ENDIAN: LEVEL (9) (1BIT): RUN (5 ~ 0) (6BIT): LEVEL (8) (1BIT) ... TOTAL 8BIT
// LITTLE_ENDIAN: LEVEL (7 ~ 0) (8BIT) ... TOTAL 8BIT
// -----------------------------------------------------------------------------------

// LEVEL'S LOWER 8 BIT(0 ~ 7)
#define BIG_ENDIAN_DIX 0
// LEVEL'S UPPER 2 BIT (8 ~ 9) + RUN'S LOWER 6 BIT (0 ~ 5)
#define LITTLE_ENDIAN_IDX 1

#define RUN_OFFSET 1
// LEVEL'S UPPER 8 BIT OFFSET(ACTUALY 2 BIT)
#define LEVEL_BIG_OFFSET 0
// LEVEL'S LOWER 8 BIT OFFSET
#define LEVEL_LITTLE_OFFSET 8

#define PIXEL_VALUE 0

#define BLOCK_COLOR_SIZE_IDX_OFFSET 2

#define NO_NEED_TO_ENCODE 0

#define ERROR_CHECK 1

__global__ void EntropyForward(
	short* dctFrameBuffer,
	char* encFrameBuffer,
	unsigned short* blockDiffSumBuffer
) {
	// Block index.
	const short bIdx = blockIdx.y * gridDim.x + blockIdx.x;

	// Color index
	const char cIdx = threadIdx.x;

	// EncFrame's block hedder.
	char* hedder =
		encFrameBuffer +
		(size_t)bIdx *
		(BLOCK_HEDDER_SIZE + ENC_BUFFER_BLOCK_SIZE * DST_COLOR_SIZE);

	// Check this block needs to send as packet.
#if 1
	if (blockDiffSumBuffer[bIdx] == 0) {

		// Reset block diff sum buffer.
		blockDiffSumBuffer[bIdx] = 0;

		// Set flag "no need to encode".
		hedder[BLOCK_COLOR_SIZE_IDX_OFFSET + (int)cIdx] = (char)NO_NEED_TO_ENCODE;

		return;
	}
#endif

	// this pixel color's data grame pointer.
	char* pixelData = hedder + BLOCK_HEDDER_SIZE + ((size_t)cIdx << ENC_BUFFER_BLOCK_SIZE_LOG2);

	// DCTFrame's data gram index.
	short* dctFrameBufferPt =
		dctFrameBuffer +
		(size_t)bIdx *
		(BLOCK_SIZE * DST_COLOR_SIZE) + ((size_t)cIdx << BLOCK_SIZE_LOG2);

	short level = 0;
	unsigned char run = 0;
	unsigned char sum = 0;

#pragma unroll

	// 0 ~ 62 (63 ELEM)
	for (int i = 0; i < BLOCK_SIZE - 1; i++, dctFrameBufferPt++) {
		level = dctFrameBufferPt[PIXEL_VALUE];

		if (level == 0 || level == (short)(1 << 15)) {
			// if DCT buffer's level is invalid.

			// count up run value.
			run++;

			continue;
		}

		// if DCT buffer's level is validity.
		// set level and run value to buffer.

		// Convert to a positive number so that the sign bit can also be shifted.
		pixelData[BIG_ENDIAN_DIX] = (char)((unsigned short)level << LEVEL_BIG_OFFSET);

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

		pixelData[LITTLE_ENDIAN_IDX] =
			(char)(
				((unsigned short)run << RUN_OFFSET) |
				((unsigned short)level >> LEVEL_LITTLE_OFFSET)
			);

		// printf("level: %d\n", (unsigned short)(level >> LEVEL_LITTLE_OFFSET));

		// | | | | | | | |
		// count buffer sum
		sum++;

		// reset run value.
		run = 0;

		// advance the pointer.
		pixelData += ENDIAN_SIZE;
	}

	// 63 (1ELEM)
	level = dctFrameBufferPt[PIXEL_VALUE];
	pixelData[BIG_ENDIAN_DIX] = (char)((unsigned short)level << LEVEL_BIG_OFFSET);
	pixelData[LITTLE_ENDIAN_IDX] =
		(char)(
			((unsigned short)run << RUN_OFFSET) |
			((unsigned short)level >> LEVEL_LITTLE_OFFSET)
		);

	// printf("level: %d\n", (unsigned short)(level >> LEVEL_LITTLE_OFFSET));

	// Reset save level's count.
	hedder[BLOCK_COLOR_SIZE_IDX_OFFSET + (int)cIdx] = (char)++sum;
}

__global__ void EntropyInvert(char* encFrame, short* dctFrameBuffer) {
	// Block index.
	const short bIdx = blockIdx.y * gridDim.x + blockIdx.x;
	const char cIdx = threadIdx.x;

	// EncFrame's block hedder.
	char* hedder = encFrame + (size_t)bIdx * (BLOCK_HEDDER_SIZE + ENC_BUFFER_BLOCK_SIZE * DST_COLOR_SIZE);

	// this pixel color's data grame pointer.
	char* pixelData = hedder + BLOCK_HEDDER_SIZE + ((size_t)cIdx << ENC_BUFFER_BLOCK_SIZE_LOG2);

	// DiffFrame's data gram index.
	short* dctFrameBufferPt = dctFrameBuffer + (size_t)bIdx * (BLOCK_SIZE * DST_COLOR_SIZE) + ((size_t)cIdx << BLOCK_SIZE_LOG2);

	unsigned char bigEndian;
	unsigned char littleEndian;
	unsigned char run;

	for (int i = 0, int count = 0; i < (BLOCK_SIZE - 1); i++, pixelData += ENDIAN_SIZE, count++) {
		bigEndian = (unsigned char)pixelData[BIG_ENDIAN_DIX];
		littleEndian = (unsigned char)pixelData[LITTLE_ENDIAN_IDX];

		// extract only bits corresponding to run.
		// 1 1 1 1 1 1 1 1
		//       &
		// 0 1 1 1 1 1 1 0 (= 126)
		//      >> 1
		run = (littleEndian & (unsigned char)126) >> RUN_OFFSET;
		i += run;

#if ERROR_CHECK
		if (i > 63) {
			printf("access validation."
				   "run: % d, i : % d, count : % d, size : % d\n",
				   run, i, count, hedder[BLOCK_COLOR_SIZE_IDX_OFFSET + cIdx]
			);
			return;
		}
#endif

		// 1 0 0 0 0 0 0 1 (= 129)
		dctFrameBufferPt[i] =
			(short)((unsigned short)(littleEndian & (unsigned char)129) << LEVEL_LITTLE_OFFSET) |
			(short)((unsigned short)bigEndian >> LEVEL_BIG_OFFSET);
	}
}