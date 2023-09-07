#pragma once

#include "TPEG_Common.h"

// This is an algorithm based on the orthogonality of trigonometric functions
// https://youtu.be/HNHb0_mOTYw?t=664

#pragma region CosinTable
__constant__ float DCTv8matrix[] = {
	0.3535533905932738f,  0.4903926402016152f,  0.4619397662556434f,  0.4157348061512726f,  0.3535533905932738f,  0.2777851165098011f,  0.1913417161825449f,  0.0975451610080642f,
	0.3535533905932738f,  0.4157348061512726f,  0.1913417161825449f, -0.0975451610080641f, -0.3535533905932737f, -0.4903926402016152f, -0.4619397662556434f, -0.2777851165098011f,
	0.3535533905932738f,  0.2777851165098011f, -0.1913417161825449f, -0.4903926402016152f, -0.3535533905932738f,  0.0975451610080642f,  0.4619397662556433f,  0.4157348061512727f,
	0.3535533905932738f,  0.0975451610080642f, -0.4619397662556434f, -0.2777851165098011f,  0.3535533905932737f,  0.4157348061512727f, -0.1913417161825450f, -0.4903926402016153f,
	0.3535533905932738f, -0.0975451610080641f, -0.4619397662556434f,  0.2777851165098009f,  0.3535533905932738f, -0.4157348061512726f, -0.1913417161825453f,  0.4903926402016152f,
	0.3535533905932738f, -0.2777851165098010f, -0.1913417161825452f,  0.4903926402016153f, -0.3535533905932733f, -0.0975451610080649f,  0.4619397662556437f, -0.4157348061512720f,
	0.3535533905932738f, -0.4157348061512727f,  0.1913417161825450f,  0.0975451610080640f, -0.3535533905932736f,  0.4903926402016152f, -0.4619397662556435f,  0.2777851165098022f,
	0.3535533905932738f, -0.4903926402016152f,  0.4619397662556433f, -0.4157348061512721f,  0.3535533905932733f, -0.2777851165098008f,  0.1913417161825431f, -0.0975451610080625f
};
#pragma endregion

#pragma region ZigZagIndex
__constant__ int ZigZagIndexForward[] = {
	 0,  1,  5,  6, 14, 15, 27, 28,
	 2,  4,  7, 13, 16, 26, 29, 42,
	 3,  8, 12, 17, 25, 30, 41, 43,
	 9, 11, 18, 24, 31, 40, 44, 53,
	10, 19, 23, 32, 39, 45, 52, 54,
	20, 22, 33, 38, 46, 51, 55, 60,
	21, 34, 37, 47, 50, 56, 59, 61,
	35, 36, 48, 49, 57, 58, 62, 63
};

__constant__ int ZigZagIndexInvert[] = {
	 0,  1,  8, 16,  9,  2,  3, 10,
	17, 24, 32, 25, 18, 11,  4,  5,
	12, 19, 26, 33, 40, 48, 41, 34,
	27, 20, 13,  6,  7, 14, 21, 28,
	35, 42, 49, 56, 57, 50, 43, 36,
	29, 22, 15, 23, 30, 37, 44, 51,
	58, 59, 52, 45, 38, 31, 39, 46,
	53, 60, 61, 54, 47, 55, 62, 63
};
#pragma endregion

// https://www.fastcompression.com/blog/jpeg-optimization-review.htm
#pragma region QuantizationTable
// 50% COMPRESSION
__constant__ int ForwardQuantizationTable50Luminance[] = {
	 16,  11,  10,  16,  24,  40,  51,  61,
	 12,  12,  14,  19,  26,  58,  60,  55,
	 14,  13,  16,  24,  40,  57,  69,  56,
	 14,  17,  22,  29,  51,  87,  80,  62,
	 18,  22,  37,  56,  68, 109, 103,  77,
	 24,  35,  55,  64,  81, 104, 113,  92,
	 49,  64,  78,  87, 103, 121, 120, 101,
	 72,  92,  95,  98, 112, 100, 103,  99
};

__constant__ int ForwardQuantizationTable50Chrominance[] = {
	17, 18, 42, 47, 99, 99, 99, 99,
	18, 21, 26, 66, 99, 99, 99, 99,
	24, 26, 56, 99, 99, 99, 99, 99,
	47, 66, 99, 99, 99, 99, 99, 99,
	99, 99, 99, 99, 99, 99, 99, 99,
	99, 99, 99, 99, 99, 99, 99, 99,
	99, 99, 99, 99, 99, 99, 99, 99,
	99, 99, 99, 99, 99, 99, 99, 99
};

__constant__ float InvertQuantizationTable50Luminance[] = {
	0.062500f, 0.090909f, 0.100000f, 0.062500f, 0.041667f, 0.025000f, 0.019608f, 0.016393,
	0.083333f, 0.083333f, 0.071429f, 0.052632f, 0.038462f, 0.017241f, 0.016667f, 0.018182,
	0.071429f, 0.076923f, 0.062500f, 0.041667f, 0.025000f, 0.017544f, 0.014493f, 0.017857,
	0.071429f, 0.058824f, 0.045455f, 0.034483f, 0.019608f, 0.011494f, 0.012500f, 0.016129,
	0.055556f, 0.045455f, 0.027027f, 0.017857f, 0.014706f, 0.009174f, 0.009709f, 0.012987,
	0.041667f, 0.028571f, 0.018182f, 0.015625f, 0.012346f, 0.009615f, 0.008850f, 0.010870,
	0.020408f, 0.015625f, 0.012821f, 0.011494f, 0.009709f, 0.008264f, 0.008333f, 0.009901,
	0.013889f, 0.010870f, 0.010526f, 0.010204f, 0.008929f, 0.010000f, 0.009709f, 0.010101
};

__constant__ float InvertQuantizationTable50Chrominance[] = {
	0.058824f, 0.055556f, 0.023810f, 0.021277f, 0.010101f, 0.010101f, 0.010101f, 0.010101f,
	0.055556f, 0.047619f, 0.038462f, 0.015152f, 0.010101f, 0.010101f, 0.010101f, 0.010101f,
	0.041667f, 0.038462f, 0.017857f, 0.010101f, 0.010101f, 0.010101f, 0.010101f, 0.010101f,
	0.021277f, 0.015152f, 0.010101f, 0.010101f, 0.010101f, 0.010101f, 0.010101f, 0.010101f,
	0.010101f, 0.010101f, 0.010101f, 0.010101f, 0.010101f, 0.010101f, 0.010101f, 0.010101f,
	0.010101f, 0.010101f, 0.010101f, 0.010101f, 0.010101f, 0.010101f, 0.010101f, 0.010101f,
	0.010101f, 0.010101f, 0.010101f, 0.010101f, 0.010101f, 0.010101f, 0.010101f, 0.010101f,
	0.010101f, 0.010101f, 0.010101f, 0.010101f, 0.010101f, 0.010101f, 0.010101f, 0.010101f
};
#pragma endregion

namespace RGB2yCbCr_InvertConvertor {
	/*
	* R	= Y+1.402Cr
	* G	= Y-0.714Cr-0.344Cb
	* B	= Y+1.772Cb
	*/
	__device__ inline float R(float y, float cr) {
		return y + 1.402 * (cr - 128);
	}

	__device__ inline float G(float y, float cr, float cb) {
		return y - 0.7141 * (cr - 128) - 0.3441 * (cb - 128);
	}

	__device__ inline float B(float y, float cb) {
		return y + 1.772 * (cb - 128);
	}
}

namespace RGB2yCbCr_ForwardConvertor {
	/*
	* Y	= 0.299R+0.587G+0.114B
	* Cr = 0.500R-0.419G-0.081B
	* Cb = -0.169R-0.332G+0.500B
	*/
	__device__ inline unsigned char Y(unsigned char r, unsigned char g, unsigned char b) {
		return (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b);
	}

	__device__ inline unsigned char Cr(unsigned char r, unsigned char g, unsigned char b) {
		return (unsigned char)(0.500 * r - 0.4187 * g - 0.0813 * b + 128);
	}

	__device__ inline unsigned char Cb(unsigned char r, unsigned char g, unsigned char b) {
		return (unsigned char)(-0.169 * r - 0.322 * g + 0.500 * b + 128);
	}
};

// Temporary blocks
float __shared__ CurBlockLocal1[BLOCK_SIZE * DST_COLOR_SIZE];
float __shared__ CurBlockLocal2[BLOCK_SIZE * DST_COLOR_SIZE];

__device__ inline short TmpForward(char sign, float result, float quantization) {
	return (short)(
		(unsigned short)(result * quantization * (1 - sign * 2)) +
		(unsigned short)(sign * (1 << (sizeof(unsigned short) * 8 - 1)))
	);
}

__device__ inline float TmpInvert(char sign, short swap, int quantization) {
	return (float)(
		(short)(
			(unsigned short)swap -
			(unsigned short)(sign * (1 << (sizeof(unsigned short) * 8 - 1)))
		) * (1 - sign * 2) * quantization
	);
}

__device__ inline void ErrorCheck(float result, float quantization, char* color) {
	if ((short)(result * quantization) > (short)256 ||
		(short)(result * quantization) < (short)-1)
		printf("error result%s: %d\n", color, (short)(result * quantization));
}

__global__ void DCTForward(
	unsigned char* prevFrameBuffer,
	unsigned short* blockDiffSumBuffer,
	short* dctForwardFrameBuffer
) {
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;
	const int bx = blockIdx.x;
	const int by = blockIdx.y;

	const int bIdx = by * gridDim.x + bx;
	const int tIdx = (ty << BLOCK_AXIS_SIZE_LOG2) + tx;

	const int bOffset = bIdx * BLOCK_SIZE * DST_COLOR_SIZE;
	const int tOffset = tIdx * DST_COLOR_SIZE;

	// check this block need to send.
	// if false, reset with zero to this block's change pixel cont.
#if 1
	if (blockDiffSumBuffer[bIdx] == 0) return;
#endif

	// Start DCT
	short* dctFrameBufferPt = dctForwardFrameBuffer + bOffset;
	unsigned char* prevFrameBufferPt = prevFrameBuffer + bOffset + tOffset;

	CurBlockLocal1[tOffset + Y_IDX] = RGB2yCbCr_ForwardConvertor::Y(
		prevFrameBufferPt[R_IDX],
		prevFrameBufferPt[G_IDX],
		prevFrameBufferPt[B_IDX]
	);
	CurBlockLocal1[tOffset + Cr_IDX] = RGB2yCbCr_ForwardConvertor::Cr(
		prevFrameBufferPt[R_IDX],
		prevFrameBufferPt[G_IDX],
		prevFrameBufferPt[B_IDX]
	);
	CurBlockLocal1[tOffset + Cb_IDX] = RGB2yCbCr_ForwardConvertor::Cb(
		prevFrameBufferPt[R_IDX],
		prevFrameBufferPt[G_IDX],
		prevFrameBufferPt[B_IDX]
	);

	// this is intelisense worning. it can be safely ignored if project set up propely.
	// so I' cant build this code.
	__syncthreads();

	// Calculate column sums.
	float resultY = 0.0f;
	float resultCr = 0.0f;
	float resultCb = 0.0f;
	float currentElem;

	// line frequency.
	int DCTv8matrixIndex = (0 << BLOCK_AXIS_SIZE_LOG2) + ty;

	// what row.
	int CurBlockLocal1Index = ((0 << BLOCK_AXIS_SIZE_LOG2) + tx) * DST_COLOR_SIZE;

#pragma unroll

	for (int i = 0; i < BLOCK_AXIS_SIZE; i++) {
		currentElem = DCTv8matrix[DCTv8matrixIndex];
		resultY += currentElem * CurBlockLocal1[CurBlockLocal1Index + Y_IDX];
		resultCr += currentElem * CurBlockLocal1[CurBlockLocal1Index + Cr_IDX];
		resultCb += currentElem * CurBlockLocal1[CurBlockLocal1Index + Cb_IDX];
		DCTv8matrixIndex += BLOCK_AXIS_SIZE;
		CurBlockLocal1Index += BLOCK_AXIS_SIZE * DST_COLOR_SIZE;
	}

	CurBlockLocal2[tOffset + Y_IDX] = resultY;
	CurBlockLocal2[tOffset + Cr_IDX] = resultCr;
	CurBlockLocal2[tOffset + Cb_IDX] = resultCb;

	__syncthreads();

	// Calculate sum of columns.
	resultY = 0.0f;
	resultCr = 0.0f;
	resultCb = 0.0f;

	// column frequency.
	DCTv8matrixIndex = (0 << BLOCK_AXIS_SIZE_LOG2) + tx;

	// what line.
	int CurBlockLocal2Index = ((ty << BLOCK_AXIS_SIZE_LOG2) + 0) * DST_COLOR_SIZE;

#pragma unroll

	for (int i = 0; i < BLOCK_AXIS_SIZE; i++) {
		currentElem = DCTv8matrix[DCTv8matrixIndex];
		resultY += currentElem * CurBlockLocal2[CurBlockLocal2Index + Y_IDX];
		resultCr += currentElem * CurBlockLocal2[CurBlockLocal2Index + Cr_IDX];
		resultCb += currentElem * CurBlockLocal2[CurBlockLocal2Index + Cb_IDX];
		DCTv8matrixIndex += BLOCK_AXIS_SIZE;
		CurBlockLocal2Index += DST_COLOR_SIZE;
	}

	char signY = resultY < 0;
	char signCr = resultCr < 0;
	char signCb = resultCb < 0;

	const unsigned char zigzagIndex = ZigZagIndexForward[tIdx];
	float quantizationLuminance = InvertQuantizationTable50Luminance[tIdx];
	float quantizationChrominance = InvertQuantizationTable50Chrominance[tIdx];

	dctFrameBufferPt[(Y_IDX << BLOCK_SIZE_LOG2) + zigzagIndex] = TmpForward(
		signY, resultY, quantizationLuminance
	);
	dctFrameBufferPt[(Cr_IDX << BLOCK_SIZE_LOG2) + zigzagIndex] = TmpForward(
		signCr, resultCr, quantizationChrominance
	);
	dctFrameBufferPt[(Cb_IDX << BLOCK_SIZE_LOG2) + zigzagIndex] = TmpForward(
		signCb, resultCb, quantizationChrominance
	);

	__syncthreads();

	return;
}

__global__ void DCTInvert(short* dctInvertFrameBuffer, unsigned char* decFrameBuffer) {
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;
	const int bx = blockIdx.x;
	const int by = blockIdx.y;

	const int bIdx = by * gridDim.x + bx;
	const int tIdx = (ty << BLOCK_AXIS_SIZE_LOG2) + tx;

	const int bOffset = bIdx * BLOCK_SIZE * DST_COLOR_SIZE;
	const int tOffset = tIdx * DST_COLOR_SIZE;

	// decoded frame's pointer.
	unsigned char* decFrameBufferPt =
		decFrameBuffer +
		(
			(((size_t)by << BLOCK_AXIS_SIZE_LOG2) + ty) *
			((size_t)gridDim.x << BLOCK_AXIS_SIZE_LOG2) +
			(((size_t)bx << BLOCK_AXIS_SIZE_LOG2) + tx)
		) * ORG_COLOR_SIZE;

	short* dctFrameBufferPt = dctInvertFrameBuffer + bOffset;

	short swapY = dctFrameBufferPt[(Y_IDX << BLOCK_SIZE_LOG2) + tIdx];
	short swapCr = dctFrameBufferPt[(Cr_IDX << BLOCK_SIZE_LOG2) + tIdx];
	short swapCb = dctFrameBufferPt[(Cb_IDX << BLOCK_SIZE_LOG2) + tIdx];

	__syncthreads();
	
	// Rearrange the zigzag index to its original order.
	const unsigned char zigzagInvert = ZigZagIndexInvert[tIdx];
	const unsigned char zigzagInvertOffset = zigzagInvert * DST_COLOR_SIZE;
	char signY = swapY < 0;
	char signCr = swapCr < 0;
	char signCb = swapCb < 0;

	int quantizationLuminance = ForwardQuantizationTable50Luminance[zigzagInvert];
	int quantizationChrominance = ForwardQuantizationTable50Chrominance[zigzagInvert];
	CurBlockLocal1[zigzagInvertOffset + Y_IDX] = TmpInvert(signY, swapY, quantizationLuminance);
	CurBlockLocal1[zigzagInvertOffset + Cr_IDX] = TmpInvert(signCr, swapCr, quantizationChrominance);
	CurBlockLocal1[zigzagInvertOffset + Cb_IDX] = TmpInvert(signCb, swapCb, quantizationChrominance);

	__syncthreads();

	float resultY = 0.0f;
	float resultCr = 0.0f;
	float resultCb = 0.0f;
	float currentElem;

	// row 1.
	int DCTv8matrixIndex = (ty << BLOCK_AXIS_SIZE_LOG2) + 0;

	// what row.
	int CurBlockLocal1Index = ((0 << BLOCK_AXIS_SIZE_LOG2) + tx) * DST_COLOR_SIZE;

#pragma unroll

	// Calculate column sums.
	for (int i = 0; i < BLOCK_AXIS_SIZE; i++) {
		currentElem = DCTv8matrix[DCTv8matrixIndex];
		resultY += currentElem * CurBlockLocal1[CurBlockLocal1Index + Y_IDX];
		resultCr += currentElem * CurBlockLocal1[CurBlockLocal1Index + Cr_IDX];
		resultCb += currentElem * CurBlockLocal1[CurBlockLocal1Index + Cb_IDX];

		DCTv8matrixIndex++;
		CurBlockLocal1Index += BLOCK_AXIS_SIZE * DST_COLOR_SIZE;
	}

	CurBlockLocal2[tOffset + Y_IDX] = resultY;
	CurBlockLocal2[tOffset + Cr_IDX] = resultCr;
	CurBlockLocal2[tOffset + Cb_IDX] = resultCb;

	__syncthreads();

	resultY = 0.0f;
	resultCr = 0.0f;
	resultCb = 0.0f;

	// row 2.
	DCTv8matrixIndex = (tx << BLOCK_AXIS_SIZE_LOG2) + 0;

	// what line.
	int CurBlockLocal2Index = ((ty << BLOCK_AXIS_SIZE_LOG2) + 0) * DST_COLOR_SIZE;

#pragma unroll

	// Calculate sum of columns.
	for (int i = 0; i < BLOCK_AXIS_SIZE; i++)
	{
		currentElem = DCTv8matrix[DCTv8matrixIndex];
		resultY += currentElem * CurBlockLocal2[CurBlockLocal2Index + Y_IDX];
		resultCr += currentElem * CurBlockLocal2[CurBlockLocal2Index + Cr_IDX];
		resultCb += currentElem * CurBlockLocal2[CurBlockLocal2Index + Cb_IDX];

		DCTv8matrixIndex++;
		CurBlockLocal2Index += DST_COLOR_SIZE;
	}

	// lerp value. avoid over 255.
	float r = RGB2yCbCr_InvertConvertor::R(resultY, resultCr);
	float g = RGB2yCbCr_InvertConvertor::G(resultY, resultCr, resultCb);
	float b = RGB2yCbCr_InvertConvertor::B(resultY, resultCb);
	char rb = r > (float)255;
	char gb = g > (float)255;
	char bb = b > (float)255;
	decFrameBufferPt[R_IDX] = (unsigned char)(rb * (float)255 + (1 - rb) * r);
	decFrameBufferPt[G_IDX] = (unsigned char)(gb * (float)255 + (1 - gb) * g);
	decFrameBufferPt[B_IDX] = (unsigned char)(bb * (float)255 + (1 - bb) * b);
	decFrameBufferPt[A_IDX] = (unsigned char)255;

	return;
}