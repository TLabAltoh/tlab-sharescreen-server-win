#pragma once

#include "tpeg_common.h"
#include "tpeg_cuda.h"

__device__ void CUD8x8DCT_Butterfly(float* Vect0, int Step)
{
	float* Vect1 = Vect0 + Step;
	float* Vect2 = Vect1 + Step;
	float* Vect3 = Vect2 + Step;
	float* Vect4 = Vect3 + Step;
	float* Vect5 = Vect4 + Step;
	float* Vect6 = Vect5 + Step;
	float* Vect7 = Vect6 + Step;

	float X07P = (*Vect0) + (*Vect7);
	float X16P = (*Vect1) + (*Vect6);
	float X25P = (*Vect2) + (*Vect5);
	float X34P = (*Vect3) + (*Vect4);

	float X07M = (*Vect0) - (*Vect7);
	float X61M = (*Vect6) - (*Vect1);
	float X25M = (*Vect2) - (*Vect5);
	float X43M = (*Vect4) - (*Vect3);

	float X07P34PP = X07P + X34P;
	float X07P34PM = X07P - X34P;
	float X16P25PP = X16P + X25P;
	float X16P25PM = X16P - X25P;

	(*Vect0) = C_norm * (X07P34PP + X16P25PP);
	(*Vect2) = C_norm * (C_b * X07P34PM + C_e * X16P25PM);
	(*Vect4) = C_norm * (X07P34PP - X16P25PP);
	(*Vect6) = C_norm * (C_e * X07P34PM - C_b * X16P25PM);

	(*Vect1) = C_norm * (C_a * X07M - C_c * X61M + C_d * X25M - C_f * X43M);
	(*Vect3) = C_norm * (C_c * X07M + C_f * X61M - C_a * X25M + C_d * X43M);
	(*Vect5) = C_norm * (C_d * X07M + C_a * X61M + C_f * X25M - C_c * X43M);
	(*Vect7) = C_norm * (C_f * X07M + C_d * X61M + C_c * X25M + C_a * X43M);
}

__global__ void CUD8x8DCT_RGBFrame(unsigned char* frame_buffer, short* dct_result_buffer, unsigned short* block_diff_sum_buffer)
{
	__shared__ float block_y[BLOCK_SIZE];
	__shared__ float block_cr[BLOCK_SIZE];
	__shared__ float block_cb[BLOCK_SIZE];

	const int block_dispatch_idx = blockIdx.y * gridDim.x + blockIdx.x;

	if (block_diff_sum_buffer[block_dispatch_idx] == 0) return;

	short* dct_result_buffer_ptr = dct_result_buffer + (size_t)block_dispatch_idx * BLOCK_SIZE * DST_COLOR_SIZE;

	unsigned int frame_buffer_y = blockIdx.y << BLOCK_AXIS_SIZE_LOG2;
	unsigned int frame_buffer_x = blockIdx.x << BLOCK_AXIS_SIZE_LOG2;
	unsigned int frame_buffer_stride = gridDim.x << BLOCK_AXIS_SIZE_LOG2;

	float* block_y_ptr = block_y;
	float* block_cr_ptr = block_cr;
	float* block_cb_ptr = block_cb;

#pragma unroll

	for (unsigned int i = 0; i < BLOCK_AXIS_SIZE; i++) {
		for (unsigned int j = 0; j < BLOCK_AXIS_SIZE; j++) {

			unsigned int frame_buffer_idx = (frame_buffer_y + j) * frame_buffer_stride + (frame_buffer_x + i);
			frame_buffer_idx *= DST_COLOR_SIZE;
			
			unsigned char* frame_buffer_ptr = frame_buffer + frame_buffer_idx;

			unsigned int block_copy_dst_idx = (j << BLOCK_AXIS_SIZE_LOG2) + i;

			const unsigned char r = frame_buffer_ptr[R_INDEX];
			const unsigned char g = frame_buffer_ptr[G_INDEX];
			const unsigned char b = frame_buffer_ptr[B_INDEX];

			block_y_ptr[block_copy_dst_idx] = Conv2Y(r, g, b);
			block_cr_ptr[block_copy_dst_idx] = Conv2Cr(r, g, b);
			block_cb_ptr[block_copy_dst_idx] = Conv2Cb(r, g, b);
		}
	}

	// rocess rows
	unsigned int row_offset;
	for (unsigned int row = 0; row < BLOCK_AXIS_SIZE; row++) {
		row_offset = row << BLOCK_AXIS_SIZE_LOG2;
		CUD8x8DCT_Butterfly(block_y + row_offset, 1);
		CUD8x8DCT_Butterfly(block_cr + row_offset, 1);
		CUD8x8DCT_Butterfly(block_cb + row_offset, 1);
	}

	// process columns
	unsigned int col_offset;
	for (unsigned int col = 0; col < BLOCK_AXIS_SIZE; col++) {
		col_offset = col << BLOCK_AXIS_SIZE_LOG2;
		CUD8x8DCT_Butterfly(block_y + col, BLOCK_AXIS_SIZE);
		CUD8x8DCT_Butterfly(block_cr + col, BLOCK_AXIS_SIZE);
		CUD8x8DCT_Butterfly(block_cb + col, BLOCK_AXIS_SIZE);
	}

	for (unsigned int i = 0; i < BLOCK_SIZE; i++) {
		const unsigned char zigzagIndex = ZigZagIndexForward[i];
		const float quantizationLuminance = InvertQuantizationTable50Luminance[i];
		const float quantizationChrominance = InvertQuantizationTable50Chrominance[i];

		dct_result_buffer_ptr[(Y_INDEX << BLOCK_SIZE_LOG2) + zigzagIndex] = Float2SignedShort(block_y_ptr[i], quantizationLuminance);
		dct_result_buffer_ptr[(Cr_INDEX << BLOCK_SIZE_LOG2) + zigzagIndex] = Float2SignedShort(block_cr_ptr[i], quantizationChrominance);
		dct_result_buffer_ptr[(Cb_INDEX << BLOCK_SIZE_LOG2) + zigzagIndex] = Float2SignedShort(block_cb_ptr[i], quantizationChrominance);
	}
}

__device__ void CUD8x8IDCT_Butterfly(float* Vect0, int Step)
{
	float* Vect1 = Vect0 + Step;
	float* Vect2 = Vect1 + Step;
	float* Vect3 = Vect2 + Step;
	float* Vect4 = Vect3 + Step;
	float* Vect5 = Vect4 + Step;
	float* Vect6 = Vect5 + Step;
	float* Vect7 = Vect6 + Step;

	float Y04P = (*Vect0) + (*Vect4);
	float Y2b6eP = C_b * (*Vect2) + C_e * (*Vect6);

	float Y04P2b6ePP = Y04P + Y2b6eP;
	float Y04P2b6ePM = Y04P - Y2b6eP;
	float Y7f1aP3c5dPP = C_f * (*Vect7) + C_a * (*Vect1) + C_c * (*Vect3) + C_d * (*Vect5);
	float Y7a1fM3d5cMP = C_a * (*Vect7) - C_f * (*Vect1) + C_d * (*Vect3) - C_c * (*Vect5);

	float Y04M = (*Vect0) - (*Vect4);
	float Y2e6bM = C_e * (*Vect2) - C_b * (*Vect6);

	float Y04M2e6bMP = Y04M + Y2e6bM;
	float Y04M2e6bMM = Y04M - Y2e6bM;
	float Y1c7dM3f5aPM = C_c * (*Vect1) - C_d * (*Vect7) - C_f * (*Vect3) - C_a * (*Vect5);
	float Y1d7cP3a5fMM = C_d * (*Vect1) + C_c * (*Vect7) - C_a * (*Vect3) + C_f * (*Vect5);

	(*Vect0) = C_norm * (Y04P2b6ePP + Y7f1aP3c5dPP);
	(*Vect7) = C_norm * (Y04P2b6ePP - Y7f1aP3c5dPP);
	(*Vect4) = C_norm * (Y04P2b6ePM + Y7a1fM3d5cMP);
	(*Vect3) = C_norm * (Y04P2b6ePM - Y7a1fM3d5cMP);

	(*Vect1) = C_norm * (Y04M2e6bMP + Y1c7dM3f5aPM);
	(*Vect5) = C_norm * (Y04M2e6bMM - Y1d7cP3a5fMM);
	(*Vect2) = C_norm * (Y04M2e6bMM + Y1d7cP3a5fMM);
	(*Vect6) = C_norm * (Y04M2e6bMP - Y1c7dM3f5aPM);
}


__global__ void CUD8x8IDCT_RGBFrame(short* dct_result_buffer, unsigned char* frame_buffer) {
	__shared__ float block_y[BLOCK_SIZE];
	__shared__ float block_cr[BLOCK_SIZE];
	__shared__ float block_cb[BLOCK_SIZE];

	const int block_dispatch_idx = blockIdx.y * gridDim.x + blockIdx.x;

	short* dct_result_buffer_ptr = dct_result_buffer + (size_t)block_dispatch_idx * BLOCK_SIZE * DST_COLOR_SIZE;

	float* block_y_ptr = block_y;
	float* block_cr_ptr = block_cr;
	float* block_cb_ptr = block_cb;

#pragma unroll

	for (unsigned int i = 0; i < BLOCK_SIZE; i++) {
		const unsigned char zigzagInvert = ZigZagIndexInvert[i];
		const int quantizationLuminance = ForwardQuantizationTable50Luminance[zigzagInvert];
		const int quantizationChrominance = ForwardQuantizationTable50Chrominance[zigzagInvert];

		const short swapY = dct_result_buffer_ptr[(Y_INDEX << BLOCK_SIZE_LOG2) + i];
		const short swapCr = dct_result_buffer_ptr[(Cr_INDEX << BLOCK_SIZE_LOG2) + i];
		const short swapCb = dct_result_buffer_ptr[(Cb_INDEX << BLOCK_SIZE_LOG2) + i];

		block_y_ptr[zigzagInvert] = SignedShort2Float(swapY, quantizationLuminance);;
		block_cr_ptr[zigzagInvert] = SignedShort2Float(swapCr, quantizationChrominance);
		block_cb_ptr[zigzagInvert] = SignedShort2Float(swapCb, quantizationChrominance);
	}

	// rocess rows
	unsigned int row_offset;
	for (unsigned int row = 0; row < BLOCK_AXIS_SIZE; row++) {
		row_offset = row << BLOCK_AXIS_SIZE_LOG2;
		CUD8x8IDCT_Butterfly(block_y + row_offset, 1);
		CUD8x8IDCT_Butterfly(block_cr + row_offset, 1);
		CUD8x8IDCT_Butterfly(block_cb + row_offset, 1);
	}

	// process columns
	unsigned int col_offset;
	for (unsigned int col = 0; col < BLOCK_AXIS_SIZE; col++) {
		col_offset = col << BLOCK_AXIS_SIZE_LOG2;
		CUD8x8IDCT_Butterfly(block_y + col, BLOCK_AXIS_SIZE);
		CUD8x8IDCT_Butterfly(block_cr + col, BLOCK_AXIS_SIZE);
		CUD8x8IDCT_Butterfly(block_cb + col, BLOCK_AXIS_SIZE);
	}

	unsigned int frame_buffer_y = blockIdx.y << BLOCK_AXIS_SIZE_LOG2;
	unsigned int frame_buffer_x = blockIdx.x << BLOCK_AXIS_SIZE_LOG2;
	unsigned int frame_buffer_stride = gridDim.x << BLOCK_AXIS_SIZE_LOG2;

	for (unsigned int i = 0; i < BLOCK_AXIS_SIZE; i++) {
		for (unsigned int j = 0; j < BLOCK_AXIS_SIZE; j++) {

			unsigned int frame_buffer_idx = (frame_buffer_y + j) * frame_buffer_stride + (frame_buffer_x + i);
			frame_buffer_idx *= SRC_COLOR_SIZE;

			unsigned char* frame_buffer_ptr = frame_buffer + frame_buffer_idx;

			unsigned int block_copy_dst_idx = (j << BLOCK_AXIS_SIZE_LOG2) + i;

			float y = block_y[block_copy_dst_idx];
			float cr = block_cr[block_copy_dst_idx];
			float cb = block_cb[block_copy_dst_idx];

			float r = Conv2R(y, cr);
			float g = Conv2G(y, cr, cb);
			float b = Conv2B(y, cb);

			char rb = r > 255.0f;
			char gb = g > 255.0f;
			char bb = b > 255.0f;

			frame_buffer_ptr[R_INDEX] = (unsigned char)(rb * 255.0f + (1.0f - rb) * r);
			frame_buffer_ptr[G_INDEX] = (unsigned char)(gb * 255.0f + (1.0f - gb) * g);
			frame_buffer_ptr[B_INDEX] = (unsigned char)(bb * 255.0f + (1.0f - bb) * b);
			frame_buffer_ptr[A_INDEX] = (unsigned char)255;
		}
	}
}

__device__ inline void ErrorCheck(float result, float quantization, char* color) {
	if ((short)(result * quantization) > (short)256 ||
		(short)(result * quantization) < (short)-1)
		printf("error result%s: %d\n", color, (short)(result * quantization));
}