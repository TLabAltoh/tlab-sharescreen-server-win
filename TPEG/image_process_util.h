#pragma once

#include "test_common.h"
#include "test_util.h"
#include "TPEG.h"
#include "tpeg_common.h"

CImage org_frame;
int width;
int height;
int pitch;
int res;
int block_num;
int src_frame_buffer_size;
int encoded_frame_buffer_size;
unsigned short* block_diff_sum_buffer;
unsigned char* org_frame_buffer;
unsigned char* src_frame_buffer;
unsigned char* decoded_frame_buffer;
char* encoded_frame_buffer;

char* InitializeEncodedFrameBuffer(int size) {
	char* encoded_frame_bufferTmp = new char[size];

	for (int i = 0; i < size; i++)
		encoded_frame_bufferTmp[i] = (char)0;

	return encoded_frame_bufferTmp;
}

unsigned char* InitializeDecodedFrameBuffer(int size) {
	unsigned char* decoded_frame_bufferTmp = new unsigned char[size];

	for (int i = 0; i < size; i++)
		decoded_frame_bufferTmp[i] = (char)0;

	return decoded_frame_bufferTmp;
}

unsigned short* InitializeBlockDiffSumBuffer(int size) {
	unsigned short* block_diff_sum_bufferTmp = new unsigned short[size];

	for (int i = 0; i < size; i++)
		block_diff_sum_bufferTmp[i] = (unsigned short)0;

	return block_diff_sum_bufferTmp;
}

int EncodeImage(char* path, int color_size) {
	if (CheckFilePath(path, (char*)".png") == 1) {
		return 1;
	}

	if (org_frame.Load(path) == S_FALSE) {
		printf("Error: org_frame.Load");
		return 1;
	}

	printf(
		"org_frame.Save(sample_image.bmp): %s\n",
		org_frame.Save("sample_image.bmp") == S_OK ? "succeeded" : "failed");

	width = org_frame.GetWidth();
	height = org_frame.GetHeight();
	pitch = org_frame.GetPitch();
	res = width * height;

	printf("input image info ---------------------------\n");
	printf("widht\t:\t%d\n", width);
	printf("height\t:\t%d\n", height);
	printf("pitch\t:\t%d\n", pitch);
	printf("resolutin\t:\t %d\n", res);
	printf("width * height\t:\t %d\n", width * height);
	printf("color size:\t%d\t\n", color_size);
	printf("---------------------------------------\n\n");

	block_num = (width >> BLOCK_AXIS_SIZE_LOG2) * (height >> BLOCK_AXIS_SIZE_LOG2);

	src_frame_buffer_size = width * height * SRC_COLOR_SIZE;
	encoded_frame_buffer_size =
		block_num * BLOCK_HEDDER_SIZE +
		width * height * ENDIAN_SIZE * DST_COLOR_SIZE;

	block_diff_sum_buffer = InitializeBlockDiffSumBuffer(block_num);
	src_frame_buffer = InitializeDecodedFrameBuffer(src_frame_buffer_size);
	decoded_frame_buffer = InitializeDecodedFrameBuffer(src_frame_buffer_size);
	encoded_frame_buffer = InitializeEncodedFrameBuffer(encoded_frame_buffer_size);

	if (TPEG::InitializeDevice(
		width, height,
		encoded_frame_buffer,
		decoded_frame_buffer) == 1) {
		printf("Error: TPEG::InitializeDevice\n");
		return 1;
	}

	printf("\ntpeg device initialized ...\n");

	int src_idx = 0;
	for (int i = 0; i < height; i++) {
		org_frame_buffer = (unsigned char*)org_frame.GetBits() + (size_t)i * pitch;
		for (int j = 0; j < width; j++) {
			src_frame_buffer[src_idx + R_INDEX] = org_frame_buffer[R_INDEX];
			src_frame_buffer[src_idx + G_INDEX] = org_frame_buffer[G_INDEX];
			src_frame_buffer[src_idx + B_INDEX] = org_frame_buffer[B_INDEX];
			src_frame_buffer[src_idx + A_INDEX] = org_frame_buffer[A_INDEX];
			src_idx += SRC_COLOR_SIZE;
			org_frame_buffer += color_size;
		}
	}

	int loop_num;

	printf("\nbitmap copyed ...\n");

#if true
	loop_num = BLOCK_SIZE;
	printf("Log encoded_frame_buffer first one block -------------------\n\n");
	for (int i = 0; i < loop_num; i++) {
		printf("src_frame_buffer[%d]\t: %d\n", i, src_frame_buffer[i]);
	}
	printf("------------------------------------------------------\n\n");
#endif

	clock_t start = clock();
	TPEG::EncodeFrame(src_frame_buffer);
	clock_t end = clock();

	printf("\nbitmap encoded ...\n");

	TPEG::DestroyDevice();

#if true
	loop_num = BLOCK_SIZE * ENDIAN_SIZE * DST_COLOR_SIZE;
	printf("Log encoded_frame_buffer first one block -------------\n\n");
	for (int i = 0; i < loop_num; i++) {
		printf("encoded_frame_buffer[%d]\t: %d\n", i, encoded_frame_buffer[i]);
	}
	printf("------------------------------------------------------\n\n");
#endif

#if true
	printf("Log decoded_frame_buffer first one block -------------------\n\n");
	loop_num = BLOCK_SIZE;
	for (int i = 0; i < loop_num; i++) {
		printf("decoded_frame_buffer[%d]\t: %d\n", i, decoded_frame_buffer[i]);
	}
	printf("------------------------------------------------------\n\n");
#endif

#if true
	printf("Log encoded_frame_buffer size ------------------------------\n\n");
	loop_num = width / BLOCK_AXIS_SIZE * height / BLOCK_AXIS_SIZE;
	int sum = 0;
	for (int i = 0; i < loop_num; i++) {
		char* encoded_frame_buffer_ptr = encoded_frame_buffer + (size_t)i * (BLOCK_HEDDER_SIZE + BLOCK_SIZE * DST_COLOR_SIZE * ENDIAN_SIZE);

		unsigned int tmp = encoded_frame_buffer_ptr[BLOCK_BIT_SIZE_B + R_INDEX];
		if ((tmp > 64 || tmp < 0)) printf("inviled value in r: %d [%d]\n", tmp, i);
		sum += tmp * ENDIAN_SIZE;

		tmp = encoded_frame_buffer_ptr[BLOCK_BIT_SIZE_B + G_INDEX];
		if ((tmp > 64 || tmp < 0)) printf("inviled value in g: %d [%d]\n", tmp, i);
		sum += tmp * ENDIAN_SIZE;

		tmp = encoded_frame_buffer_ptr[BLOCK_BIT_SIZE_B + B_INDEX];
		if ((tmp > 64 || tmp < 0)) printf("inviled value in b: %d [%d]\n", tmp, i);
		sum += tmp * ENDIAN_SIZE;
	}
	float befor_size = (float)width * height * DST_COLOR_SIZE * pow(10, -3);
	float after_size = (float)sum * pow(10, -3);
	printf("original frame size: %f\n", befor_size);
	printf("decoded frame size: %f\n", after_size);
	printf("compression rate: %f %%\n", after_size / (double)befor_size * 100);
	printf("------------------------------------------------------\n\n");
#endif

	// copy decoded frame to org_frame_buffer.
	src_idx = 0;
	for (int i = 0; i < height; i++) {
		org_frame_buffer = (unsigned char*)org_frame.GetBits() + (size_t)i * pitch;
		for (int j = 0; j < width; j++) {
			org_frame_buffer[R_INDEX] = decoded_frame_buffer[src_idx + R_INDEX];
			org_frame_buffer[G_INDEX] = decoded_frame_buffer[src_idx + G_INDEX];
			org_frame_buffer[B_INDEX] = decoded_frame_buffer[src_idx + B_INDEX];
			org_frame_buffer[A_INDEX] = decoded_frame_buffer[src_idx + A_INDEX];
			src_idx += SRC_COLOR_SIZE;
			org_frame_buffer += color_size;
		}
	}

	printf(
		"org_frame.Save(decoded_result.bmp): %s\n",
		org_frame.Save("decoded_result.bmp") == S_OK ? "succeeded" : "failed");

	printf("Finish all process !\n");

	return end - start;
}
