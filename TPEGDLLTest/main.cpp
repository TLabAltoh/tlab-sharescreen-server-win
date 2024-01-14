#pragma once

#include "stdio.h"
#include "stdlib.h"
#include "iostream"
#include "TPEG.h"
#include "image_process_util.h"
#include "test_common.h"

void memory_mollic_test() {
	std::cout << "---- this is test program using CUDA ----" << std::endl;

	float* host_x, * host_y;
	int N = 1000000;

	// Create CPUs area
	host_x = (float*)malloc(N * sizeof(float));
	host_y = (float*)malloc(N * sizeof(float));

	// do any function here.

	// Check if the calculation is done correctly.
	float sum = 0.0f;
	for (int j = 0; j < N; j++) {
		sum += host_y[j];
	}

	std::cout << "sum: " << sum << std::endl;

	std::cout << "---- process finish ! ----" << std::endl;
}

int cosin_table_create() {
	// create cosin value table.
	float pi = 3.141592654f;
	for (int i = 0; i < 8; i++) {
		float ch0 = i == 0 ? (float)1 / (float)sqrt(2) : 1;
		for (int j = 0; j < 8; j++) {
			float ch1 = j == 0 ? (float)1 / (float)sqrt(2) : 1;
			// float tmp = cos((float)((2 * i + 1) * j * pi) / (float)(2 * 8)) * (pow(2, 2) / pow(8, 2)) * ch0 * ch1;
			float tmp = cos((float)((2 * i + 1) * j * pi) / (float)(2 * 8)) * (pow(2, 2) / pow(8, 2));
			printf("%lf\t", tmp);
		}
		printf("\n");
	}

	return 0;
}

#pragma region CreateInvertZigZagIndex
int ZigZagIndexForward[] = {
	 0,  1,  5,  6, 14, 15, 27, 28,
	 2,  4,  7, 13, 16, 26, 29, 42,
	 3,  8, 12, 17, 25, 30, 41, 43,
	 9, 11, 18, 24, 31, 40, 44, 53,
	10, 19, 23, 32, 39, 45, 52, 54,
	20, 22, 33, 38, 46, 51, 55, 60,
	21, 34, 37, 47, 50, 56, 59, 61,
	35, 36, 48, 49, 57, 58, 62, 63
};

int ZigZagIndexInvert[BLOCK_SIZE];

int invert_zigzag_index_create() {
	for (int i = 0; i < BLOCK_SIZE; i++) {
		ZigZagIndexInvert[ZigZagIndexForward[i]] = i;
	}

	for (int i = 0; i < BLOCK_AXIS_SIZE; i++) {
		for (int j = 0; j < BLOCK_AXIS_SIZE; j++) {
			printf("%d\t", ZigZagIndexInvert[i * BLOCK_AXIS_SIZE + j]);
		}
		printf("\n");
	}

	return 0;
}
#pragma endregion

#pragma region CreateInvertQuantizationTable
int QuantizationTable50Luminance[] = {
	16,  11,  10,  16,  24,  40,  51,  61,
	12,  12,  14,  19,  26,  58,  60,  55,
	14,  13,  16,  24,  40,  57,  69,  56,
	14,  17,  22,  29,  51,  87,  80,  62,
	18,  22,  37,  56,  68, 109, 103,  77,
	24,  35,  55,  64,  81, 104, 113,  92,
	49,  64,  78,  87, 103, 121, 120, 101,
	72,  92,  95,  98, 112, 100, 103,  99
};

int QuantizationTable50Chrominance[] = {
	17, 18, 42, 47, 99, 99, 99, 99,
	18, 21, 26, 66, 99, 99, 99, 99,
	24, 26, 56, 99, 99, 99, 99, 99,
	47, 66, 99, 99, 99, 99, 99, 99,
	99, 99, 99, 99, 99, 99, 99, 99,
	99, 99, 99, 99, 99, 99, 99, 99,
	99, 99, 99, 99, 99, 99, 99, 99,
	99, 99, 99, 99, 99, 99, 99, 99
};

int invert_quantization_table_create(int forwardTable[]) {
	for (int i = 0; i < BLOCK_AXIS_SIZE; i++) {
		for (int j = 0; j < BLOCK_AXIS_SIZE; j++) {
			printf("%lf\t", (float)1 / (float)forwardTable[i * BLOCK_AXIS_SIZE + j]);
		}
		printf("\n");
	}
	return 0;
}

int invert_quantization_table_create(int type) {
	printf("start create invert_quantization_table type: %s\n", type == 0 ? "Luminance" : "Chrominance");
	switch (type)
	{
	case 0:
		invert_quantization_table_create(QuantizationTable50Luminance);
		break;
	case 1:
		invert_quantization_table_create(QuantizationTable50Chrominance);
		break;
	default:
		printf("no such level\n");
		break;
	}
	printf("finish create invert table\n");
	return 0;
}
#pragma endregion

int pixel_tweak(int* width, int* height, int block_axis_size) {
	// Adjust to be a multiple of 8.
	int tmp;
	float tmp1;
	float tmp2;

	tmp = *width / block_axis_size;
	tmp1 = (float)*width / (float)block_axis_size;
	tmp2 = tmp1 - tmp;

	if (tmp2 > 0) {
		*width = *width + tmp2 * block_axis_size;
	}

	tmp = *height / block_axis_size;
	tmp1 = (float)*height / (float)block_axis_size;
	tmp2 = tmp1 - tmp;

	if (tmp2 > 0) {
		*height = *height + tmp2 * block_axis_size;
	}

	return 0;
}

#pragma region tpeg_encode_test

int tpeg_encode_test() {
	const char* image_path = "sample_image.png";
	int color_size = 4;

	// std::wcout << L"----\t current directory : " << current_working_directory() << L"\t----" << std::endl;	// not working correctly now

	std::cout << "----\t this is test program using CUDA code as a library ! \t----" << std::endl;

	std::cout << "----\t process finish ! " << EncodeImage((char*)image_path, color_size) << " [ms] \t----" << std::endl;

	std::cout << "----\t destroyed TPEG device \t\t----" << std::endl;

	return 0;
}
#pragma endregion

int operator_test() {
	printf("-64 >> 3: %d\n", -64 >> 3);
	printf("(short)-2.555555f: %d", (short)-2.555555f);
	printf("(unsigned char)-4: %d\n", (unsigned char)(char)-4);
	printf("(short)(char)-64: %d\n", (short)(unsigned char)(char)-64);

	// 1 1 1 1 1 1 1 1 : 255
	//        &
	// 0 1 1 1 1 1 1 0 : 126
	// ----------------------
	// 0 1 1 1 1 1 1 0 : 126
	printf("(unsigned char)255 & (unsigned char)126 = %d\n", (unsigned char)255 & (unsigned char)126);
	printf("(unsigned char)67 & (unsigned char)126 = %d\n", (unsigned char)67 & (unsigned char)126);

	printf("(unsigned short)-1: %d\n", (unsigned short)-1);
	printf("(short)-1: %d\n", (short)-1);
	printf("(unsigned short)(1 << 16): %d\n", (unsigned short)(1 << 16));
	printf("(unsigned short)-1 & (unsigned short)(1 << 15): %d\n", (unsigned short)-1 & (unsigned short)(1 << 15));
	printf("(unsigned short)-1 == (unsigned short)65535: %d\n", (unsigned short)-1 == (unsigned short)65535);
	printf("(unsigned short)-1 == ((unsigned short)1 & (unsigned short)(1 << 16): %d\n)", (unsigned short)-1 == ((unsigned short)1 & (unsigned short)(1 << 16)));

	return 0;
}

int for_loop_test() {
	for (int i = 0; i < BLOCK_SIZE; i++) {
		if (i % 2 == 0)
			continue;

		printf("i: %d\n", i);
	}
	return 0;
}

#pragma region bit_translate_test
short Float2SignedShort(char sign, float result, float quantization) {
	return (short)(
		(unsigned short)(result * quantization * (1 - sign * 2)) +
		(unsigned short)(sign * (1 << (sizeof(unsigned short) * 8 - 1)))
		);
}

float SignedShort2Float(char sign, short swap, int quantization) {
	return (float)(
		(short)(
			(unsigned short)swap -
			(unsigned short)(sign * (1 << (sizeof(unsigned short) * 8 - 1)))
			) * (1 - sign * 2) * quantization
		);
}

int bit_translate_test(float value) {

	printf("\nquantization forward / invert test ...\n");

	float quantization_forward = (float)1 / (float)16;
	int quantizatoin_invert = 16;

	char sign = value < 0;
	short forward = Float2SignedShort(sign, value, quantization_forward);
	printf("sign: %d\t, value: %f\t, quantization_forward: %f\t, forward: %d\n", sign, value, quantization_forward, forward);

	float invert = SignedShort2Float(sign, forward, quantizatoin_invert);
	printf("sign: %d\t, forward: %d\t, quantization_invert: %d\t, invert: %f\n", sign, forward, quantizatoin_invert, invert);

	short tmp = (short)(value * quantization_forward);
	printf("(short)(value * quantization_forward): %d\t, (value * quantization_forward) * quantizatoin_invert: %d\n", tmp, tmp * quantizatoin_invert);

	printf("\nfinish all process ...\n");

	return 0;
}
#pragma endregion

int main() {
	// return bit_translate_test(-859.0f);
	return tpeg_encode_test();
	// return memory_mollic_test()
	// return cosin_table_create();
	// return invert_quantization_table_create(1);
}
