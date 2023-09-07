#pragma once

#include "Common.h"
#include "TPEG.h"

CImage orgFrame;
int width;
int height;
int pitch;
int res;
int blockNum;
int srcFrameBufferSize;
int encFrameBufferSize;
unsigned short* blockDiffSumBuffer;
unsigned char* orgFrameBuffer;
unsigned char* srcFrameBuffer;
unsigned char* decFrameBuffer;
char* encFrameBuffer;

int FileCheck(char* path, char* extension) {
	int pathSize = strlen(path);
	int extensionSize = strlen(extension);
	int size = pathSize - extensionSize;

	printf("\npath name: %s\n", path);
	printf("pathSizde: %d, extensionSize: %d, size: %d\n", pathSize, extensionSize, size);

	if (size < 1 || strcmp(path + size, extension) != 0) {
		printf("FileCheck: error\n");
		printf("path value: %.*s\n", extensionSize, path + size);
		return 1;
	}

	printf("---------------------------------------\n\n");
	return 0;
}

wchar_t* TLabUTF8To16(char* charStr) {
	int size = MultiByteToWideChar(CP_UTF8, 0, charStr, -1, nullptr, 0);
	wchar_t* utf16str = new wchar_t[size];
	MultiByteToWideChar(CP_UTF8, 0, charStr, -1, utf16str, size);
	return utf16str;
}

char* InitializeEncFrameBuffer(int size) {
	char* encFrameBufferTmp = new char[size];

	for (int i = 0; i < size; i++) encFrameBufferTmp[i] = (char)0;

	return encFrameBufferTmp;
}

unsigned char* InitializeDecFrameBuffer(int size) {
	unsigned char* decFrameBufferTmp = new unsigned char[size];

	for (int i = 0; i < size; i++) decFrameBufferTmp[i] = (char)0;

	return decFrameBufferTmp;
}

unsigned short* InitializeBlockDiffSumBuffer(int size) {
	unsigned short* blockDiffSumBufferTmp = new unsigned short[size];

	for (int i = 0; i < size; i++) blockDiffSumBufferTmp[i] = (unsigned short)0;

	return blockDiffSumBufferTmp;
}

int BitMapEncodeTest(char* path, int colorSize) {
	if (FileCheck(path, (char*)".bmp") == 1) {
		return 1;
	}

	wchar_t* pathUtf16 = TLabUTF8To16(path);

	orgFrame.Load(pathUtf16);
	printf(
		"orgFrame.Save(testFile.bmp): %s\n",
		orgFrame.Save(L"testFile.bmp") == S_OK ? "succeeded" : "failed"
	);

	printf("Bitmap loaded.\n\n");

	width = orgFrame.GetWidth();
	height = orgFrame.GetHeight();
	pitch = orgFrame.GetPitch();
	res = width * height;

	printf("Bitmap info ---------------------------\n");
	printf("Widht\t:\t%d\n", width);
	printf("Height\t:\t%d\n", height);
	printf("Pitch\t:\t%d\n", pitch);
	printf("Resolutin\t:\t %d\n", res);
	printf("Width * Height\t:\t %d\n", width * height);
	printf("Color Size:\t%d\t\n", colorSize);
	printf("---------------------------------------\n\n");

	blockNum = (width >> BLOCK_AXIS_SIZE_LOG2) * (height >> BLOCK_AXIS_SIZE_LOG2);

	srcFrameBufferSize = width * height * SRC_COLOR_SIZE;
	encFrameBufferSize =
		blockNum * BLOCK_HEDDER_SIZE +
		width * height * ENDIAN_SIZE * DST_COLOR_SIZE;

	blockDiffSumBuffer = InitializeBlockDiffSumBuffer(blockNum);
	srcFrameBuffer = InitializeDecFrameBuffer(srcFrameBufferSize);
	decFrameBuffer = InitializeDecFrameBuffer(srcFrameBufferSize);
	encFrameBuffer = InitializeEncFrameBuffer(encFrameBufferSize);


	if (TPEG::InitializeDevice(
		width, height,
		encFrameBuffer,
		decFrameBuffer) == 1) {
		printf("Error: TPEG::InitializeDevice\n");
		return 1;
	}

	printf("TPEG device initialized\n");

	int srcIdx = 0;
	for (int i = 0; i < height; i++) {
		orgFrameBuffer = (unsigned char*)orgFrame.GetBits() + (size_t)i * pitch;
		for (int j = 0; j < width; j++) {
			srcFrameBuffer[srcIdx + R] = orgFrameBuffer[R];
			srcFrameBuffer[srcIdx + G] = orgFrameBuffer[G];
			srcFrameBuffer[srcIdx + B] = orgFrameBuffer[B];
			srcFrameBuffer[srcIdx + A] = orgFrameBuffer[A];
			srcIdx += SRC_COLOR_SIZE;
			orgFrameBuffer += colorSize;
		}
	}

	int loopNum;
	printf("Bitmap copyed.\n\n");

#if true
	// copy decoded frame to orgFrameBuffer.
	srcIdx = 0;
	for (int i = 0; i < height; i++) {
		orgFrameBuffer = (unsigned char*)orgFrame.GetBits() + (size_t)i * pitch;
		for (int j = 0; j < width; j++) {
			orgFrameBuffer[R] = srcFrameBuffer[srcIdx + R];
			orgFrameBuffer[G] = srcFrameBuffer[srcIdx + G];
			orgFrameBuffer[B] = srcFrameBuffer[srcIdx + B];
			orgFrameBuffer[A] = srcFrameBuffer[srcIdx + A];
			srcIdx += SRC_COLOR_SIZE;
			orgFrameBuffer += colorSize;
		}
	}

	printf(
		"orgFrame.Save(testFile1.bmp): %s\n",
		orgFrame.Save(L"testFile1.bmp") == S_OK ? "succeeded" : "failed"
	);
#endif

#if false
	loopNum = BLOCK_SIZE;
	printf("Log encFrameBuffer first one block -------------------\n\n");
	for (int i = 0; i < loopNum; i++) {
		printf("srcFrameBuffer[%d]\t: %d\n", i, srcFrameBuffer[i]);
	}
	printf("------------------------------------------------------\n\n");
#endif

#if false
	loopNum = BLOCK_SIZE * ENDIAN_SIZE * DST_COLOR_SIZE;
	printf("Log before encFrameBuffer first one block ------------\n\n");
	for (int i = 0; i < loopNum; i++) {
		printf("before: encFrameBuffer[%d]\t: %d\n", i, encFrameBuffer[i]);
	}
	printf("------------------------------------------------------\n\n");
#endif
	clock_t start = clock();
	// Encode frame and decode frame.
	TPEG::EncFrame(srcFrameBuffer);
	clock_t end = clock();

	printf("Bitmap encoded.\n\n");

	// Destoroy tpeg device.
	TPEG::DestroyDevice();

#if false
	loopNum = BLOCK_SIZE * ENDIAN_SIZE * DST_COLOR_SIZE;
	printf("Log after encFrameBuffer first one block -------------\n\n");
	for (int i = 0; i < loopNum; i++) {
		printf("after: encFrameBuffer[%d]\t: %d\n", i, encFrameBuffer[i]);
	}
	printf("------------------------------------------------------\n\n");
#endif

#if false
	printf("Log decFrameBuffer first one block -------------------\n\n");
	loopNum = BLOCK_SIZE;
	for (int i = 0; i < loopNum; i++) {
		printf("decFrameBuffer[%d]\t: %d\n", i, decFrameBuffer[i]);
	}
	printf("------------------------------------------------------\n\n");
#endif

#if true
	printf("Log decFrameBuffer size ------------------------------\n\n");
	loopNum = width / BLOCK_AXIS_SIZE * height / BLOCK_AXIS_SIZE;
	int sum = 0;
	for (int i = 0; i < loopNum; i++) {
		char* encFrameBufferPt =
			encFrameBuffer + (size_t)i * (BLOCK_HEDDER_SIZE + BLOCK_SIZE * DST_COLOR_SIZE * ENDIAN_SIZE);

		int tmp = encFrameBufferPt[BLOCK_BIT_SIZE_IDX_OFFSET + R];
		if (tmp > 64)printf("inviled value in R: %d\n", tmp);
		sum += tmp * ENDIAN_SIZE;

		tmp = encFrameBufferPt[BLOCK_BIT_SIZE_IDX_OFFSET + G];
		if (tmp > 64)printf("inviled value in G: %d\n", tmp);
		sum += tmp * ENDIAN_SIZE;

		tmp = encFrameBufferPt[BLOCK_BIT_SIZE_IDX_OFFSET + B];
		if (tmp > 64)printf("inviled value in B: %d\n", tmp);
		sum += tmp * ENDIAN_SIZE;
	}
	float beforSize = (size_t)width * height * DST_COLOR_SIZE * pow(10, -3);
	float afterSize = (size_t)sum / 2 * ENDIAN_SIZE * pow(10, -3);
	printf("orgImage size: %f\n", beforSize);
	printf("encBuffer, decoded size: %f\n", afterSize);
	printf("compression rate: %f\n", (size_t)afterSize / (size_t)beforSize * 100);
	printf("------------------------------------------------------\n\n");
#endif

	// copy decoded frame to orgFrameBuffer.
	srcIdx = 0;
	for (int i = 0; i < height; i++) {
		orgFrameBuffer = (unsigned char*)orgFrame.GetBits() + (size_t)i * pitch;
		for (int j = 0; j < width; j++) {
			orgFrameBuffer[R] = decFrameBuffer[srcIdx + R];
			orgFrameBuffer[G] = decFrameBuffer[srcIdx + G];
			orgFrameBuffer[B] = decFrameBuffer[srcIdx + B];
			orgFrameBuffer[A] = decFrameBuffer[srcIdx + A];
			srcIdx += SRC_COLOR_SIZE;
			orgFrameBuffer += colorSize;
		}
	}

	printf(
		"orgFrame.Save(result.bmp): %s\n",
		orgFrame.Save(L"result.bmp") == S_OK ? "succeeded" : "failed"
	);

	orgFrame.ReleaseDC();

	printf("Finish all process !\n");

	return end - start;
}
