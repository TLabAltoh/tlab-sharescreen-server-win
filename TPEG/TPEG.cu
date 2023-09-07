#pragma once

#include "TPEG.h"
#include "TPEG_Buffer.cuh"
#include "TPEG_Kernels.cuh"

#include "stdio.h"
#include "stdlib.h"
#include "string.h"

#define DEBUG 0
#define DEBUG_1 1

namespace TPEG {

	int WarmingUp(int arg) {
		// Warming up.
		cudaFree(NULL);
		return 0;
	}

	template <class T>
	inline int CreateBuffer(T& buffer, int size, int num, int typeSize) {
		if (cudaMalloc((void**)&buffer, size * typeSize) != cudaSuccess) {
			printf("%d Last error: %s\n", num++, cudaGetErrorString(cudaGetLastError()));
			return 1;
		}
		return 0;
	}

	int CreateBuffer() {
		int num = 0;
		// Encode.
		if (CreateBuffer(_currentFrameBuffer_G, _RGBAFrameBufferSize, num, sizeof(unsigned char)) == 1)return 1;
		if (CreateBuffer(_prevFrameBuffer_G, _RGBFrameBufferSize, num, sizeof(unsigned char)) == 1)return 1;
		if (CreateBuffer(_blockDiffSumBuffer_G, _blockDiffSumBufferSize, num, sizeof(unsigned short)) == 1)return 1;
		if (CreateBuffer(_dctForwardFrameBuffer_G, _DCTFrameBufferSize, num, sizeof(short)) == 1)return 1;
		if (CreateBuffer(_encFrameBuffer_G, _encFrameBufferSize, num, sizeof(char)) == 1)return 1;
		// Decode.
		if (CreateBuffer(_dctInvertFrameBuffer_G, _DCTFrameBufferSize, num, sizeof(short)) == 1)return 1;
		if (CreateBuffer(_decFrameBuffer_G, _RGBAFrameBufferSize, num, sizeof(unsigned char)) == 1)return 1;

		return 0;
	}

	template <class T>
	inline int DestroyBuffer(T& buffer, int num) {
		if (cudaFree((void*)buffer) != cudaSuccess) {
			printf("%d Last error: %s\n", num++, cudaGetErrorString(cudaGetLastError()));
			return 1;
		}
		return 0;
	}

	int DestroyBuffer() {
		int num = 0;
		// Encode.
		if (DestroyBuffer(_currentFrameBuffer_G, num) == 1)return 1;
		if (DestroyBuffer(_prevFrameBuffer_G, num) == 1)return 1;
		if (DestroyBuffer(_blockDiffSumBuffer_G, num) == 1)return 1;
		if (DestroyBuffer(_dctForwardFrameBuffer_G, num) == 1)return 1;
		if (DestroyBuffer(_encFrameBuffer_G, num) == 1)return 1;
		// Decode.
		if (DestroyBuffer(_decFrameBuffer_G, num) == 1)return 1;
		if (DestroyBuffer(_dctInvertFrameBuffer_G, num) == 1)return 1;

		return 0;
	}

	int InitializeDevice(
		int width,
		int height,
		char* encFrameBuffer,
		unsigned char* decFrameBuffer)
	{
		// Setup cording device.

		_width = width;
		_height = height;

		_blockWidth = width >> BLOCK_AXIS_SIZE_LOG2;
		_blockHeight = height >> BLOCK_AXIS_SIZE_LOG2;

		int resolution = _width * _height;
		int blockResolution = _blockWidth * _blockHeight;
		int encBufferBlockUnitSize =
			BLOCK_HEDDER_SIZE +
			(BLOCK_SIZE * ENDIAN_SIZE) *
			DST_COLOR_SIZE;

		_RGBAFrameBufferSize = resolution * ORG_COLOR_SIZE;
		_RGBFrameBufferSize = resolution * DST_COLOR_SIZE;
		_blockDiffSumBufferSize = blockResolution;
		_DCTFrameBufferSize = resolution * DST_COLOR_SIZE;
		_encFrameBufferSize = blockResolution * encBufferBlockUnitSize;

		_encFrameBuffer_C = encFrameBuffer;
		_decFrameBuffer_C = decFrameBuffer;

		printf("strlen(encFrameBuffer): %d\n", strlen((char*)encFrameBuffer));
		printf("strlen(decFrameBuffer): %d\n", strlen((char*)decFrameBuffer));

		if (CreateBuffer() == 1) {
			printf("Error: CreateBuffer\n");
			return 1;
		}

#if DEBUG_1
		// Copy encBuffer's meta buffer to gpu encBuffer

		printf("\n-------------------------------------------------------\n");
		printf("start copy encFrameBuffer's cpu data to gpu memory\n\n");

		// CPU --> GPU
		cudaMemcpy(
			_encFrameBuffer_G,
			encFrameBuffer,
			_encFrameBufferSize * sizeof(char),
			cudaMemcpyHostToDevice
		);
		cudaDeviceSynchronize();

		printf("Last error: %s\n", cudaGetErrorString(cudaGetLastError()));
		printf("process finish\n");
		printf("\n-------------------------------------------------------\n");
#endif

#if 0
		printf("\n-------------------------------------------------------\n");
		printf("set block's index to encFrameBuffer\n\n");

		dim3 GridDim(_blockWidth, _blockHeight, 1);
		dim3 BlockDim1(1, 1, 1);

		SetBlockIdx <<< GridDim, BlockDim1 >>> (
			_encFrameBuffer_G
		);
		cudaDeviceSynchronize();

		printf("Last error: %s\n", cudaGetErrorString(cudaGetLastError()));
		printf("process finish\n");
		printf("\n-------------------------------------------------------\n");
#endif

#if DEBUG_1
		printf("\n-------------------------------------------------------\n");
		printf("start copy encFrameBuffer's gpu data to cpu memory\n\n");

		// CPU --> GPU
		cudaMemcpy(
			encFrameBuffer,
			_encFrameBuffer_G,
			_encFrameBufferSize * sizeof(char),
			cudaMemcpyDeviceToHost
		);
		cudaDeviceSynchronize();

		printf("Last error: %s\n", cudaGetErrorString(cudaGetLastError()));
		printf("process finish\n");
		printf("\n-------------------------------------------------------\n");
#endif

		return 0;
	}

	int DestroyDevice() {
		if (DestroyBuffer() == 1) {
			printf("Error: DestroyBuffer\n");
			return 1;
		}

		return 0;
	}

	void DecFrame(char* encFrameBuffer) {
		dim3 GridDim(_blockWidth, _blockHeight, 1);
		dim3 BlockDim88(BLOCK_AXIS_SIZE, BLOCK_AXIS_SIZE, 1);
		dim3 BlockDim8(BLOCK_AXIS_SIZE, 1, 1);
		dim3 BlockDim3(DST_COLOR_SIZE, 1, 1);
		dim3 BlockDim1(1, 1, 1);

		int num = 0;

#if DEBUG
		printf("start decoding ---------------------------------------\n\n");
#endif

		///////////////////////////////////////////////////////////////////////////////////////////////////////
#if true
#if DEBUG
		printf("kernel%d start ----------------------------------------\n\n", num);
#endif
		// CPU --> GPU
		cudaMemcpy(
			_encFrameBuffer_G,
			encFrameBuffer,
			_encFrameBufferSize * sizeof(char),
			cudaMemcpyHostToDevice
		);
		cudaDeviceSynchronize();
#if DEBUG
		printf("Last error: %s\n", cudaGetErrorString(cudaGetLastError()));
		printf("process finish\n");
		printf("\nkernel%d finished -------------------------------------\n\n", num++);
#endif
#endif
		///////////////////////////////////////////////////////////////////////////////////////////////////////
#if true
#if DEBUG
		printf("kernel%d start ----------------------------------------\n\n", num);
#endif
		EntropyInvert <<< GridDim, BlockDim3 >> > (
			_encFrameBuffer_G,
			_dctInvertFrameBuffer_G
		);
		cudaDeviceSynchronize();
#if DEBUG
		printf("Last error: %s\n", cudaGetErrorString(cudaGetLastError()));
		printf("process finish\n");
		printf("\nkernel%d finished -------------------------------------\n\n", num++);
#endif
#endif
		///////////////////////////////////////////////////////////////////////////////////////////////////////
#if true
#if DEBUG
		printf("kernel%d start ----------------------------------------\n\n", num);
#endif
		DCTInvert <<< GridDim, BlockDim88 >>> (
			_dctInvertFrameBuffer_G,
			_decFrameBuffer_G
		);
		cudaDeviceSynchronize();
#if DEBUG
		printf("Last error: %s\n", cudaGetErrorString(cudaGetLastError()));
		printf("process finish\n");
		printf("\nkernel%d finished -------------------------------------\n\n", num++);
#endif
#endif
		///////////////////////////////////////////////////////////////////////////////////////////////////////
#if true
#if DEBUG
		printf("kernel%d start ----------------------------------------\n\n", num);
#endif
		// GPU --> CPU
		cudaMemcpy(
			_decFrameBuffer_C,
			_decFrameBuffer_G,
			_RGBAFrameBufferSize * sizeof(unsigned char),
			cudaMemcpyDeviceToHost
		);
		cudaDeviceSynchronize();
#if DEBUG
		printf("Last error: %s\n", cudaGetErrorString(cudaGetLastError()));
		printf("process finish\n");
		printf("\nkernel%d finished -------------------------------------\n\n", num++);
#endif
#endif
	}

	void EncFrame(unsigned char* currentFrame) {

		// Thread Divid
		// https://youtu.be/cRY5utouJzQ?t=343
		// https://zukaaax.com/archives/233#:~:text=%E3%83%96%E3%83%AD%E3%83%83%E3%82%AF%E6%95%B0%E3%81%A8%E3%82%B9%E3%83%AC%E3%83%83%E3%83%89%E6%95%B0%E3%81%AF%E4%BB%A5%E4%B8%8B%E3%81%AE%E3%82%88%E3%81%86%E3%81%AB%E3%80%8Cdim3%E3%80%8D%E5%9E%8B%E3%81%A7%E5%AE%9A%E7%BE%A9%E3%81%97%E3%80%81%E3%82%AB%E3%83%BC%E3%83%8D%E3%83%AB%E9%96%A2%E6%95%B0%E3%81%AE%3C%3C%3C%E3%80%80%3E%3E%3E%E3%81%A7%E5%9B%B2%E3%82%8F%E3%82%8C%E3%81%9F%E7%AE%87%E6%89%80%E3%81%AB%E6%8C%87%E5%AE%9A%E3%81%97%E3%81%BE%E3%81%99%E3%80%82
		// https://co-crea.jp/wp-content/uploads/2016/07/File_2.pdf p24.(Unlimited number of threads per axis).
		// Get frame difference.
		dim3 GridDim(_blockWidth, _blockHeight, 1);
		dim3 BlockDim88(BLOCK_AXIS_SIZE, BLOCK_AXIS_SIZE, 1);
		dim3 BlockDim8(BLOCK_AXIS_SIZE, 1, 1);
		dim3 BlockDim3(DST_COLOR_SIZE, 1, 1);
		dim3 BlockDim1(1, 1, 1);

		WarmingUp(NULL);
		int num = 0;
		///////////////////////////////////////////////////////////////////////////////////////////////////////
#if true
#if DEBUG
		printf("kernel%d start ----------------------------------------\n\n", num);
#endif
		cudaMemcpy(
			_currentFrameBuffer_G,
			currentFrame,
			_RGBAFrameBufferSize * sizeof(unsigned char),
			cudaMemcpyHostToDevice
		);
		cudaDeviceSynchronize();
#if DEBUG
		printf("Last error: %s\n", cudaGetErrorString(cudaGetLastError()));
		printf("process finish\n");
		printf("\nkernel%d finished -------------------------------------\n\n", num++);
#endif
#endif
		///////////////////////////////////////////////////////////////////////////////////////////////////////
#if true
#if DEBUG
		printf("kernel%d start ----------------------------------------\n\n", num);
#endif
		// Calc block row's total diff.
		GetDiffSum <<< GridDim, BlockDim8 >>> (
			_currentFrameBuffer_G,
			_prevFrameBuffer_G,
			_blockDiffSumBuffer_G
		);
		cudaDeviceSynchronize();
#if DEBUG
		printf("Last error: %s\n", cudaGetErrorString(cudaGetLastError()));
		printf("process finish\n");
		printf("\nkernel%d finished -------------------------------------\n\n", num++);
#endif
#endif
		///////////////////////////////////////////////////////////////////////////////////////////////////////
#if true
#if DEBUG
		printf("kernel%d start ----------------------------------------\n\n", num);
#endif
		DCTForward <<< GridDim, BlockDim88 >>> (
			_prevFrameBuffer_G,
			_blockDiffSumBuffer_G,
			_dctForwardFrameBuffer_G
		);
		cudaDeviceSynchronize();
#if DEBUG
		printf("Last error: %s\n", cudaGetErrorString(cudaGetLastError()));
		printf("process finish\n");
		printf("\nkernel%d finished -------------------------------------\n\n", num++);
#endif
#endif
		///////////////////////////////////////////////////////////////////////////////////////////////////////
#if true
#if DEBUG
		printf("kernel%d start ----------------------------------------\n\n", num);
#endif
		EntropyForward <<< GridDim, BlockDim3 >>> (
			_dctForwardFrameBuffer_G,
			_encFrameBuffer_G,
			_blockDiffSumBuffer_G
		);
		cudaDeviceSynchronize();
#if DEBUG
		printf("Last error: %s\n", cudaGetErrorString(cudaGetLastError()));
		printf("process finish\n");
		printf("\nkernel%d finished -------------------------------------\n\n", num++);
#endif
#endif
		///////////////////////////////////////////////////////////////////////////////////////////////////////
#if true
#if DEBUG
		printf("kernel%d start ----------------------------------------\n\n", num);
#endif
		// GPU --> CPU
		cudaMemcpy(
			_encFrameBuffer_C,
			_encFrameBuffer_G,
			_encFrameBufferSize * sizeof(char),
			cudaMemcpyDeviceToHost
		);
		cudaDeviceSynchronize();
#if DEBUG
		printf("Last error: %s\n", cudaGetErrorString(cudaGetLastError()));
		printf("process finish\n");
		printf("\nkernel%d finished -------------------------------------\n\n", num++);
#endif
#endif
		///////////////////////////////////////////////////////////////////////////////////////////////////////
#if DEBUG
		printf("\nstart image decoding ---------------------------------\n\n");
		DecFrame(_encFrameBuffer_C);
		printf("\nfinish image decoding --------------------------------\n\n");
#endif
	}
}