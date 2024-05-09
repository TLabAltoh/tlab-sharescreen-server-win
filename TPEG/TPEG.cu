#pragma once

#include "TPEG.h"
#include "kernels.cuh"
#include "tpeg_cuda.h"
#include "tpeg_common.h"
#include "tpeg_buffer.cuh"

#include "stdio.h"
#include "stdlib.h"
#include "string.h"

#define DEBUG 0

namespace TPEG {

	int WarmingUp(int arg) {
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
		if (CreateBuffer(_current_frame_buffer_gpu, _rgba_frame_buffer_size, num, sizeof(unsigned char)) == 1)return 1;
		if (CreateBuffer(_prev_frame_buffer_gpu, _rgb_frame_buffer_size, num, sizeof(unsigned char)) == 1)return 1;
		if (CreateBuffer(_block_diff_sum_buffer_gpu, _block_diff_sum_buffer_size, num, sizeof(unsigned short)) == 1)return 1;
		if (CreateBuffer(_dct_result_frame_buffer_gpu, _dct_frame_buffer_size, num, sizeof(short)) == 1)return 1;
		if (CreateBuffer(_encoded_frame_buffer_gpu, _encoded_frame_buffer_size, num, sizeof(char)) == 1)return 1;
		// Decode.
		if (CreateBuffer(_idct_result_frame_buffer_gpu, _dct_frame_buffer_size, num, sizeof(short)) == 1)return 1;
		if (CreateBuffer(_decoded_frame_buffer_gpu, _rgba_frame_buffer_size, num, sizeof(unsigned char)) == 1)return 1;

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
		// Encode
		if (DestroyBuffer(_current_frame_buffer_gpu, num) == 1) return 1;
		if (DestroyBuffer(_prev_frame_buffer_gpu, num) == 1) return 1;
		if (DestroyBuffer(_block_diff_sum_buffer_gpu, num) == 1) return 1;
		if (DestroyBuffer(_dct_result_frame_buffer_gpu, num) == 1) return 1;
		if (DestroyBuffer(_encoded_frame_buffer_gpu, num) == 1) return 1;
		// Decode
		if (DestroyBuffer(_decoded_frame_buffer_gpu, num) == 1) return 1;
		if (DestroyBuffer(_idct_result_frame_buffer_gpu, num) == 1) return 1;

		return 0;
	}

	int InitializeDevice(int width, int height, char* encoded_frame_buffer, unsigned char* decoded_frame_buffer)
	{
		_width = width;
		_height = height;

		_blockWidth = width >> BLOCK_AXIS_SIZE_LOG2;
		_blockHeight = height >> BLOCK_AXIS_SIZE_LOG2;

		int frame_resolution = _width * _height;
		int block_resolution = _blockWidth * _blockHeight;
		int encoded_buffer_block_size = BLOCK_HEDDER_SIZE + (BLOCK_SIZE * ENDIAN_SIZE) * DST_COLOR_SIZE;

		_rgba_frame_buffer_size = frame_resolution * SRC_COLOR_SIZE;
		_rgb_frame_buffer_size = frame_resolution * DST_COLOR_SIZE;
		_block_diff_sum_buffer_size = block_resolution;
		_dct_frame_buffer_size = frame_resolution * DST_COLOR_SIZE;
		_encoded_frame_buffer_size = block_resolution * encoded_buffer_block_size;

		_encoded_frame_buffer_cpu = encoded_frame_buffer;
		_decoded_frame_buffer_cpu = decoded_frame_buffer;

		printf("strlen(encoded_frame_buffer): %d\n", strlen((char*)encoded_frame_buffer));
		printf("strlen(decoded_frame_buffer): %d\n", strlen((char*)decoded_frame_buffer));

		if (CreateBuffer() == 1) {
			printf("Error: CreateBuffer\n");
			return 1;
		}

		// copy encoded_frame_buffer's meta buffer to gpu encoded_frame_buffer
#if DEBUG
		printf("\n-------------------------------------------------------\n");
		printf("start copy encoded_frame_buffer's cpu data to gpu memory\n\n");
#endif
		// CPU --> GPU
		cudaMemcpy(
			_encoded_frame_buffer_gpu,
			encoded_frame_buffer,
			_encoded_frame_buffer_size * sizeof(char),
			cudaMemcpyHostToDevice
		);
		cudaDeviceSynchronize();

#if DEBUG
		printf("Last error: %s\n", cudaGetErrorString(cudaGetLastError()));
		printf("process finish\n");
		printf("\n-------------------------------------------------------\n");

		printf("\n-------------------------------------------------------\n");
		printf("set block's index to encoded_frame_buffer\n\n");
#endif
		dim3 GridDim(_blockWidth, _blockHeight, 1);
		dim3 BlockDim1(1, 1, 1);

		SetBlockIdx << < GridDim, BlockDim1 >> > (
			_encoded_frame_buffer_gpu);

		cudaDeviceSynchronize();

#if DEBUG
		printf("Last error: %s\n", cudaGetErrorString(cudaGetLastError()));
		printf("process finish\n");
		printf("\n-------------------------------------------------------\n");

		printf("\n-------------------------------------------------------\n");
		printf("start copy encoded_frame_buffer's gpu data to cpu memory\n\n");
#endif
		cudaMemcpy(	// CPU --> GPU
			encoded_frame_buffer,
			_encoded_frame_buffer_gpu,
			_encoded_frame_buffer_size * sizeof(char),
			cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();

#if DEBUG
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

	void DecodeFrame(char* encoded_frame_buffer) {
		dim3 GridDim(_blockWidth, _blockHeight, 1);
		dim3 BlockDim8(BLOCK_AXIS_SIZE, 1, 1);
		dim3 BlockDim3(DST_COLOR_SIZE, 1, 1);
		dim3 BlockDim1(1, 1, 1);

		int num = 0;

#if DEBUG
		printf("start decoding ---------------------------------------\n\n");
#endif

		///////////////////////////////////////////////////////////////////////////////////////////////////////

#if DEBUG
		printf("kernel%d start ----------------------------------------\n\n", num);
#endif

		// CPU --> GPU
		cudaMemcpy(
			_encoded_frame_buffer_gpu,
			encoded_frame_buffer,
			_encoded_frame_buffer_size * sizeof(char),
			cudaMemcpyHostToDevice);

		cudaDeviceSynchronize();

#if DEBUG
		printf("Last error: %s\n", cudaGetErrorString(cudaGetLastError()));
		printf("process finish\n");
		printf("\nkernel%d finished -------------------------------------\n\n", num++);
#endif

		///////////////////////////////////////////////////////////////////////////////////////////////////////

#if DEBUG
		printf("kernel%d start ----------------------------------------\n\n", num);
#endif

		EntropyInvert << < GridDim, BlockDim3 >> > (
			_encoded_frame_buffer_gpu,
			_idct_result_frame_buffer_gpu);

		cudaDeviceSynchronize();

#if DEBUG
		printf("Last error: %s\n", cudaGetErrorString(cudaGetLastError()));
		printf("process finish\n");
		printf("\nkernel%d finished -------------------------------------\n\n", num++);
#endif

		///////////////////////////////////////////////////////////////////////////////////////////////////////

#if DEBUG
		printf("kernel%d start ----------------------------------------\n\n", num);
#endif

		CUD8x8IDCT_RGBFrame << < GridDim, BlockDim1 >> > (
			_idct_result_frame_buffer_gpu,
			_decoded_frame_buffer_gpu);

		cudaDeviceSynchronize();

#if DEBUG
		printf("Last error: %s\n", cudaGetErrorString(cudaGetLastError()));
		printf("process finish\n");
		printf("\nkernel%d finished -------------------------------------\n\n", num++);
#endif

		///////////////////////////////////////////////////////////////////////////////////////////////////////

#if DEBUG
		printf("kernel%d start ----------------------------------------\n\n", num);
#endif

		// GPU --> CPU
		cudaMemcpy(
			_decoded_frame_buffer_cpu,
			_decoded_frame_buffer_gpu,
			_rgba_frame_buffer_size * sizeof(unsigned char),
			cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();

#if DEBUG
		printf("Last error: %s\n", cudaGetErrorString(cudaGetLastError()));
		printf("process finish\n");
		printf("\nkernel%d finished -------------------------------------\n\n", num++);
#endif
	}

	void EncodeFrame(unsigned char* current_frame) {

		// Thread Divid
		// https://youtu.be/cRY5utouJzQ?t=343
		// https://zukaaax.com/archives/233#:~:text=%E3%83%96%E3%83%AD%E3%83%83%E3%82%AF%E6%95%B0%E3%81%A8%E3%82%B9%E3%83%AC%E3%83%83%E3%83%89%E6%95%B0%E3%81%AF%E4%BB%A5%E4%B8%8B%E3%81%AE%E3%82%88%E3%81%86%E3%81%AB%E3%80%8Cdim3%E3%80%8D%E5%9E%8B%E3%81%A7%E5%AE%9A%E7%BE%A9%E3%81%97%E3%80%81%E3%82%AB%E3%83%BC%E3%83%8D%E3%83%AB%E9%96%A2%E6%95%B0%E3%81%AE%3C%3C%3C%E3%80%80%3E%3E%3E%E3%81%A7%E5%9B%B2%E3%82%8F%E3%82%8C%E3%81%9F%E7%AE%87%E6%89%80%E3%81%AB%E6%8C%87%E5%AE%9A%E3%81%97%E3%81%BE%E3%81%99%E3%80%82
		// https://co-crea.jp/wp-content/uploads/2016/07/File_2.pdf p24.(Unlimited number of threads per axis).
		// Get frame difference.
		dim3 GridDim(_blockWidth, _blockHeight, 1);
		dim3 BlockDim8(BLOCK_AXIS_SIZE, 1, 1);
		dim3 BlockDim3(DST_COLOR_SIZE, 1, 1);
		dim3 BlockDim1(1, 1, 1);

		WarmingUp(NULL);
		int num = 0;

		///////////////////////////////////////////////////////////////////////////////////////////////////////

#if DEBUG
		printf("kernel%d start ----------------------------------------\n\n", num);
#endif

		cudaMemcpy(
			_current_frame_buffer_gpu,
			current_frame,
			_rgba_frame_buffer_size * sizeof(unsigned char),
			cudaMemcpyHostToDevice);

		cudaDeviceSynchronize();

#if DEBUG
		printf("Last error: %s\n", cudaGetErrorString(cudaGetLastError()));
		printf("process finish\n");
		printf("\nkernel%d finished -------------------------------------\n\n", num++);
#endif

		///////////////////////////////////////////////////////////////////////////////////////////////////////

#if DEBUG
		printf("kernel%d start ----------------------------------------\n\n", num);
#endif

		GetDiffSum << < GridDim, BlockDim8 >> > (	// Calc block row's total diff
			_current_frame_buffer_gpu,
			_prev_frame_buffer_gpu,
			_block_diff_sum_buffer_gpu);

		cudaDeviceSynchronize();

#if DEBUG
		printf("Last error: %s\n", cudaGetErrorString(cudaGetLastError()));
		printf("process finish\n");
		printf("\nkernel%d finished -------------------------------------\n\n", num++);
#endif

		///////////////////////////////////////////////////////////////////////////////////////////////////////

#if DEBUG
		printf("kernel%d start ----------------------------------------\n\n", num);
#endif

		CUD8x8DCT_RGBFrame << < GridDim, BlockDim1 >> > (
			_prev_frame_buffer_gpu,
			_dct_result_frame_buffer_gpu,
			_block_diff_sum_buffer_gpu);

		cudaDeviceSynchronize();

#if DEBUG
		printf("Last error: %s\n", cudaGetErrorString(cudaGetLastError()));
		printf("process finish\n");
		printf("\nkernel%d finished -------------------------------------\n\n", num++);
#endif

		///////////////////////////////////////////////////////////////////////////////////////////////////////

#if DEBUG
		printf("kernel%d start ----------------------------------------\n\n", num);
#endif

		EntropyForward << < GridDim, BlockDim3 >> > (
			_dct_result_frame_buffer_gpu,
			_encoded_frame_buffer_gpu,
			_block_diff_sum_buffer_gpu);

		cudaDeviceSynchronize();

#if DEBUG
		printf("Last error: %s\n", cudaGetErrorString(cudaGetLastError()));
		printf("process finish\n");
		printf("\nkernel%d finished -------------------------------------\n\n", num++);
#endif

		///////////////////////////////////////////////////////////////////////////////////////////////////////

#if DEBUG
		printf("kernel%d start ----------------------------------------\n\n", num);
#endif

		// GPU --> CPU
		cudaMemcpy(
			_encoded_frame_buffer_cpu,
			_encoded_frame_buffer_gpu,
			_encoded_frame_buffer_size * sizeof(char),
			cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();

#if DEBUG
		printf("Last error: %s\n", cudaGetErrorString(cudaGetLastError()));
		printf("process finish\n");
		printf("\nkernel%d finished -------------------------------------\n\n", num++);
#endif

		///////////////////////////////////////////////////////////////////////////////////////////////////////

#if DEBUG
		printf("\nstart image decoding ---------------------------------\n\n");
		DecodeFrame(_encoded_frame_buffer_cpu);
		printf("\nfinish image decoding --------------------------------\n\n");
#endif
	}
}