#include "stdio.h"
#include "stdlib.h"
#include "iostream"
#include "TPEG.h"

int main() {

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