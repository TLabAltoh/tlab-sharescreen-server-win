#pragma once

#include "TLabCUDA_Common.h"

__device__ int Pow2(int x) {
	return x * x;
}