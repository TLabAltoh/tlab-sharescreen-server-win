#pragma once

#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "iostream"
#include "fstream"
#include "sstream"
#include "time.h"
#include "atlimage.h"
#include "Windows.h"

#define R 0
#define G 1
#define B 2
#define A 3

#define BLOCK_AXIS_SIZE 8
#define BLOCK_AXIS_SIZE_LOG2 3

#define BLOCK_SIZE 64
#define BLOCK_SIZE_LOG2 6

#define BLOCK_HEDDER_SIZE 5
#define BLOCK_IDX_UPPER_IDX 0
#define BLOCK_IDX_LOWER_IDX 1
#define BLOCK_BIT_SIZE_IDX_B 2
#define BLOCK_BIT_SIZE_IDX_G 3
#define BLOCK_BIT_SIZE_IDX_R 4
#define BLOCK_BIT_SIZE_IDX_OFFSET 2

#define SRC_COLOR_SIZE 4
#define DST_COLOR_SIZE 3

#define ENDIAN_SIZE 2
#define BIG_ENDIAN_IDX 0
#define LITTLE_ENDIAN_IDX 1
