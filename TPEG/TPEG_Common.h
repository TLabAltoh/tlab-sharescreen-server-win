#pragma once

#define R 0
#define G 1
#define B 2
#define A 3

#define Y 0
#define Cr 1
#define Cb 2

#define DIFF_THRESHOLD 32
#define SAME_VALUE_FLAG 0
#define COUNT 1

/**
*  BLOCK_HEDDER_SIZE
*  ----------------------
*  BLOCK INDEX: 2BYTE
*  PIXEL DATA SIZE PER BLOCK: 3BYTE
*/
#define BLOCK_HEDDER_SIZE 5
#define BLOCK_INDEX_BE 0
#define BLOCK_INDEX_LE 1
#define BLOCK_BIT_SIZE_B 2
#define BLOCK_BIT_SIZE_G 3
#define BLOCK_BIT_SIZE_R 4

#define SRC_COLOR_SIZE 4
#define DST_COLOR_SIZE 3

#define BLOCK_AXIS_SIZE	8
#define BLOCK_AXIS_SIZE_LOG2 3	// LOG2(BLOCK_AXIS_SIZE)

#define BLOCK_SIZE 64	// BLOCK_AXIS_SIZE * BLOCK_AXIS_SIZE
#define BLOCK_SIZE_LOG2 6	// LOG2(BLOCK_SIZE)

#define ENDIAN_SIZE 2	// run_length (6 bit) + level (10 bit) = 16 bit = 2 byte
#define ENDIAN_SIZE_LOG2 1	// LOG2(ENDIAN_SIZE)
