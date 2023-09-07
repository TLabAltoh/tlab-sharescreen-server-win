#pragma once

#include "TLabWindows.h"
#include "TPEG.h"

/*******************************************************
    this is library that encode captured texture.
********************************************************/

// BGR_OR_YCrCb
#define DST_COLOR_SIZE 3
// BGRA
#define SRC_COLOR_SIZE 4
// PIXEL_VALUE_AND_RUN_ENDIAN_SIZE
#define ENDIAN_SIZE 2
#define ENDIAN_SIZE_LOG2 1

#define NO_NEED_TO_ENCODE 0

////////////////////////////////////////////////////////////
// BLOCK_AXIS_SIZE UTIL
//
#define BLOCK_AXIS_SIZE 8
#define BLOCX_AXIS_SIZE_LOG2 3

////////////////////////////////////////////////////////////
// PACKET_HEDDER:
// ---------------------------------------------------------
// PACKET_IDX_UPPERBIT(1BYTE)
// PACKET_IDX_UPPERBIT(1BYTE)
// LAST_FRAME'S_ENDIDX_UPPERBIT(1BYTE)
// LAST_FRAME'S_ENDIDX_LOWERBIT(1BYTE)
// FRAME_OFFSET(1BYTE)
// FRAG_THIS_PACKET_IS_FRAME'S_LAST_PACKET(1BYTE)
// FRAG_THIS_PACKET_IS FIX_PACKET(1BYTE)
//
#define PACKET_HEDDER_SIZE 7
#define PACKET_IDX_UPPER_IDX 1
#define PACKET_IDX_LOWER_IDX 0
#define LAST_FRAME_END_IDX_UPPER_IDX 3
#define LAST_FRAME_END_IDX_LOWER_IDX 2
#define FRAME_OFFSET_IDX 4
#define IS_THIS_PACKETEND_IDX 5
#define IS_THIS_FIX_PACKET_IDX 6

#define THIS_PACEKT_IS_NOT_FRAMES_LAST 0
#define THIS_PACEKT_IS_FRAMES_LAST 1

#define THIS_PACKET_IS_NOT_FOR_FIX 0
#define THIS_PACKET_IS_FOR_FIX 1

//////////////////////////////////////////
// BLOCK_HEDDER:
// --------------------------------------
// BLOCK_INDEX(2BYTE)
// BIT_COLOR_SIZE_Y(1BYTE)
// BIT_COLOR_SIZE_Cr(1BYTE)
// BIT_COLOR_SIZE_Cb(1BYTE)
// --------------------------------------
// BLOCK_HEDDER'S_BIT_SIZE_OFFSET
//
#define BLOCK_HEDDER_SIZE 5
#define BLOCK_IDX_UPPER_IDX 0
#define BLOCK_IDX_LOWER_IDX 1
#define Y_BIT_SIZE_IDX 2
#define Cr_BIT_SIZE_IDX 3
#define Cb_BIT_SIZE_IDX 4
#define BLOCK_BIT_SIZE_IDX 2

int CalcEncBufferSize(int width, int height) {
    int blockWidth = width / BLOCK_AXIS_SIZE;
    int blockHeight = height / BLOCK_AXIS_SIZE;
    int blockUnitSize =
        BLOCK_HEDDER_SIZE +
        BLOCK_AXIS_SIZE *
        BLOCK_AXIS_SIZE *
        ENDIAN_SIZE *
        DST_COLOR_SIZE;

    return blockWidth * blockHeight * blockUnitSize;
}

int DecoderInitialize(
    int width,
    int height,
    char* encBuffer,
    unsigned char* decBuffer)
{
    if (TPEG::InitializeDevice(width, height, encBuffer, decBuffer) == 1)return 1;
    return 0;
}

int DecoderDestroy() {
    if (TPEG::DestroyDevice() == 1)return 1;
    return 0;
}

void EncodeFrame(unsigned char* frameBuffer) {
    TPEG::EncFrame(frameBuffer);
}
