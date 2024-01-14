#pragma once

#include "windows_common.h"
#include "TPEG.h"

/**
*  Packet Hedder
*
*/
#define PACKET_HEDDER_SIZE 7
#define PACKET_INDEX_BE 1
#define PACKET_INDEX_LE 0
#define LAST_FRAME_FINAL_INDEX_BE 3
#define LAST_FRAME_FINAL_INDEX_LE 2
#define FRAME_OFFSET_INDEX 4
#define IS_THIS_PACKETEND 5
#define IS_THIS_FIX_PACKET 6

int EncodedFrameBufferSize(int width, int height) {
    return (width / BLOCK_AXIS_SIZE) * (height / BLOCK_AXIS_SIZE) * (BLOCK_HEDDER_SIZE + BLOCK_AXIS_SIZE * BLOCK_AXIS_SIZE * ENDIAN_SIZE * DST_COLOR_SIZE);
}

int DecoderInitialize(int width, int height, char* encoded_frame_buffer, unsigned char* decoded_frame_buffer)
{
    if (TPEG::InitializeDevice(width, height, encoded_frame_buffer, decoded_frame_buffer) == 1) {
        return 1;
    }

    return 0;
}

int DecoderDestroy() {
    if (TPEG::DestroyDevice() == 1) {
        return 1;
    }

    return 0;
}

void EncodeFrame(unsigned char* frame_buffer) {
    TPEG::EncodeFrame(frame_buffer);
}
