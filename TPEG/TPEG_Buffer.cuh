#pragma once

#include "TPEG_Common.h"

//////////////////////////////////////////////////////////////////////////////
// RGBA_FRAME_BUFFER
//

/// <summary>
/// current frame buffer on gpu.
/// </summary>
unsigned char* _currentFrameBuffer_G;

/// <summary>
/// Buffer to record decoding result.
/// GPU
/// </summary>
unsigned char* _decFrameBuffer_G;

/// <summary>
/// RGBA frame buffer size.
/// </summary>
int _RGBAFrameBufferSize;

//////////////////////////////////////////////////////////////////////////////
// RGB_FRAME_BUFFER
//

/// <summary>
/// prev frame buffer on gpu.
/// </summary>
unsigned char* _prevFrameBuffer_G;

/// <summary>
/// RGB frame buffer size.
/// </summary>
int _RGBFrameBufferSize;

//////////////////////////////////////////////////////////////////////////////
// BLOCK_DIFF_SUM_BUFFER
//

/// <summary>
/// Total per block buffer on gpu.
/// </summary>
unsigned short* _blockDiffSumBuffer_G;

/// <summary>
/// _blockDiffSumBuffer's buffer size.
/// </summary>
int _blockDiffSumBufferSize;

//////////////////////////////////////////////////////////////////////////////
// DCT_FRAME_BUFFER
//

/// <summary>
/// A buffer that records the result of the
/// dct transform for each block.
/// </summary>
short* _dctForwardFrameBuffer_G;

/// <summary>
/// Buffer for inverse DCT transform.
/// </summary>
short* _dctInvertFrameBuffer_G;

/// <summary>
/// DCT frame buffer size.
/// </summary>
int _DCTFrameBufferSize;

//////////////////////////////////////////////////////////////////////////////
// ENC_FRAME_BUFFER
//

/// <summary>
/// Buffer to record encoding result.
/// GPU
/// </summary>
char* _encFrameBuffer_G;

/// <summary>
/// Encoded frame buffer size.
/// </summary>
int _encFrameBufferSize;

//////////////////////////////////////////////////////////////////////////////
// ENC_AND_DEC_FRAME_BUFFER_HOST
//

/// <summary>
/// Buffer to record encoding result.
/// CPU
/// </summary>
char* _encFrameBuffer_C;

/// <summary>
/// Buffer to record decoding result.
/// CPU
/// </summary>
unsigned char* _decFrameBuffer_C;

//////////////////////////////////////////////////////////////////////////////
// RESOLUTION
//

/// <summary>
/// texture width.
/// </summary>
int _width;

/// <summary>
/// texture height.
/// </summary>
int _height;

/// <summary>
/// The width of the resolution
/// where the texture is divided into blocks.
/// </summary>
int _blockWidth;

/// <summary>
/// The height of the resolution
/// where the texture is divided into blocks.
/// </summary>
int _blockHeight;