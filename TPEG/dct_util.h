#pragma once

__constant__ float DCTv8matrix[] = {
	0.3535533905932738f,  0.4903926402016152f,  0.4619397662556434f,  0.4157348061512726f,  0.3535533905932738f,  0.2777851165098011f,  0.1913417161825449f,  0.0975451610080642f,
	0.3535533905932738f,  0.4157348061512726f,  0.1913417161825449f, -0.0975451610080641f, -0.3535533905932737f, -0.4903926402016152f, -0.4619397662556434f, -0.2777851165098011f,
	0.3535533905932738f,  0.2777851165098011f, -0.1913417161825449f, -0.4903926402016152f, -0.3535533905932738f,  0.0975451610080642f,  0.4619397662556433f,  0.4157348061512727f,
	0.3535533905932738f,  0.0975451610080642f, -0.4619397662556434f, -0.2777851165098011f,  0.3535533905932737f,  0.4157348061512727f, -0.1913417161825450f, -0.4903926402016153f,
	0.3535533905932738f, -0.0975451610080641f, -0.4619397662556434f,  0.2777851165098009f,  0.3535533905932738f, -0.4157348061512726f, -0.1913417161825453f,  0.4903926402016152f,
	0.3535533905932738f, -0.2777851165098010f, -0.1913417161825452f,  0.4903926402016153f, -0.3535533905932733f, -0.0975451610080649f,  0.4619397662556437f, -0.4157348061512720f,
	0.3535533905932738f, -0.4157348061512727f,  0.1913417161825450f,  0.0975451610080640f, -0.3535533905932736f,  0.4903926402016152f, -0.4619397662556435f,  0.2777851165098022f,
	0.3535533905932738f, -0.4903926402016152f,  0.4619397662556433f, -0.4157348061512721f,  0.3535533905932733f, -0.2777851165098008f,  0.1913417161825431f, -0.0975451610080625f
};


__constant__ int ZigZagIndexForward[] = {
	 0,  1,  5,  6, 14, 15, 27, 28,
	 2,  4,  7, 13, 16, 26, 29, 42,
	 3,  8, 12, 17, 25, 30, 41, 43,
	 9, 11, 18, 24, 31, 40, 44, 53,
	10, 19, 23, 32, 39, 45, 52, 54,
	20, 22, 33, 38, 46, 51, 55, 60,
	21, 34, 37, 47, 50, 56, 59, 61,
	35, 36, 48, 49, 57, 58, 62, 63
};


__constant__ int ZigZagIndexInvert[] = {
	 0,  1,  8, 16,  9,  2,  3, 10,
	17, 24, 32, 25, 18, 11,  4,  5,
	12, 19, 26, 33, 40, 48, 41, 34,
	27, 20, 13,  6,  7, 14, 21, 28,
	35, 42, 49, 56, 57, 50, 43, 36,
	29, 22, 15, 23, 30, 37, 44, 51,
	58, 59, 52, 45, 38, 31, 39, 46,
	53, 60, 61, 54, 47, 55, 62, 63
};


// https://www.fastcompression.com/blog/jpeg-optimization-review.htm
__constant__ int ForwardQuantizationTable50Luminance[] = {	// 50% COMPRESSION
	 16,  11,  10,  16,  24,  40,  51,  61,
	 12,  12,  14,  19,  26,  58,  60,  55,
	 14,  13,  16,  24,  40,  57,  69,  56,
	 14,  17,  22,  29,  51,  87,  80,  62,
	 18,  22,  37,  56,  68, 109, 103,  77,
	 24,  35,  55,  64,  81, 104, 113,  92,
	 49,  64,  78,  87, 103, 121, 120, 101,
	 72,  92,  95,  98, 112, 100, 103,  99
};


__constant__ int ForwardQuantizationTable50Chrominance[] = {
	17, 18, 42, 47, 99, 99, 99, 99,
	18, 21, 26, 66, 99, 99, 99, 99,
	24, 26, 56, 99, 99, 99, 99, 99,
	47, 66, 99, 99, 99, 99, 99, 99,
	99, 99, 99, 99, 99, 99, 99, 99,
	99, 99, 99, 99, 99, 99, 99, 99,
	99, 99, 99, 99, 99, 99, 99, 99,
	99, 99, 99, 99, 99, 99, 99, 99
};


__constant__ float InvertQuantizationTable50Luminance[] = {
	0.062500f, 0.090909f, 0.100000f, 0.062500f, 0.041667f, 0.025000f, 0.019608f, 0.016393,
	0.083333f, 0.083333f, 0.071429f, 0.052632f, 0.038462f, 0.017241f, 0.016667f, 0.018182,
	0.071429f, 0.076923f, 0.062500f, 0.041667f, 0.025000f, 0.017544f, 0.014493f, 0.017857,
	0.071429f, 0.058824f, 0.045455f, 0.034483f, 0.019608f, 0.011494f, 0.012500f, 0.016129,
	0.055556f, 0.045455f, 0.027027f, 0.017857f, 0.014706f, 0.009174f, 0.009709f, 0.012987,
	0.041667f, 0.028571f, 0.018182f, 0.015625f, 0.012346f, 0.009615f, 0.008850f, 0.010870,
	0.020408f, 0.015625f, 0.012821f, 0.011494f, 0.009709f, 0.008264f, 0.008333f, 0.009901,
	0.013889f, 0.010870f, 0.010526f, 0.010204f, 0.008929f, 0.010000f, 0.009709f, 0.010101
};


__constant__ float InvertQuantizationTable50Chrominance[] = {
	0.058824f, 0.055556f, 0.023810f, 0.021277f, 0.010101f, 0.010101f, 0.010101f, 0.010101f,
	0.055556f, 0.047619f, 0.038462f, 0.015152f, 0.010101f, 0.010101f, 0.010101f, 0.010101f,
	0.041667f, 0.038462f, 0.017857f, 0.010101f, 0.010101f, 0.010101f, 0.010101f, 0.010101f,
	0.021277f, 0.015152f, 0.010101f, 0.010101f, 0.010101f, 0.010101f, 0.010101f, 0.010101f,
	0.010101f, 0.010101f, 0.010101f, 0.010101f, 0.010101f, 0.010101f, 0.010101f, 0.010101f,
	0.010101f, 0.010101f, 0.010101f, 0.010101f, 0.010101f, 0.010101f, 0.010101f, 0.010101f,
	0.010101f, 0.010101f, 0.010101f, 0.010101f, 0.010101f, 0.010101f, 0.010101f, 0.010101f,
	0.010101f, 0.010101f, 0.010101f, 0.010101f, 0.010101f, 0.010101f, 0.010101f, 0.010101f
};


#define C_a 1.387039845322148f //!< a = (2^0.5) * cos(    pi / 16);  Used in forward and inverse DCT.  
#define C_b 1.306562964876377f //!< b = (2^0.5) * cos(    pi /  8);  Used in forward and inverse DCT.  
#define C_c 1.175875602419359f //!< c = (2^0.5) * cos(3 * pi / 16);  Used in forward and inverse DCT.  
#define C_d 0.785694958387102f //!< d = (2^0.5) * cos(5 * pi / 16);  Used in forward and inverse DCT.  
#define C_e 0.541196100146197f //!< e = (2^0.5) * cos(3 * pi /  8);  Used in forward and inverse DCT.  
#define C_f 0.275899379282943f //!< f = (2^0.5) * cos(7 * pi / 16);  Used in forward and inverse DCT.  


/**
*  Normalization constant that is used in forward and inverse DCT
*/
#define C_norm 0.3535533905932737f // 1 / (8^0.5)


/**
*  R	= Y + 1.402 * Cr
*  G	= Y - 0.714 * Cr - 0.344 * Cb
*  B	= Y + 1.772 * Cb
*/
__device__ inline float Conv2R(float y, float cr) {
	return y + 1.402 * ((double)cr - 128);
}

__device__ inline float Conv2G(float y, float cr, float cb) {
	return y - 0.7141 * ((double)cr - 128) - 0.3441 * ((double)cb - 128);
}

__device__ inline float Conv2B(float y, float cb) {
	return y + 1.772 * ((double)cb - 128);
}


/**
*  Y	= 0.299 * R + 0.587 * G + 0.114 * B
*  Cr = 0.500 * R - 0.419 * G - 0.081 * B
*  Cb = -0.169 * R - 0.332 * G + 0.500 * B
*/
__device__ inline unsigned char Conv2Y(unsigned char r, unsigned char g, unsigned char b) {
	return (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b);
}

__device__ inline unsigned char Conv2Cr(unsigned char r, unsigned char g, unsigned char b) {
	return (unsigned char)(0.500 * r - 0.4187 * g - 0.0813 * b + 128);
}

__device__ inline unsigned char Conv2Cb(unsigned char r, unsigned char g, unsigned char b) {
	return (unsigned char)(-0.169 * r - 0.322 * g + 0.500 * b + 128);
}


/**
*  negative values are usually expressed in two's complement.
*  so convert from two's complement representation to signed binary.
*/

#define BYTE2BIT 8

__device__ inline short Float2SignedShort(float signed_short, float quantization) {
	char sign = signed_short < 0;

	unsigned short signed_bit_pos = 1 << (sizeof(unsigned short) * BYTE2BIT - 1);
	unsigned short sign_bit = (unsigned short)(sign * signed_bit_pos);
	unsigned short abs_value = (unsigned short)(signed_short * quantization * (1 - sign * 2));

	return (short)(sign_bit | abs_value);
}

__device__ inline float SignedShort2Float(short signed_short, int quantization) {
	char sign = signed_short < 0;

	unsigned short signed_bit_pos = 1 << (sizeof(unsigned short) * BYTE2BIT - 1);
	unsigned short sign_bit = (unsigned short)(sign * signed_bit_pos);
	unsigned short abs_value = (unsigned short)signed_short - sign * sign_bit;

	return (float)abs_value * quantization * (1 - sign * 2);
}