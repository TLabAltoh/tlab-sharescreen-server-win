#pragma once

unsigned char* _current_frame_buffer_gpu;

unsigned char* _decoded_frame_buffer_gpu;

unsigned char* _prev_frame_buffer_gpu;

unsigned int _rgba_frame_buffer_size;

unsigned int _rgb_frame_buffer_size;

unsigned short* _block_diff_sum_buffer_gpu;

unsigned int _block_diff_sum_buffer_size;

short* _dct_result_frame_buffer_gpu;

short* _idct_result_frame_buffer_gpu;

unsigned int _dct_frame_buffer_size;

char* _encoded_frame_buffer_gpu;

unsigned int _encoded_frame_buffer_size;

char* _encoded_frame_buffer_cpu;

unsigned char* _decoded_frame_buffer_cpu;

int _width;

int _height;

int _blockWidth;

int _blockHeight;