#pragma once

namespace TPEG {

#ifdef DLL_EXPORT
	__declspec(dllexport) int InitializeDevice(int width, int height, char* encoded_frame_buffer, unsigned char* decoded_frame_buffer);
	__declspec(dllexport) int DestroyDevice();
	__declspec(dllexport) void EncodeFrame(unsigned char* current_frame);
#else
	__declspec(dllimport) int InitializeDevice(int width, int height, char* encoded_frame_buffer, unsigned char* decoded_frame_buffer);
	__declspec(dllimport) int DestroyDevice();
	__declspec(dllimport) void EncodeFrame(unsigned char* current_frame);
#endif
}
