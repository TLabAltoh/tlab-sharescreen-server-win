#pragma once

namespace TPEG {

#ifdef DLL_EXPORT
	__declspec(dllexport) int InitializeDevice(int width, int height, char* encFrameBuffer, unsigned char* decFrameBuffer);
	__declspec(dllexport) int DestroyDevice();
	__declspec(dllexport) void EncFrame(unsigned char* currentFrame);
#else
	__declspec(dllimport) int InitializeDevice(int width, int height, char* encFrameBuffer, unsigned char* decFrameBuffer);
	__declspec(dllimport) int DestroyDevice();
	__declspec(dllimport) void EncFrame(unsigned char* currentFrame);
#endif
}
