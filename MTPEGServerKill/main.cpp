#pragma once

#include "pch.h"

using namespace winrt;
using namespace Windows::Foundation;

HANDLE _shm_handle = NULL;
HANDLE _mutex_handle = NULL;
byte* _mapping_object = NULL;

int OpenSharedMemoryMappingObiect(LPCTSTR shm_name, LPCTSTR mutex_name) {

    _shm_handle = OpenFileMapping(  // open file mapping object
        FILE_MAP_ALL_ACCESS,
        FALSE,
        shm_name
    );

    if (_shm_handle == NULL) {  // if the file mapping object could not be opened
        
        _shm_handle = CreateFileMapping(    // create file mapping object
            INVALID_HANDLE_VALUE,
            NULL,
            PAGE_READWRITE,
            0,
            sizeof(byte),
            shm_name
        );

        if (_shm_handle == NULL) {
            return 0;
        }
    }

    _mapping_object = (byte*)MapViewOfFile(
        _shm_handle,
        FILE_MAP_ALL_ACCESS,
        0,
        0,
        sizeof(byte)
    );

    if (_mapping_object == NULL) {
        return 0;
    }

    _mutex_handle = OpenMutex(
        MUTEX_ALL_ACCESS,
        FALSE,
        mutex_name
    );

    if (_mutex_handle == NULL) {
        return 0;
    }

    return 1;
}

int FreeUp() {

    if (_mapping_object != NULL)    // free up resources
    {
        UnmapViewOfFile(_mapping_object);
    }

    if (_shm_handle != NULL)
    {
        CloseHandle(_shm_handle);
    }

    return 0;
}

int main() {
    init_apartment();

    if (OpenSharedMemoryMappingObiect(L"MTPEGServer", L"Shareing") != 1)
    {
        printf("error while starting mtpeg server kill, press enter to finish");

        while (1) {
            if ('\r' == _getch()) {
                break;
            }
        }

        return 0;
    }

    WaitForSingleObject(_mutex_handle, INFINITE);
    *_mapping_object = 0;
    ReleaseMutex(_mutex_handle);

    FreeUp();

    // http://chokuto.ifdef.jp/advanced/function/FindWindow.html
    TCHAR lp_class_name[] = TEXT("Motion TPEG Server");
    TCHAR lp_title_name[] = TEXT("Capture Result Debug Window");
    HWND target = FindWindow(lp_class_name, lp_title_name);
    RECT target_rect;
    GetWindowRect(target, &target_rect);
    SendMessage(
        target,
        WM_LBUTTONDOWN,
        MK_LBUTTON,
        MAKELPARAM(target_rect.left + 1, target_rect.bottom + 1)
    );

    return 0;
}
