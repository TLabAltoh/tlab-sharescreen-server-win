#pragma once

#include "pch.h"

using namespace winrt;
using namespace Windows::Foundation;

HANDLE _shmHandle = NULL;
HANDLE _mutexHandle = NULL;
byte* _mappingObject = NULL;

int OpenSharedMemoryMappingObiect(
    LPCTSTR shmName,
    LPCTSTR mutexName)
{
    // open file mapping object.
    _shmHandle = OpenFileMapping(
        FILE_MAP_ALL_ACCESS,
        FALSE,
        shmName
    );

    // If the file mapping object could not be opened
    if (_shmHandle == NULL) {
        // create file mapping object
        _shmHandle = CreateFileMapping(
            INVALID_HANDLE_VALUE,
            NULL,
            PAGE_READWRITE,
            0,
            sizeof(byte),
            shmName
        );

        if (_shmHandle == NULL) return 0;
    }

    _mappingObject = (byte*)MapViewOfFile(
        _shmHandle,
        FILE_MAP_ALL_ACCESS,
        0,
        0,
        sizeof(byte)
    );

    if (_mappingObject == NULL) return 0;

    _mutexHandle = OpenMutex(
        MUTEX_ALL_ACCESS,
        FALSE,
        mutexName
    );

    if (_mutexHandle == NULL) return 0;

    return 1;
}

int FreeUp() {
    // free up resources
    if (_mappingObject != NULL)
    {
        UnmapViewOfFile(_mappingObject);
    }

    if (_shmHandle != NULL)
    {
        CloseHandle(_shmHandle);
    }

    return 0;
}

int main() {
    init_apartment();

    if (OpenSharedMemoryMappingObiect(
        L"TLabShareScreenServer",
        L"Shareing") != 1)
    {
        printf("an error has occurred.\n");
        printf("press enter to finish.");
        while (1) if ('\r' == getch()) break;
        return 0;
    }

    printf("wait for mutex is opened ...\n");

    WaitForSingleObject(_mutexHandle, INFINITE);
    *_mappingObject = 0;
    ReleaseMutex(_mutexHandle);

    FreeUp();

    // http://chokuto.ifdef.jp/advanced/function/FindWindow.html
    TCHAR lpClassName[] = TEXT("TLabShareScreen");
    TCHAR lpTitleName[] = TEXT("Debug");
    HWND target = FindWindow(lpClassName, lpTitleName);
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
