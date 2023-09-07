#pragma once

#include "TLabWindows.h"

namespace {

    HANDLE _shmHandle = NULL;
    HANDLE _keepAliveMutexHandle = NULL;
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

        _keepAliveMutexHandle = CreateMutex(
            NULL,
            FALSE,
            mutexName
        );

        if (_keepAliveMutexHandle == NULL) return 0;

        return 1;
    }

    int FreeUp() {
        // free up resources
        if (_mappingObject != NULL) UnmapViewOfFile(_mappingObject);

        if (_shmHandle != NULL) CloseHandle(_shmHandle);

        if (_keepAliveMutexHandle != NULL) CloseHandle(_keepAliveMutexHandle);

        return 0;
    }
}