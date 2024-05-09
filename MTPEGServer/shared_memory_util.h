#pragma once

#include "windows_common.h"

namespace {

    byte* _mapping_object = NULL;
    HANDLE _shm_handle = NULL;
    HANDLE _keep_alive_mutex_handle = NULL;

    int OpenSharedMemoryMappingObiect(LPCTSTR shm_name, LPCTSTR mutex_name)
    {
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

        _keep_alive_mutex_handle = CreateMutex(
            NULL,
            FALSE,
            mutex_name
        );

        if (_keep_alive_mutex_handle == NULL) {
            return 0;
        }

        return 1;
    }

    int FreeUp() {  // free up resources

        if (_mapping_object != NULL) {
            UnmapViewOfFile(_mapping_object);
        }

        if (_shm_handle != NULL) {
            CloseHandle(_shm_handle);
        }

        if (_keep_alive_mutex_handle != NULL) {
            CloseHandle(_keep_alive_mutex_handle);
        }

        return 0;
    }
}