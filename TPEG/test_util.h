#pragma once

#include "test_common.h"

int CheckFilePath(char* path, char* extension) {
	int path_size = strlen(path);
	int extension_size = strlen(extension);
	int filename_size = path_size - extension_size;

	printf("\npath name: %s\n", path);
	printf("path_size: %d, extension_size: %d, filename_size (path_size - extension_size): %d\n", path_size, extension_size, filename_size);

	if (filename_size < 1 || strcmp(path + filename_size, extension) != 0) {
		printf("FileCheck: error\n");
		printf("path value: %.*s\n", extension_size, path + filename_size);
		return 1;
	}

	printf("\n");

	return 0;
}

wchar_t* UTF8To16(char* char_str) {
	int size = MultiByteToWideChar(CP_UTF8, 0, char_str, -1, nullptr, 0);
	wchar_t* utf16str = new wchar_t[size];
	MultiByteToWideChar(CP_UTF8, 0, char_str, -1, utf16str, size);
	return utf16str;
}

TCHAR* current_working_directory()
{
	TCHAR pwd[MAX_PATH];
	GetCurrentDirectory(MAX_PATH, pwd);
	return pwd;
}