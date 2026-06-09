#ifndef AOTRITON_V2_API_PACKED_KERNEL_WIN32_H
#define AOTRITON_V2_API_PACKED_KERNEL_WIN32_H

// Copyright © 2024-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/_internal/fd.h>
#include <algorithm>
#include <cstdio>       // SEEK_SET/SEEK_CUR/SEEK_END
#include <cwchar>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <sys/types.h>  // off_t (POSIX type; MSVC/MinGW provide it here)

#if !defined(ssize_t)
    #include <BaseTsd.h> // For SSIZE_T
    typedef SSIZE_T ssize_t;
#endif

namespace AOTRITON_NS {

// Helper function to add long path prefix for Windows
static std::wstring add_long_path_prefix(std::wstring path) {
    std::replace(path.begin(), path.end(), L'/', L'\\');
    if (path.rfind(LR"(\\?\)", 0) == 0 || path.rfind(LR"(\\.\)", 0) == 0) {
        return path;
    }
    if (path.rfind(LR"(\\)", 0) == 0) {
        return LR"(\\?\UNC\)" + path.substr(2);
    }
    return LR"(\\?\)" + path;
}

static fd_t fd_open(const std::filesystem::path& pathname) {
    std::wstring wide_path = pathname.wstring();
    if (wide_path.empty()) {
        return invalid_fd();
    }

    wide_path = add_long_path_prefix(wide_path);
    
    DWORD desired_access = GENERIC_READ;
    DWORD share_mode = FILE_SHARE_READ;
    DWORD creation_disposition = OPEN_EXISTING;
    DWORD file_flags = FILE_ATTRIBUTE_NORMAL;

    HANDLE file_handle = CreateFileW(
        wide_path.c_str(),
        desired_access,
        share_mode,
        nullptr,  // Default security
        creation_disposition,
        file_flags,
        nullptr   // No template file
    );
    
    if (file_handle == INVALID_HANDLE_VALUE) {
        DWORD error_code = GetLastError();
        
#if AOTRITON_KERNEL_VERBOSE
        wchar_t error_message[256] = {0};
        FormatMessageW(
            FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
            nullptr,
            error_code,
            MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
            error_message,
            sizeof(error_message) / sizeof(error_message[0]),
            nullptr
        );
        for (size_t len = wcslen(error_message);
             len > 0 && (error_message[len - 1] == L'\n' || error_message[len - 1] == L'\r');
             --len) {
            error_message[len - 1] = L'\0';
        }
        std::wcerr << L"CreateFileW failed for: " << wide_path
                   << L", Win32 error " << error_code
                   << L": " << error_message << std::endl;
#endif
        
        return invalid_fd();
    }
    
    return file_handle;
}

static int fd_close(fd_t fd) {
    if (!fd_is_valid(fd)) {
        return -1;
    }
    
    // Close using Win32 API
    if (!CloseHandle(fd)) {
        DWORD error_code = GetLastError();
        
#if AOTRITON_KERNEL_VERBOSE
        std::cerr << "CloseHandle failed with error: " << error_code << std::endl;
#endif
        
        return -1;
    }
    
    return 0;
}

static ssize_t fd_read(fd_t fd, void *buf, size_t count) {
    if (count == 0) {
        return 0;
    }
    
    if (!fd_is_valid(fd)) {
        return -1;
    }
    
    // ReadFile uses DWORD for byte count, so we need to handle large reads
    DWORD bytes_to_read;
    if (count > MAXDWORD) {
        bytes_to_read = MAXDWORD;
    } else {
        bytes_to_read = static_cast<DWORD>(count);
    }
    
    DWORD bytes_read = 0;
    
    // Read using Win32 API
    if (!ReadFile(fd, buf, bytes_to_read, &bytes_read, nullptr)) {
        DWORD error_code = GetLastError();
        
        // ERROR_HANDLE_EOF is not really an error
        if (error_code == ERROR_HANDLE_EOF) {
            return 0;
        }
        
#if AOTRITON_KERNEL_VERBOSE
        std::cerr << "ReadFile failed with error: " << error_code << std::endl;
#endif
        
        return -1;
    }
    
    return static_cast<ssize_t>(bytes_read);
}

static off_t fd_seek(fd_t fd, off_t offset, int whence) {
    if (!fd_is_valid(fd))
        return -1;
    DWORD move_method;
    switch (whence) {
        case SEEK_SET: move_method = FILE_BEGIN;   break;
        case SEEK_CUR: move_method = FILE_CURRENT; break;
        case SEEK_END: move_method = FILE_END;     break;
        default: return -1;
    }
    LARGE_INTEGER li_offset, li_new;
    li_offset.QuadPart = static_cast<LONGLONG>(offset);
    if (!SetFilePointerEx(fd, li_offset, &li_new, move_method))
        return -1;
    return static_cast<off_t>(li_new.QuadPart);
}

} // namespace AOTRITON_NS

#endif // AOTRITON_V2_API_PACKED_KERNEL_WIN32_H
