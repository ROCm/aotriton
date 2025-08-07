#ifndef AOTRITON_V2_API_PACKED_KERNEL_WIN32_H
#define AOTRITON_V2_API_PACKED_KERNEL_WIN32_H

// Copyright © 2024-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <cstring>
#include <iostream>
#include <string>
#include <windows.h>

#if !defined(ssize_t)
    #include <BaseTsd.h> // For SSIZE_T
    typedef SSIZE_T ssize_t;
#endif

// Invalid handle value that can be returned as int
constexpr int INVALID_FD = -1;

// Helper function to convert UTF-8 string to wide string
static std::wstring utf8_to_wide(const std::string& utf8_str) {
    if (utf8_str.empty()) {
        return std::wstring();
    }
    
    // Calculate required buffer size
    int wpath_len = MultiByteToWideChar(CP_UTF8, 0, utf8_str.c_str(), -1, nullptr, 0);
    if (wpath_len == 0) {
        return std::wstring();
    }
    
    // Allocate and convert
    std::wstring wide_str(wpath_len - 1, L'\0'); // -1 because MultiByteToWideChar includes null terminator
    if (MultiByteToWideChar(CP_UTF8, 0, utf8_str.c_str(), -1, &wide_str[0], wpath_len) == 0) {
        return std::wstring();
    }
    
    return wide_str;
}

// Helper function to add long path prefix for Windows
static std::wstring add_long_path_prefix(const std::wstring& path) {
    // Regular paths - add \\?\ prefix
    return LR"(\\?\)" + path;
}

// Convert HANDLE to int for our interface
// We use intptr_t to ensure the HANDLE fits
static int handle_to_fd(HANDLE handle) {
    if (handle == INVALID_HANDLE_VALUE) {
        return INVALID_FD;
    }
    // Cast HANDLE to intptr_t then to int
    // On 64-bit Windows, HANDLE is 64-bit, but we need to maintain int interface
    // This is safe as Windows guarantees handles fit in 32 bits for compatibility
    return static_cast<int>(reinterpret_cast<intptr_t>(handle));
}

// Convert int back to HANDLE
static HANDLE fd_to_handle(int fd) {
    if (fd == INVALID_FD) {
        return INVALID_HANDLE_VALUE;
    }
    return reinterpret_cast<HANDLE>(static_cast<intptr_t>(fd));
}

static int fd_open(const char* pathname) {
    // Convert UTF-8 pathname to wide string
    std::wstring wide_path = utf8_to_wide(std::string(pathname));
    if (wide_path.empty()) {
        return INVALID_FD;
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
        char error_message[256] = {0};
        FormatMessageA(
            FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
            nullptr,
            error_code,
            MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
            error_message,
            sizeof(error_message),
            nullptr
        );
        // Remove trailing newline
        size_t len = strlen(error_message);
        if (len > 0 && (error_message[len-1] == '\n' || error_message[len-1] == '\r')) {
            error_message[len-1] = '\0';
        }
        std::cerr << "CreateFileW failed for: " << pathname 
                  << ", Win32 error " << error_code 
                  << ": " << error_message << std::endl;
#endif
        
        return INVALID_FD;
    }
    
    // Return the handle as an int
    return handle_to_fd(file_handle);
}

static int fd_close(int fd) {
    HANDLE file_handle = fd_to_handle(fd);
    if (file_handle == INVALID_HANDLE_VALUE) {
        return -1;
    }
    
    // Close using Win32 API
    if (!CloseHandle(file_handle)) {
        DWORD error_code = GetLastError();
        
#if AOTRITON_KERNEL_VERBOSE
        std::cerr << "CloseHandle failed with error: " << error_code << std::endl;
#endif
        
        return -1;
    }
    
    return 0;
}

static ssize_t fd_read(int fd, void *buf, size_t count) {
    if (count == 0) {
        return 0;
    }
    
    HANDLE file_handle = fd_to_handle(fd);
    if (file_handle == INVALID_HANDLE_VALUE) {
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
    if (!ReadFile(file_handle, buf, bytes_to_read, &bytes_read, nullptr)) {
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

#endif // AOTRITON_V2_API_PACKED_KERNEL_WIN32_H