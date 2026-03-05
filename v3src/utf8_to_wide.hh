#ifndef AOTRITON_SOURCE_INTERNAL_U8_TO_WIDE_H
#define AOTRITON_SOURCE_INTERNAL_U8_TO_WIDE_H

// Copyright © 2024-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <windows.h>

// Helper function to convert UTF-8 string to wide string
inline static std::wstring utf8_to_wide(const std::string& utf8_str) {
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

#endif
