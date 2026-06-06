// Copyright © 2024-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AOTRITON_V2_API_PACKED_KERNEL_UNIX_H
#define AOTRITON_V2_API_PACKED_KERNEL_UNIX_H

#include <fcntl.h>
#include <cstdint>
#include <unistd.h>

inline static int fd_close(std::intptr_t fd) {
    return close(static_cast<int>(fd));
}

inline static ssize_t fd_read(std::intptr_t fd, void *buf, size_t count) {
    return ::read(static_cast<int>(fd), buf, count);
}

inline static off_t fd_seek(std::intptr_t fd, off_t offset, int whence) {
    return ::lseek(static_cast<int>(fd), offset, whence);
}

#endif // AOTRITON_V2_API_PACKED_KERNEL_UNIX_H
