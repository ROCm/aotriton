// Copyright © 2024-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AOTRITON_V2_API_PACKED_KERNEL_UNIX_H
#define AOTRITON_V2_API_PACKED_KERNEL_UNIX_H

#include <fcntl.h>
#include <unistd.h>

inline static int fd_close(int fd) {
    return close(fd);
}

inline static ssize_t fd_read(int fd, void *buf, size_t count) {
    return ::read(fd, buf, count);
}

inline static off_t fd_seek(int fd, off_t offset, int whence) {
    return ::lseek(fd, offset, whence);
}

#endif // AOTRITON_V2_API_PACKED_KERNEL_UNIX_H
