// Copyright © 2024-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AOTRITON_V2_API_PACKED_KERNEL_UNIX_H
#define AOTRITON_V2_API_PACKED_KERNEL_UNIX_H

#include <fcntl.h>
#include <unistd.h>

static int fd_close(int fd) {
    return close(fd);
}

static ssize_t fd_read(int fd, void *buf, size_t count) {
    return ::read(fd, buf, count);
}

#endif // AOTRITON_V2_API_PACKED_KERNEL_UNIX_H