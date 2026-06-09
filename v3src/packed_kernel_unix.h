// Copyright © 2024-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AOTRITON_V2_API_PACKED_KERNEL_UNIX_H
#define AOTRITON_V2_API_PACKED_KERNEL_UNIX_H

#include <aotriton/_internal/fd.h>
#include <fcntl.h>
#include <unistd.h>

namespace AOTRITON_NS {

inline static int fd_close(fd_t fd) {
    return close(fd);
}

inline static ssize_t fd_read(fd_t fd, void *buf, size_t count) {
    return ::read(fd, buf, count);
}

inline static off_t fd_seek(fd_t fd, off_t offset, int whence) {
    return ::lseek(fd, offset, whence);
}

} // namespace AOTRITON_NS

#endif // AOTRITON_V2_API_PACKED_KERNEL_UNIX_H
