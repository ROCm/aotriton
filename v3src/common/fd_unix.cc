// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/_internal/fd.h>

#if !defined(_WIN32)

#include <unistd.h>

namespace AOTRITON_NS {

int fd_close(fd_t fd) {
  return close(fd);
}

ssize_t fd_read(fd_t fd, void *buf, size_t count) {
  return ::read(fd, buf, count);
}

off_t fd_seek(fd_t fd, off_t offset, int whence) {
  return ::lseek(fd, offset, whence);
}

} // namespace AOTRITON_NS

#endif // !defined(_WIN32)
