// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AOTRITON_V3_FD_H
#define AOTRITON_V3_FD_H

#include <aotriton/config.h>
#include <cstddef>
#include <sys/types.h>

#if defined(_WIN32)
#include <filesystem>
#include <windows.h>

#if !defined(ssize_t)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif
#endif

namespace AOTRITON_NS {

#if defined(_WIN32)
using fd_t = HANDLE;

inline fd_t invalid_fd() {
  return INVALID_HANDLE_VALUE;
}

inline bool fd_is_valid(fd_t fd) {
  return fd != INVALID_HANDLE_VALUE;
}

fd_t fd_open(const std::filesystem::path& pathname);
#else
using fd_t = int;

inline constexpr fd_t invalid_fd() {
  return -1;
}

inline bool fd_is_valid(fd_t fd) {
  return fd != -1;
}
#endif

int fd_close(fd_t fd);
ssize_t fd_read(fd_t fd, void *buf, size_t count);
off_t fd_seek(fd_t fd, off_t offset, int whence);

} // namespace AOTRITON_NS

#endif // AOTRITON_V3_FD_H
