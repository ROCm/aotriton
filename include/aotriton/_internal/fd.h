// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AOTRITON_V3_FD_H
#define AOTRITON_V3_FD_H

#include <aotriton/config.h>

#if defined(_WIN32)
#include <windows.h>
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
#else
using fd_t = int;

inline constexpr fd_t invalid_fd() {
  return -1;
}

inline constexpr bool fd_is_valid(fd_t fd) {
  return fd != -1;
}
#endif

} // namespace AOTRITON_NS

#endif // AOTRITON_V3_FD_H
