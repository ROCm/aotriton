// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/_internal/fd.h>
#include <aotriton/_internal/log.h>

#if defined(_WIN32)

#include <algorithm>
#include <cstdio>
#include <cwchar>
#include <string>

namespace AOTRITON_NS {

namespace {

std::wstring add_long_path_prefix(std::wstring path) {
  std::replace(path.begin(), path.end(), L'/', L'\\');
  if (path.rfind(LR"(\\?\)", 0) == 0 || path.rfind(LR"(\\.\)", 0) == 0) {
    return path;
  }
  if (path.rfind(LR"(\\)", 0) == 0) {
    return LR"(\\?\UNC\)" + path.substr(2);
  }
  return LR"(\\?\)" + path;
}

} // namespace

fd_t fd_open(const std::filesystem::path& pathname) {
  std::wstring wide_path = pathname.wstring();
  if (wide_path.empty()) {
    return invalid_fd();
  }

  wide_path = add_long_path_prefix(wide_path);
  HANDLE file_handle = CreateFileW(wide_path.c_str(),
                                   GENERIC_READ,
                                   FILE_SHARE_READ,
                                   nullptr,
                                   OPEN_EXISTING,
                                   FILE_ATTRIBUTE_NORMAL,
                                   nullptr);

  if (file_handle == INVALID_HANDLE_VALUE) {
    DWORD error_code = GetLastError();
    char error_message[256] = {0};
    FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                   nullptr,
                   error_code,
                   MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                   error_message,
                   sizeof(error_message),
                   nullptr);
    size_t len = strlen(error_message);
    while (len > 0 && (error_message[len-1] == '\n' || error_message[len-1] == '\r'))
      error_message[--len] = '\0';
    AOTRITON_LOG(LOG_DEBUG, "CreateFileW failed for: %s, Win32 error %lu: %s",
                 pathname.string().c_str(), error_code, error_message);
    return invalid_fd();
  }

  return file_handle;
}

int fd_close(fd_t fd) {
  if (!fd_is_valid(fd)) {
    return -1;
  }
  if (!CloseHandle(fd)) {
    AOTRITON_LOG(LOG_DEBUG, "CloseHandle failed with error: %lu", GetLastError());
    return -1;
  }
  return 0;
}

ssize_t fd_read(fd_t fd, void *buf, size_t count) {
  if (count == 0) {
    return 0;
  }
  if (!fd_is_valid(fd)) {
    return -1;
  }

  DWORD bytes_to_read = static_cast<DWORD>(std::min<size_t>(count, MAXDWORD));
  DWORD bytes_read = 0;
  if (!ReadFile(fd, buf, bytes_to_read, &bytes_read, nullptr)) {
    DWORD error_code = GetLastError();
    if (error_code == ERROR_HANDLE_EOF) {
      return 0;
    }
    AOTRITON_LOG(LOG_DEBUG, "ReadFile failed with error: %lu", error_code);
    return -1;
  }

  return static_cast<ssize_t>(bytes_read);
}

off_t fd_seek(fd_t fd, off_t offset, int whence) {
  if (!fd_is_valid(fd)) {
    return -1;
  }

  DWORD move_method;
  switch (whence) {
  case SEEK_SET:
    move_method = FILE_BEGIN;
    break;
  case SEEK_CUR:
    move_method = FILE_CURRENT;
    break;
  case SEEK_END:
    move_method = FILE_END;
    break;
  default:
    return -1;
  }

  LARGE_INTEGER li_offset;
  LARGE_INTEGER li_new;
  li_offset.QuadPart = static_cast<LONGLONG>(offset);
  if (!SetFilePointerEx(fd, li_offset, &li_new, move_method)) {
    return -1;
  }
  return static_cast<off_t>(li_new.QuadPart);
}

} // namespace AOTRITON_NS

#endif // defined(_WIN32)
