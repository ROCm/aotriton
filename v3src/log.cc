// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/_internal/log.h>
#include <algorithm>
#include <cerrno>
#include <cstdlib>

namespace AOTRITON_NS {

const DebugConfig& debug_config() {
  static DebugConfig cfg = []() {
    DebugConfig c;
    if (const char* lvl = std::getenv("AOTRITON_DEBUG_LEVEL")) {
      char* end = nullptr;
      errno = 0;
      long val = std::strtol(lvl, &end, 10);
      if (end != lvl && *end == '\0' && errno == 0)
        c.debug_level = static_cast<int>(std::clamp(val, 0L, 5L));
    }
    if (const char* dir = std::getenv("AOTRITON_TENSOR_DUMP"))
      c.tensor_dump_dir = dir;
    return c;
  }();
  return cfg;
}

const char* log_level_name(int level) noexcept {
  switch (level) {
    case 1: return "ERROR";
    case 2: return "WARNING";
    case 3: return "INFO";
    case 4: return "DEBUG";
    case 5: return "EXTRA_DEBUG";
    default: return "LOG";
  }
}

const char* log_basename(const char* path) noexcept {
  const char* last = path;
  for (const char* p = path; *p; ++p)
    if (*p == '/' || *p == '\\') last = p + 1;
  return last;
}

} // namespace AOTRITON_NS
