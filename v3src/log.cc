// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/_internal/log.h>
#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <iostream>

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

static const char* level_name(int level) noexcept {
  switch (level) {
    case 1: return "ERROR";
    case 2: return "WARNING";
    case 3: return "INFO";
    case 4: return "DEBUG";
    case 5: return "EXTRA_DEBUG";
    default: return "LOG";
  }
}

static const char* basename(const char* path) noexcept {
  const char* last = path;
  for (const char* p = path; *p; ++p)
    if (*p == '/' || *p == '\\') last = p + 1;
  return last;
}

void emit_log(int level, const char* file, int line, std::string_view msg) {
  std::cerr << std::format("[{}] {}:{}: {}\n", level_name(level), basename(file), line, msg);
}

// std::vformat is instantiated here ONCE, instead of being inlined into every
// AOTRITON_LOG call site.  See the header for the rationale.
void emit_log(int level, const char* file, int line,
              std::string_view fmt, std::format_args args) {
  emit_log(level, file, line, std::vformat(fmt, args));
}

} // namespace AOTRITON_NS
