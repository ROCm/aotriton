// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AOTRITON_V2_INTERNAL_LOG_H
#define AOTRITON_V2_INTERNAL_LOG_H

#include <aotriton/config.h>
#include <cstdio>
#include <string>

namespace AOTRITON_NS {

// Log-level integers — same values as AMD_LOG_LEVEL (ROCm/HIP runtime convention).
// Higher value = more verbose.  0 = disabled.
inline constexpr int LOG_NONE        = 0;
inline constexpr int LOG_ERROR       = 1;
inline constexpr int LOG_WARNING     = 2;
inline constexpr int LOG_INFO        = 3;
inline constexpr int LOG_DEBUG       = 4;
inline constexpr int LOG_EXTRA_DEBUG = 5;

struct DebugConfig {
  // Parsed from AOTRITON_DEBUG_LEVEL (default 0 = disabled).
  int debug_level = LOG_NONE;

  // Parsed from AOTRITON_TENSOR_DUMP (default "" = disabled).
  // Reserved for the TensorDump feature (next PR); not used here.
  std::string tensor_dump_dir;
};

// Thread-safe singleton — initialised once on first call.
const DebugConfig& debug_config();

// Level name ("DEBUG", ...) for the AOTRITON_LOG prefix.  Defined in log.cc.
const char* log_level_name(int level) noexcept;

// Basename of a __FILE__ path; only called when a line is actually emitted.
const char* log_basename(const char* path) noexcept;

// True when `level` should be emitted.  Guards multi-statement log blocks that
// cannot be written as a single AOTRITON_LOG call.
inline bool log_enabled(int level) noexcept {
  return level > 0 && debug_config().debug_level >= level;
}

} // namespace AOTRITON_NS

// printf-style log to stderr, emitted when debug_level >= level (0 = disabled).
// Usage: AOTRITON_LOG(LOG_DEBUG, "x = %d, y = %s", x, y)
//
// `fmt` MUST be a string literal; it is concatenated with the prefix at compile
// time into one std::fprintf.  fprintf lives in libc, so no formatting code is
// emitted into the library (unlike std::format).  -Wformat still checks the args,
// and a single fprintf is atomic under glibc's per-stream lock.  Pass a
// std::string_view as "%.*s" with (int)sv.size(), sv.data() (not NUL-terminated).
#define AOTRITON_LOG(level, fmt, ...)                                            \
  do {                                                                           \
    const int _aotriton_level = (level);                                         \
    if (_aotriton_level > 0 &&                                                   \
        AOTRITON_NS::debug_config().debug_level >= _aotriton_level)              \
      std::fprintf(stderr, "[%s] %s:%d: " fmt "\n",                             \
                   AOTRITON_NS::log_level_name(_aotriton_level),                 \
                   AOTRITON_NS::log_basename(__FILE__), __LINE__                 \
                   __VA_OPT__(,) __VA_ARGS__);                                    \
  } while (0)

#endif // AOTRITON_V2_INTERNAL_LOG_H
