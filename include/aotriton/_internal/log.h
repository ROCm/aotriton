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

// Human-readable name for a log level ("DEBUG", "ERROR", ...), used by the
// AOTRITON_LOG prefix.  Defined in log.cc.
const char* log_level_name(int level) noexcept;

// Return the basename component of a (compile-time __FILE__) path.  Runs only
// when a message is actually emitted, so the scan cost is negligible.
const char* log_basename(const char* path) noexcept;

// Returns true when a message at `level` should be emitted.  Use this to guard
// multi-statement log blocks that cannot be expressed as a single AOTRITON_LOG call.
inline bool log_enabled(int level) noexcept {
  return level > 0 && debug_config().debug_level >= level;
}

} // namespace AOTRITON_NS

// Print a printf-style message to stderr when the configured debug level is at
// or above `level`.  Higher debug_level = more output; 0 = disabled.
// Usage: AOTRITON_LOG(LOG_DEBUG, "x = %d, y = %s", x, y)
//
// `fmt` MUST be a string literal: it is concatenated with the "[LEVEL] file:line: "
// prefix at compile time so the whole call becomes a single std::fprintf.  This
// keeps binary size minimal — std::fprintf lives in libc, so no formatting code
// is emitted into the library (unlike std::format, which instantiates a formatter
// for every arithmetic type into every translation unit).  -Wformat still checks
// the arguments against the concatenated literal.  A single fprintf is written
// atomically under glibc's per-stream lock, so log lines never interleave.
//
// Note: pass std::string_view as `%.*s` with `(int)sv.size(), sv.data()` since it
// is not NUL-terminated.
//
// Macro lives outside the namespace so it is usable in any context; the body
// uses fully-qualified names so it resolves regardless of the caller's namespace.
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
