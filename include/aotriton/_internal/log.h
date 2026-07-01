// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AOTRITON_V2_INTERNAL_LOG_H
#define AOTRITON_V2_INTERNAL_LOG_H

#include <aotriton/config.h>
#include <format>
#include <string>
#include <string_view>

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

// Write a pre-formatted message to stderr with level/file/line prefix.
// Defined in log.cc so that <iostream> is NOT pulled into every TU via this header.
void emit_log(int level, const char* file, int line, std::string_view msg);

// Type-erased formatting sink: runs std::vformat in log.cc (a SINGLE TU) and
// forwards to emit_log.  Keeping the heavy vformat machinery out of line is the
// whole point — otherwise std::format gets inlined into every call site (e.g. the
// ~3700 generated autotune TUs), bloating .text by megabytes of duplicated,
// internal-linkage code the linker cannot deduplicate.
void emit_log(int level, const char* file, int line,
              std::string_view fmt, std::format_args args);

// Thin per-call wrapper.  Only make_format_args() (trivial) is instantiated at
// the call site; the format_string<Args...> parameter preserves compile-time
// format-string checking.  The wrapper is also required for correctness: C++23's
// std::make_format_args takes lvalue references, so a prvalue argument (e.g.
// int(kernel_index)) must first be materialised into the named parameter here.
template <typename... Args>
inline void log_formatted(int level, const char* file, int line,
                          std::format_string<Args...> fmt, Args&&... args) {
  emit_log(level, file, line, fmt.get(), std::make_format_args(args...));
}

// Returns true when a message at `level` should be emitted.  Use this to guard
// multi-statement log blocks that cannot be expressed as a single AOTRITON_LOG call.
inline bool log_enabled(int level) noexcept {
  return level > 0 && debug_config().debug_level >= level;
}

} // namespace AOTRITON_NS

// Print a std::format-style message to stderr when the configured debug level
// is at or above `level`.  Higher debug_level = more output; 0 = disabled.
// Usage: AOTRITON_LOG(LOG_DEBUG, "x = {}, y = {}", x, y)
// Macro lives outside the namespace so it is usable in any context; the body
// uses fully-qualified names so it resolves regardless of the caller's namespace.
#define AOTRITON_LOG(level, ...)                                                     \
  do {                                                                               \
    const int _aotriton_level = (level);                                             \
    if (_aotriton_level > 0 &&                                                       \
        AOTRITON_NS::debug_config().debug_level >= _aotriton_level)                  \
      AOTRITON_NS::log_formatted(_aotriton_level, __FILE__, __LINE__, __VA_ARGS__);  \
  } while (0)

#endif // AOTRITON_V2_INTERNAL_LOG_H
