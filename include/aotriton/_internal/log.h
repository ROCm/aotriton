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

// Write a pre-formatted message to stderr.  Defined in log.cc so that
// <iostream> is NOT pulled into every translation unit via this header.
void emit_log(std::string_view msg);

} // namespace AOTRITON_NS

// Print a std::format-style message to stderr when the configured debug level
// is at or above `level`.  Higher debug_level = more output; 0 = disabled.
// Usage: AOTRITON_LOG(LOG_DEBUG, "x = {}, y = {}", x, y)
// Macro lives outside the namespace so it is usable in any context; the body
// uses fully-qualified names so it resolves regardless of the caller's namespace.
#define AOTRITON_LOG(level, ...)                                                     \
  do {                                                                               \
    if ((level) > 0 && AOTRITON_NS::debug_config().debug_level >= (level))          \
      AOTRITON_NS::emit_log(std::format(__VA_ARGS__));                               \
  } while (0)

#endif // AOTRITON_V2_INTERNAL_LOG_H
