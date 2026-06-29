// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/_internal/log.h>
#include <cstdlib>
#include <iostream>

namespace AOTRITON_NS {

const DebugConfig& debug_config() {
  static DebugConfig cfg = []() {
    DebugConfig c;
    if (const char* lvl = std::getenv("AOTRITON_DEBUG_LEVEL"))
      c.debug_level = std::atoi(lvl);
    if (const char* dir = std::getenv("AOTRITON_TENSOR_DUMP"))
      c.tensor_dump_dir = dir;
    return c;
  }();
  return cfg;
}

void emit_log(std::string_view msg) {
  std::cerr << msg << '\n';
}

} // namespace AOTRITON_NS
