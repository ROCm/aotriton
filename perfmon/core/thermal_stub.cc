// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// thermal.h without amd-smi: no temperature source, so the runner refuses
// AOTRITON_THERMAL_GATE=on and reports temperatures as null.

#include "thermal.h"

#include <stdexcept>

namespace perfmon {

bool thermal_available() {
  return false;
}

double read_temperature_c() {
  throw std::runtime_error("libperfmon_core was built without amd-smi (PERFMON_ENABLE_AMDSMI=OFF)");
}

void wait_until_cool(double, std::ostream&) {
  read_temperature_c();
}

}  // namespace perfmon
