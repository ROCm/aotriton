// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// The thermal gate's temperature source. Exactly one of thermal_amdsmi.cc
// (PERFMON_ENABLE_AMDSMI=ON) and thermal_stub.cc provides these functions.
//
// There is deliberately no sysfs fallback: a second temperature source is a
// second thing to keep correct, and a silently degraded gate is how ungated
// numbers get published as gated ones. Without amd-smi the gate cannot be
// turned on at all.

#ifndef PERFMON_CORE_THERMAL_H
#define PERFMON_CORE_THERMAL_H

#include <ostream>

namespace perfmon {

// False in the stub build.
bool thermal_available();

// The junction (or, where the GPU has none, edge) temperature of this
// process's GPU (HIP device 0), in degrees Celsius. Throws on failure,
// including in the stub build.
double read_temperature_c();

// Blocks while the temperature is above `limit_c`, writing one OVERHEATING:
// keep-alive line to `out` per poll. Throws where read_temperature_c does.
void wait_until_cool(double limit_c, std::ostream& out);

}  // namespace perfmon

#endif  // PERFMON_CORE_THERMAL_H
