// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Runner settings and startup checks that need no GPU. Standard library only,
// so they are unit-tested without ROCm.

#ifndef PERFMON_CORE_CONFIG_H
#define PERFMON_CORE_CONFIG_H

#include <string>
#include <vector>

namespace perfmon {

// Fixed by design: one threshold for every host, so gated numbers from
// different hosts mean the same thing.
constexpr double kThermalLimitC = 85.0;

// AOTRITON_THERMAL_GATE: "on" or "off"; unset or empty means off. Anything
// else throws std::invalid_argument -- a typo must not silently disable the
// gate on a host that needs it.
bool parse_thermal_gate(const char* value);

// "gfx942:sramecc+:xnack-" -> "gfx942".
std::string bare_arch(const std::string& gcn_arch_name);

// "gfx90a;gfx942" -> {"gfx90a", "gfx942"}; empty items are dropped.
std::vector<std::string> split_arch_list(const std::string& list);

// Empty if `gcn_arch_name` is in `core_arches`, else why the runner refuses.
std::string check_arch(const std::string& gcn_arch_name,
                       const std::vector<std::string>& core_arches);

}  // namespace perfmon

#endif  // PERFMON_CORE_CONFIG_H
