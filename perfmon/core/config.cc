// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "config.h"

#include <algorithm>
#include <stdexcept>

namespace perfmon {

bool parse_thermal_gate(const char* value) {
  const std::string v = value ? value : "";
  if (v.empty() || v == "off") return false;
  if (v == "on") return true;
  throw std::invalid_argument("AOTRITON_THERMAL_GATE must be 'on' or 'off', got '" + v + "'");
}

std::string bare_arch(const std::string& gcn_arch_name) {
  return gcn_arch_name.substr(0, gcn_arch_name.find(':'));
}

std::vector<std::string> split_arch_list(const std::string& list) {
  std::vector<std::string> out;
  size_t pos = 0;
  while (pos <= list.size()) {
    size_t semi = list.find(';', pos);
    std::string item = list.substr(pos, semi == std::string::npos ? std::string::npos
                                                                  : semi - pos);
    if (!item.empty()) out.push_back(item);
    if (semi == std::string::npos) break;
    pos = semi + 1;
  }
  return out;
}

std::string check_arch(const std::string& gcn_arch_name,
                       const std::vector<std::string>& core_arches) {
  const std::string arch = bare_arch(gcn_arch_name);
  if (std::find(core_arches.begin(), core_arches.end(), arch) != core_arches.end()) {
    return "";
  }
  std::string known;
  for (const auto& a : core_arches) known += (known.empty() ? "" : " ") + a;
  // The core's input fill kernel has no code object for this device, which
  // would fail only at the first launch, so refuse up front.
  return "this device is " + arch + ", but libperfmon_core was built for: " +
         (known.empty() ? std::string("(nothing)") : known);
}

}  // namespace perfmon
