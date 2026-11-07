// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "check.h"
#include "config.h"

#include <stdexcept>

using namespace perfmon;

int main() {
  CHECK_EQ(parse_thermal_gate(nullptr), false);
  CHECK_EQ(parse_thermal_gate(""), false);
  CHECK_EQ(parse_thermal_gate("off"), false);
  CHECK_EQ(parse_thermal_gate("on"), true);
  CHECK_THROWS(parse_thermal_gate("ON"), "AOTRITON_THERMAL_GATE");
  CHECK_THROWS(parse_thermal_gate("1"), "AOTRITON_THERMAL_GATE");
  CHECK_EQ(kThermalLimitC, 85.0);

  CHECK_EQ(bare_arch("gfx942:sramecc+:xnack-"), "gfx942");
  CHECK_EQ(bare_arch("gfx1100"), "gfx1100");

  auto arches = split_arch_list(";gfx90a;gfx942;;gfx950");
  CHECK_EQ(arches.size(), 3u);
  CHECK_EQ(arches[1], "gfx942");
  CHECK(split_arch_list("").empty());

  CHECK(check_arch("gfx942:sramecc+:xnack-", arches).empty());
  std::string why = check_arch("gfx1100", arches);
  CHECK(why.find("gfx1100") != std::string::npos);
  CHECK(why.find("gfx90a gfx942 gfx950") != std::string::npos);
  CHECK(!check_arch("gfx942", {}).empty());
  CHECK_MAIN_RETURN();
}
