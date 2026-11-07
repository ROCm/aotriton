// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// The out-of-line members of perfmon_abi.h's classes (key functions), so that
// their vtables and typeinfo live in libperfmon_core only.

#include "perfmon_abi.h"

namespace perfmon {

Unsupported::~Unsupported() = default;
Measurement::~Measurement() = default;
Family::~Family() = default;

}  // namespace perfmon
