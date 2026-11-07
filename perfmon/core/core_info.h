// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Facts about this build of libperfmon_core, fixed at configure time. They
// live in the library, not in the runner, so a core rebuilt for a longer arch
// list reports the new list without relinking any subject.

#ifndef PERFMON_CORE_CORE_INFO_H
#define PERFMON_CORE_CORE_INFO_H

namespace perfmon {

// The source commit the core was built from ("<sha>" or "<sha>-dirty").
const char* core_commit();
// The arches the core has device code for, ';'-separated.
const char* core_arches();
// The ROCm version in the library's name.
const char* core_rocm_version();

}  // namespace perfmon

#endif  // PERFMON_CORE_CORE_INFO_H
