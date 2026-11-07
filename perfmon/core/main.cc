// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// main() of every subject's bin/runner. It is built per subject, linked with
// that subject's family adapter and the shared core; there is no plugin
// loading, since a subject's runner only ever runs that subject. Everything
// else lives in the core (runner.cc), so all subjects are timed by the same
// code.

#include "perfmon_abi.h"

int main() {
  return perfmon::run_runner(perfmon::family(), PMON_ABI_VERSION);
}
