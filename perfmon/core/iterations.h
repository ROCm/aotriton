// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// How many timed iterations a measurement takes. Standard library only, so it
// is unit-tested without ROCm. timing.h documents the rule in context.

#ifndef PERFMON_CORE_ITERATIONS_H
#define PERFMON_CORE_ITERATIONS_H

namespace perfmon {

// Upper bound of the adaptive count, and the count of every call short enough
// that kIterationBudgetSeconds / t_call reaches it (t_call <= 25 ms). 400
// iterations cut the spread of short entries from about 4 % to about 1 % on
// CDNA (perfmon-noise-repeats experiment); repeated calls or fresh processes
// did not help.
constexpr int kMaxIters = 400;
// Lower bound of the adaptive count. 20 samples gave stable medians at seqlen
// 16384 (in-run spread < 1.3 % on gfx950, phase-1 report, noise section).
constexpr int kMinIters = 20;
// Target time of the timed loop, in seconds, when the count is adaptive. The
// loop also pays a 1 GiB L2 flush per iteration, which this does not count.
constexpr double kIterationBudgetSeconds = 10.0;
// Largest PERFMON_ITERATIONS accepted: two events and 4+ graph nodes per
// iteration must stay a reasonable graph, and the override is an experiment
// knob, not a production setting.
constexpr int kMaxIterationsOverride = 5000;

// clamp(ceil(kIterationBudgetSeconds / t_call), kMinIters, kMaxIters).
// A t_call that is not a positive finite number (a zero event reading) gives
// kMaxIters.
int adaptive_iterations(double warm_call_us);

// PERFMON_ITERATIONS: unset or empty returns 0 (adaptive). Otherwise a
// positive decimal integer <= kMaxIterationsOverride; anything else throws
// std::invalid_argument, so a typo never silently falls back to adaptive.
int parse_iterations_override(const char* value);

}  // namespace perfmon

#endif  // PERFMON_CORE_ITERATIONS_H
