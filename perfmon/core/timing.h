// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Times kTimingIters calls of a Measurement's launch().
//
// hipgraph_ev100 (the default): one flat hipGraph capture of
//   [1 GiB L2-flush memset, start event, launch, end event] x kTimingIters
// launched once to warm up and once timed; each sample is one iteration's
// hipEventElapsedTime. Capturing removes host launch cost, which cannot
// otherwise be separated from GPU time. Verified on gfx942 for every forward
// and backward backend.
//
// stream_ev: the same iterations issued straight on the stream, after
// kWarmupIters warm-up calls. It is a cross-check, selected with
// PERFMON_TIMING_METHOD=stream_ev. It measured within a couple of percent of
// hipgraph_ev100 on gfx942. It is never a silent fallback: when hipgraph_ev100
// is configured and the capture fails at any step, time_launches throws and
// the measurement is an error. The method is recorded with every measurement.

#ifndef PERFMON_CORE_TIMING_H
#define PERFMON_CORE_TIMING_H

#include "perfmon_abi.h"

#include <cstddef>
#include <string>
#include <vector>

namespace perfmon {

constexpr int kTimingIters = 100;
constexpr int kWarmupIters = 10;
// Matches triton.testing.do_bench. Flushed before every timed iteration, in
// both methods, outside the iteration's event pair.
constexpr size_t kL2FlushBytes = size_t(1) << 30;

struct TimingResult {
  std::string method;               // "hipgraph_ev100" or "stream_ev"
  std::vector<double> samples_us;   // kTimingIters samples, microseconds
};

// Throws std::runtime_error on any HIP or launch() failure.
TimingResult time_launches(Measurement& measurement, hipStream_t stream);

}  // namespace perfmon

#endif  // PERFMON_CORE_TIMING_H
