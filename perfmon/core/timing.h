// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Times n calls of a Measurement's launch(), n from iterations.h.
//
// Iteration count. After the untimed first launch (the runner's), one warm
// call is timed with the same event pair and L2 flush as an iteration; its
// time is t_call. n = clamp(ceil(kIterationBudgetSeconds / t_call),
// kMinIters, kMaxIters) = clamp(ceil(10 s / t_call), 20, 100): calls of 100 ms
// or less keep 100 iterations, slower ones get fewer, down to 20. Large
// workloads are stable (the phase-1 report found 20 samples enough at seqlen
// 16384). PERFMON_ITERATIONS=<n> (1..kMaxIterationsOverride) fixes n and
// bypasses the rule; it is rejected, not ignored, when malformed. In
// hipgraph_ev100 the graph holds n x (flush, 2 events, the launch's kernels),
// so n = 5000 is a graph of 20000+ nodes: it captured and ran on hardware (see
// the commit), but capture time grows with n.
//
// hipgraph_ev100 (the default): one flat hipGraph capture of
//   [1 GiB L2-flush memset, start event, launch, end event] x n
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

#include "iterations.h"
#include "perfmon_abi.h"

#include <cstddef>
#include <string>
#include <vector>

namespace perfmon {

constexpr int kWarmupIters = 10;
// Matches triton.testing.do_bench. Flushed before every timed iteration, in
// both methods, outside the iteration's event pair.
constexpr size_t kL2FlushBytes = size_t(1) << 30;

struct TimingResult {
  std::string method;               // "hipgraph_ev100" or "stream_ev"
  std::vector<double> samples_us;   // one sample per iteration, microseconds
  double warm_call_us = 0.0;        // the warm call that sized the loop
  bool iterations_fixed = false;    // n came from PERFMON_ITERATIONS
};

// Throws std::runtime_error on any HIP or launch() failure or a malformed
// PERFMON_ITERATIONS.
TimingResult time_launches(Measurement& measurement, hipStream_t stream);

}  // namespace perfmon

#endif  // PERFMON_CORE_TIMING_H
