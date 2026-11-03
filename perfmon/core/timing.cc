// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "timing.h"

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace perfmon {

namespace {

// One buffer for the life of the runner, which serves many measurements.
void* flush_buffer() {
  static void* buf = [] {
    void* p = nullptr;
    hipError_t err = hipMalloc(&p, kL2FlushBytes);
    if (err != hipSuccess) {
      throw std::runtime_error(std::string("timing: hipMalloc(flush buffer) failed: ") +
                               hipGetErrorString(err));
    }
    return p;
  }();
  return buf;
}

void check(hipError_t err, const char* what) {
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("timing: ") + what + " failed: " +
                             hipGetErrorString(err));
  }
}

// launch() throws; say which step of the timing it was in.
void launch_checked(Measurement& measurement, hipStream_t stream, const char* what) {
  try {
    measurement.launch(stream);
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("timing: ") + what + " failed: " + e.what());
  }
}

// Events and graph of one attempt, released on every exit path. Destroy
// failures are ignored so they cannot mask the error being reported.
struct Scratch {
  std::vector<hipEvent_t> start_ev;
  std::vector<hipEvent_t> end_ev;
  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;

  explicit Scratch(int n) : start_ev(n, nullptr), end_ev(n, nullptr) {
    for (int i = 0; i < n; ++i) {
      check(hipEventCreateWithFlags(&start_ev[i], hipEventDefault), "hipEventCreateWithFlags");
      check(hipEventCreateWithFlags(&end_ev[i], hipEventDefault), "hipEventCreateWithFlags");
    }
  }
  ~Scratch() {
    if (graph_exec) static_cast<void>(hipGraphExecDestroy(graph_exec));
    if (graph) static_cast<void>(hipGraphDestroy(graph));
    for (auto e : start_ev) if (e) static_cast<void>(hipEventDestroy(e));
    for (auto e : end_ev) if (e) static_cast<void>(hipEventDestroy(e));
  }

  std::vector<double> elapsed_us() const {
    const size_t n = start_ev.size();
    std::vector<double> out(n);
    for (size_t i = 0; i < n; ++i) {
      float ms = 0.0f;
      check(hipEventElapsedTime(&ms, start_ev[i], end_ev[i]), "hipEventElapsedTime");
      out[i] = static_cast<double>(ms) * 1000.0;
    }
    return out;
  }
};

// Any failure, including a refused or failed stream capture, throws with the
// step that failed. There is no fallback: stream_ev is a cross-check chosen
// explicitly with PERFMON_TIMING_METHOD=stream_ev, never a silent substitute
// for a measurement that could not be captured.
//
// Two details are load-bearing:
//  * The events are recorded with hipEventRecordExternal. A default-flag
//    record during capture is only a dependency marker, not a graph node, and
//    every elapsed-time read then fails.
//  * Each iteration has its own start and end event. A shared event between
//    two iterations is recorded twice and keeps the later timestamp, which
//    folds the next iteration's 1 GiB flush into the sample.
std::vector<double> run_hipgraph_ev100(Measurement& measurement, hipStream_t stream, int iters) {
  void* flush_buf = flush_buffer();
  Scratch s(iters);

  check(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal),
        "hipStreamBeginCapture (hipgraph_ev100)");
  hipError_t rec_err = hipSuccess;
  std::string launch_error;
  for (int i = 0; i < iters; ++i) {
    rec_err = hipMemsetAsync(flush_buf, 0, kL2FlushBytes, stream);
    if (rec_err != hipSuccess) break;
    rec_err = hipEventRecordWithFlags(s.start_ev[i], stream, hipEventRecordExternal);
    if (rec_err != hipSuccess) break;
    try {
      measurement.launch(stream);
    } catch (const std::exception& e) {
      launch_error = e.what();
      break;
    }
    rec_err = hipEventRecordWithFlags(s.end_ev[i], stream, hipEventRecordExternal);
    if (rec_err != hipSuccess) break;
  }
  // Always end the capture, so the stream is left usable for the next command.
  hipError_t end_err = hipStreamEndCapture(stream, &s.graph);
  if (!launch_error.empty()) {
    throw std::runtime_error("timing: launch (during capture) failed: " + launch_error);
  }
  check(rec_err, "capturing the timed loop");
  check(end_err, "hipStreamEndCapture (hipgraph_ev100)");

  // The graph must hold the 2N event nodes asked for, and more non-event
  // nodes than flushes. The second catches an adapter that launched on the
  // null stream instead of the one it was given: its work is then missing
  // from the graph, and the samples would time nothing.
  size_t num_nodes = 0;
  check(hipGraphGetNodes(s.graph, nullptr, &num_nodes), "hipGraphGetNodes");
  std::vector<hipGraphNode_t> nodes(num_nodes);
  check(hipGraphGetNodes(s.graph, nodes.data(), &num_nodes), "hipGraphGetNodes");
  size_t event_nodes = 0;
  for (auto n : nodes) {
    hipGraphNodeType t;
    check(hipGraphNodeGetType(n, &t), "hipGraphNodeGetType");
    if (t == hipGraphNodeTypeEventRecord) ++event_nodes;
  }
  if (event_nodes != static_cast<size_t>(2 * iters)) {
    throw std::runtime_error("timing: the captured graph has " + std::to_string(event_nodes) +
                             " event nodes, expected " + std::to_string(2 * iters));
  }
  if (num_nodes - event_nodes <= static_cast<size_t>(iters)) {
    throw std::runtime_error("timing: launch() enqueued no work on the stream it was given");
  }

  check(hipGraphInstantiate(&s.graph_exec, s.graph, nullptr, nullptr, 0), "hipGraphInstantiate");
  check(hipGraphLaunch(s.graph_exec, stream), "hipGraphLaunch (warm-up)");
  check(hipStreamSynchronize(stream), "hipStreamSynchronize (warm-up)");
  check(hipGraphLaunch(s.graph_exec, stream), "hipGraphLaunch (timed)");
  check(hipStreamSynchronize(stream), "hipStreamSynchronize (timed)");
  return s.elapsed_us();
}

// The flush sits before each start event, as in hipgraph_ev100 and do_bench.
// An earlier batched variant (two events around 100 calls) kept the flush
// inside the window and was off by +0.21 ms per sample on gfx942, i.e. +937%
// at seqlen 128.
std::vector<double> run_stream_ev(Measurement& measurement, hipStream_t stream, int iters) {
  void* flush_buf = flush_buffer();
  Scratch s(iters);
  for (int i = 0; i < kWarmupIters; ++i) {
    check(hipMemsetAsync(flush_buf, 0, kL2FlushBytes, stream), "hipMemsetAsync (L2 flush)");
    launch_checked(measurement, stream, "launch (warm-up)");
  }
  check(hipStreamSynchronize(stream), "hipStreamSynchronize (warm-up)");
  for (int i = 0; i < iters; ++i) {
    check(hipMemsetAsync(flush_buf, 0, kL2FlushBytes, stream), "hipMemsetAsync (L2 flush)");
    check(hipEventRecord(s.start_ev[i], stream), "hipEventRecord");
    launch_checked(measurement, stream, "launch");
    check(hipEventRecord(s.end_ev[i], stream), "hipEventRecord");
  }
  check(hipStreamSynchronize(stream), "hipStreamSynchronize (timed)");
  return s.elapsed_us();
}

// One iteration of the timed loop, issued on the stream and read back: the
// flush outside the event pair, as in both methods. The caller has already
// made the launch's first (untimed) call.
double time_warm_call(Measurement& measurement, hipStream_t stream) {
  void* flush_buf = flush_buffer();
  Scratch s(1);
  check(hipMemsetAsync(flush_buf, 0, kL2FlushBytes, stream), "hipMemsetAsync (L2 flush, warm call)");
  check(hipEventRecord(s.start_ev[0], stream), "hipEventRecord");
  launch_checked(measurement, stream, "launch (warm call)");
  check(hipEventRecord(s.end_ev[0], stream), "hipEventRecord");
  check(hipStreamSynchronize(stream), "hipStreamSynchronize (warm call)");
  return s.elapsed_us()[0];
}

}  // namespace

TimingResult time_launches(Measurement& measurement, hipStream_t stream) {
  const char* forced = std::getenv("PERFMON_TIMING_METHOD");
  const std::string want = forced ? forced : "";
  if (!want.empty() && want != "hipgraph_ev100" && want != "stream_ev") {
    throw std::runtime_error("PERFMON_TIMING_METHOD must be 'hipgraph_ev100' or 'stream_ev', got '" +
                             want + "'");
  }
  int iters = 0;
  try {
    iters = parse_iterations_override(std::getenv("PERFMON_ITERATIONS"));
  } catch (const std::invalid_argument& e) {
    throw std::runtime_error(e.what());
  }
  TimingResult r;
  r.warm_call_us = time_warm_call(measurement, stream);
  r.iterations_fixed = iters != 0;
  if (iters == 0) iters = adaptive_iterations(r.warm_call_us);
  if (want == "stream_ev") {
    r.samples_us = run_stream_ev(measurement, stream, iters);
    r.method = "stream_ev";
  } else {
    r.samples_us = run_hipgraph_ev100(measurement, stream, iters);
    r.method = "hipgraph_ev100";
  }
  return r;
}

}  // namespace perfmon
