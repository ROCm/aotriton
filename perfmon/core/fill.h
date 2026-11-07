// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Deterministic device-side input fill: the core's only device code, compiled
// for every arch the core supports.

#ifndef PERFMON_CORE_FILL_H
#define PERFMON_CORE_FILL_H

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace perfmon {

// Element types fill can write: the core's own enum, so the core needs no
// family's vocabulary. An adapter maps its dtype to it.
enum class ElementType : int32_t {
  Float16 = 0,
  BFloat16 = 1,
  Float32 = 2,
};

// Fills `count` elements of `type` at `dst` with values uniform
// in [-1, 1) times 1/sqrt(scale_dim), from a counter-based PRNG keyed by `seed`, so
// the same seed reproduces the same buffer on every release and arch.
//
// Draws that come out non-finite or subnormal after narrowing to `type` are
// redrawn in the same pass: uninitialized or denormal fp16/bf16 data moves
// timings through softmax for reasons unrelated to kernel quality.
//
// Stream-ordered on `stream`, no host synchronization.
void fill_uniform_scaled(void* dst, int64_t count, ElementType type, int32_t scale_dim,
                         uint64_t seed, hipStream_t stream);

// hipMemsetAsync to zero, for outputs and accumulators. Capturable.
void fill_zero(void* dst, size_t bytes, hipStream_t stream);

// Debugging aid, host-synchronous: true if every element is finite and either
// zero or normal.
bool debug_verify_finite_normal(const void* src, int64_t count, ElementType type);

}  // namespace perfmon

#endif  // PERFMON_CORE_FILL_H
