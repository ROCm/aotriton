// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Assumes <hip/hip_bfloat16.h>'s hip_bfloat16 converts to and from float,
// as it does since ROCm 6; only from_float and the narrowed check need to
// change if a ROCm spells it differently.

#include "fill.h"
#include "perfmon_abi.h"

#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace perfmon {

namespace {

// splitmix64: cheap and good enough. perfmon checks no accuracy; this only has
// to keep garbage out of the timed buffers and be reproducible per seed. A pure
// function of its input, so each thread fills its element independently.
__device__ __host__ inline uint64_t splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  x = x ^ (x >> 31);
  return x;
}

// Uniform float in [-1, 1) from a 64-bit draw (top 24 bits -> plenty of
// precision for an fp16/bf16-scale destination).
__device__ inline float uniform_m1_1(uint64_t bits) {
  const uint32_t top24 = static_cast<uint32_t>(bits >> 40);
  const float u01 = static_cast<float>(top24) / static_cast<float>(1u << 24);  // [0,1)
  return u01 * 2.0f - 1.0f;  // [-1,1)
}

__device__ inline bool is_finite_normal_or_zero(float v) {
  if (!isfinite(v)) return false;
  if (v == 0.0f) return true;
  return isnormal(v);
}

template <typename T>
__device__ inline T from_float(float v);

template <>
__device__ inline __half from_float<__half>(float v) {
  return __float2half(v);
}

template <>
__device__ inline hip_bfloat16 from_float<hip_bfloat16>(float v) {
  return hip_bfloat16(v);
}

template <>
__device__ inline float from_float<float>(float v) {
  return v;
}

// Re-checks finiteness/subnormality AFTER narrowing to the target dtype,
// not just on the fp32 draw: fp16/bf16 have a far smaller normal range
// than fp32, so a perfectly ordinary fp32 value can still land on a
// denormal or +-inf once narrowed.
__device__ inline bool narrowed_is_finite_normal_or_zero(const __half& v) {
  return is_finite_normal_or_zero(__half2float(v));
}
__device__ inline bool narrowed_is_finite_normal_or_zero(const hip_bfloat16& v) {
  return is_finite_normal_or_zero(static_cast<float>(v));
}
__device__ inline bool narrowed_is_finite_normal_or_zero(const float& v) {
  return is_finite_normal_or_zero(v);
}

// Bounded retry budget for the guard pass. In practice a redraw almost
// always succeeds immediately -- this generator draws directly in
// [-1, 1) rather than reinterpreting raw bit patterns, so NaN/inf/
// denormal outcomes are rare, not the common case a raw-bits fill would
// hit. The bound exists so the kernel's worst-case runtime is fixed
// regardless of dtype/scale, not to tolerate a systematically bad rate.
constexpr int kMaxGuardRetries = 8;

template <typename T>
__global__ void fill_uniform_scaled_kernel(T* dst, int64_t count, float scale, uint64_t seed) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;

  uint64_t counter = static_cast<uint64_t>(i);
  T value{};
  for (int attempt = 0; attempt < kMaxGuardRetries; ++attempt) {
    const uint64_t bits = splitmix64(seed ^ splitmix64(counter));
    const float raw = uniform_m1_1(bits) * scale;
    value = from_float<T>(raw);
    if (narrowed_is_finite_normal_or_zero(value)) {
      dst[i] = value;
      return;
    }
    counter = splitmix64(counter + 0x9E3779B97F4A7C15ULL);  // re-mix, not just increment
  }
  // Retry budget exhausted: fail closed to exact zero rather than risk
  // writing a NaN/inf/denormal -- zero is finite/normal-or-zero by
  // definition and is the same value fill_zero() legitimately produces
  // for dq_acc/outputs.
  dst[i] = from_float<T>(0.0f);
}

int compute_grid(int64_t count, int block) {
  return static_cast<int>((count + block - 1) / block);
}

}  // namespace

void fill_uniform_scaled(void* dst, int64_t count, ElementType type, int32_t scale_dim, uint64_t seed,
                          hipStream_t stream) {
  const float scale = 1.0f / std::sqrt(static_cast<float>(scale_dim));
  constexpr int kBlock = 256;
  const int grid = compute_grid(count, kBlock);
  switch (type) {
    case ElementType::Float16:
      hipLaunchKernelGGL((fill_uniform_scaled_kernel<__half>), dim3(grid), dim3(kBlock), 0,
                          stream, static_cast<__half*>(dst), count, scale, seed);
      break;
    case ElementType::BFloat16:
      hipLaunchKernelGGL((fill_uniform_scaled_kernel<hip_bfloat16>), dim3(grid), dim3(kBlock), 0,
                          stream, static_cast<hip_bfloat16*>(dst), count, scale, seed);
      break;
    case ElementType::Float32:
      hipLaunchKernelGGL((fill_uniform_scaled_kernel<float>), dim3(grid), dim3(kBlock), 0,
                          stream, static_cast<float*>(dst), count, scale, seed);
      break;
    default:
      throw std::invalid_argument("fill_uniform_scaled: unknown ElementType: " +
                                   std::to_string(static_cast<int32_t>(type)));
  }
}

void fill_zero(void* dst, size_t bytes, hipStream_t stream) {
  hipError_t err = hipMemsetAsync(dst, 0, bytes, stream);
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("fill_zero: hipMemsetAsync failed: ") +
                              hipGetErrorString(err));
  }
}

namespace {

bool host_value_ok(__half v) {
  float f = __half2float(v);
  if (!std::isfinite(f)) return false;
  if (f == 0.0f) return true;
  return std::fpclassify(f) == FP_NORMAL;
}
bool host_value_ok(hip_bfloat16 v) {
  float f = static_cast<float>(v);
  if (!std::isfinite(f)) return false;
  if (f == 0.0f) return true;
  return std::fpclassify(f) == FP_NORMAL;
}
bool host_value_ok(float v) {
  if (!std::isfinite(v)) return false;
  if (v == 0.0f) return true;
  return std::fpclassify(v) == FP_NORMAL;
}

template <typename T>
bool verify_finite_normal_typed(const void* src, int64_t count) {
  std::vector<T> host(static_cast<size_t>(count));
  hipError_t err = hipMemcpy(host.data(), src, host.size() * sizeof(T), hipMemcpyDeviceToHost);
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("debug_verify_finite_normal: hipMemcpy failed: ") +
                              hipGetErrorString(err));
  }
  for (const T& v : host) {
    if (!host_value_ok(v)) {
      return false;
    }
  }
  return true;
}

}  // namespace

bool debug_verify_finite_normal(const void* src, int64_t count, ElementType type) {
  switch (type) {
    case ElementType::Float16:
      return verify_finite_normal_typed<__half>(src, count);
    case ElementType::BFloat16:
      return verify_finite_normal_typed<hip_bfloat16>(src, count);
    case ElementType::Float32:
      return verify_finite_normal_typed<float>(src, count);
    default:
      throw std::invalid_argument("debug_verify_finite_normal: unknown ElementType: " +
                                   std::to_string(static_cast<int32_t>(type)));
  }
}

}  // namespace perfmon
