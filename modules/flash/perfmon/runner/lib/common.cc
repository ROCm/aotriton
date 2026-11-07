// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// See common.h for what belongs in this file.

#include "common.h"

#include <cmath>
#include <string>

namespace perfmon_flash {

namespace {

// Mixed into the seed of every buffer, so the buffers of one entry differ.
uint64_t mix(uint64_t seed, uint64_t salt) {
  return seed ^ (salt * 0x9E3779B97F4A7C15ull);
}

size_t dtype_bytes(perfmon::ElementType type) {
  switch (type) {
    case perfmon::ElementType::Float16:
    case perfmon::ElementType::BFloat16:
      return 2;
    case perfmon::ElementType::Float32:
      return 4;
  }
  throw std::invalid_argument("unknown element type");
}

AOTRITON_NS::DType to_aotriton_dtype(perfmon::flash::Dtype dtype) {
  switch (dtype) {
    case perfmon::flash::Dtype::Float16:
      return AOTRITON_NS::kFloat16;
    case perfmon::flash::Dtype::BFloat16:
      return AOTRITON_NS::kBFloat16;
    case perfmon::flash::Dtype::Float32:
      return AOTRITON_NS::kFloat32;
  }
  throw std::invalid_argument("unknown dtype " + std::to_string(static_cast<int32_t>(dtype)));
}

uint64_t round_up_8(uint64_t n) {
  return (n + 7) / 8 * 8;
}

// Allocates `count` elements of `dtype` and fills or zeroes them.
void* alloc_filled(Arena* arena, int64_t count, perfmon::ElementType type, int32_t hdim,
                   uint64_t seed, bool zero, hipStream_t stream) {
  void* p = arena->alloc(static_cast<size_t>(count) * dtype_bytes(type));
  if (zero) {
    perfmon::fill_zero(p, static_cast<size_t>(count) * dtype_bytes(type), stream);
  } else {
    perfmon::fill_uniform_scaled(p, count, type, hdim, seed, stream);
  }
  return p;
}

TensorView<4> bias_impl(Arena* arena, const Dims& d, uint64_t salt, bool zero,
                        hipStream_t stream) {
  const bool real_batch = d.bias_shape == perfmon::flash::BiasShape::BHQK ||
                          d.bias_shape == perfmon::flash::BiasShape::B1QK;
  const bool real_head = d.bias_shape == perfmon::flash::BiasShape::BHQK;
  const uint64_t bb = real_batch ? d.batch : 1;
  const uint64_t hb = real_head ? d.heads_q : 1;
  const uint64_t padded_sk = round_up_8(d.seqlen_k);
  const uint64_t sq = d.seqlen_q;

  // The memory is (bb, hb, sq, padded_sk), or (bb, sq, hb, padded_sk) when
  // flipped; the logical view is (B, H, Sq, Sk) either way.
  uint64_t stride_b, stride_h, stride_m;
  if (!d.storage_flip) {
    stride_b = hb * sq * padded_sk;
    stride_h = sq * padded_sk;
    stride_m = padded_sk;
  } else {
    stride_b = sq * hb * padded_sk;
    stride_h = padded_sk;
    stride_m = hb * padded_sk;
  }
  if (!real_batch) stride_b = 0;
  if (!real_head) stride_h = 0;

  const int64_t count = static_cast<int64_t>(bb * hb * sq * padded_sk);
  void* p = alloc_filled(arena, count, element_type(d.dtype), static_cast<int32_t>(d.hdim),
                         mix(d.seed, salt), zero, stream);
  return TensorView<4>(reinterpret_cast<intptr_t>(p),
                       {d.batch, d.heads_q, sq, d.seqlen_k},
                       {stride_b, stride_h, stride_m, 1},
                       to_aotriton_dtype(d.dtype));
}

}  // namespace

const char* backend_name(int32_t iface, int index) {
  if (iface < 0 || iface >= kIfaceCount) return nullptr;
  const BackendNames& b = kBackendNames[iface];
  if (index < 0 || index >= b.count) return nullptr;
  return b.names[index];
}

perfmon::ElementType element_type(perfmon::flash::Dtype dtype) {
  switch (dtype) {
    case perfmon::flash::Dtype::Float16:
      return perfmon::ElementType::Float16;
    case perfmon::flash::Dtype::BFloat16:
      return perfmon::ElementType::BFloat16;
    case perfmon::flash::Dtype::Float32:
      return perfmon::ElementType::Float32;
  }
  throw std::invalid_argument("unknown dtype " + std::to_string(static_cast<int32_t>(dtype)));
}

Arena::~Arena() {
  for (auto it = ptrs_.rbegin(); it != ptrs_.rend(); ++it) {
    static_cast<void>(hipFree(*it));
  }
}

void* Arena::alloc(size_t bytes) {
  void* p = nullptr;
  hip_check(hipMalloc(&p, bytes), "hipMalloc");
  ptrs_.push_back(p);
  return p;
}

void hip_check(hipError_t err, const char* what) {
  if (err != hipSuccess) {
    throw std::runtime_error(std::string(what) + " failed: " + hipGetErrorString(err));
  }
}

void launch_check(hipError_t err) {
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("the operator returned ") + hipGetErrorName(err) + ": " +
                             hipGetErrorString(err));
  }
}

Dims dims_of(const Request& req) {
  const perfmon::flash::Entry& e = req.entry;
  if (e.varlen != 0) {
    throw Unsupported("varlen entries are not measured by this adapter");
  }
  if (e.hdim_qk != e.hdim_v) {
    throw Unsupported("hdim_qk != hdim_v: the operator has one head dimension");
  }
  if (e.gqa_ratio < 1 || e.n_heads_q < 1 || e.n_heads_q % e.gqa_ratio != 0) {
    throw std::invalid_argument("n_heads_q is not a positive multiple of gqa_ratio");
  }
  if (e.hdim_qk < 1 || e.batch < 1 || e.seqlen_q < 1 || e.seqlen_k < 1) {
    throw std::invalid_argument("sizes must be positive");
  }
  if (!(e.dropout_p >= 0.0 && e.dropout_p < 1.0)) {
    throw std::invalid_argument("dropout_p must be in [0, 1)");
  }
  if ((e.bias_type != 0) != (e.bias_shape != perfmon::flash::BiasShape::None)) {
    throw std::invalid_argument("bias_type disagrees with bias_shape");
  }

  Dims d;
  d.batch = static_cast<uint64_t>(e.batch);
  d.heads_q = static_cast<uint64_t>(e.n_heads_q);
  d.heads_kv = static_cast<uint64_t>(e.n_heads_q / e.gqa_ratio);
  d.seqlen_q = static_cast<uint64_t>(e.seqlen_q);
  d.seqlen_k = static_cast<uint64_t>(e.seqlen_k);
  d.hdim = static_cast<uint64_t>(e.hdim_qk);
  d.dtype = e.dtype;
  d.storage_flip = e.storage_flip != 0;
  d.bias_shape = e.bias_shape;
  d.dropout_p = e.dropout_p;
  d.seed = req.seed;
  // The usual 1/sqrt(d). (The first harness used 1/d, which is a bug.)
  d.sm_scale = 1.0f / std::sqrt(static_cast<float>(e.hdim_qk));
  return d;
}

Causal resolve_causal(const Request& req, bool has_window) {
  const perfmon::flash::Entry& e = req.entry;
  Causal c;
  const bool open_left = e.window_left < 0;
  const bool open_right = e.window_right < 0;
  if (open_left && open_right) {
    if (e.causal) c.mode = CausalMode::TopLeft;
    return c;
  }
  if (!has_window) {
    throw Unsupported("this release has no windowed attention");
  }
  // With a causal mask the window can only look back: no keys to the right of
  // the diagonal.
  int32_t right = e.window_right;
  if (e.causal) {
    if (right > 0) throw Unsupported("causal with a window reaching right of the diagonal");
    right = 0;
  }
  if (e.window_left < 0 || right < 0) {
    // TODO(perfmon): an open side of a window. Decide what each release's
    // kernels mean by it on hardware before translating it.
    throw Unsupported("a window with one open side");
  }
  c.mode = CausalMode::Window;
  c.left = e.window_left;
  c.right = right;
  return c;
}

TensorView<4> make_bhsd(Arena* arena, const Dims& d, uint64_t heads, uint64_t seqlen,
                        uint64_t salt, bool zero, hipStream_t stream) {
  const uint64_t b = d.batch, s = seqlen, h = heads, dd = d.hdim;
  std::array<uint64_t, 4> strides;
  if (!d.storage_flip) {
    strides = {h * s * dd, s * dd, dd, 1};
  } else {
    strides = {s * h * dd, dd, h * dd, 1};
  }
  void* p = alloc_filled(arena, static_cast<int64_t>(b * h * s * dd), element_type(d.dtype),
                         static_cast<int32_t>(dd), mix(d.seed, salt), zero, stream);
  return TensorView<4>(reinterpret_cast<intptr_t>(p), {b, h, s, dd}, strides,
                       to_aotriton_dtype(d.dtype));
}

TensorView<2> make_lse(Arena* arena, const Dims& d, uint64_t salt, bool zero,
                       hipStream_t stream) {
  const uint64_t hs = d.heads_q * d.seqlen_q;
  void* p = alloc_filled(arena, static_cast<int64_t>(d.batch * hs), perfmon::ElementType::Float32,
                         static_cast<int32_t>(d.hdim), mix(d.seed, salt), zero, stream);
  return TensorView<2>(reinterpret_cast<intptr_t>(p), {d.batch, hs}, {hs, 1},
                       AOTRITON_NS::kFloat32);
}

TensorView<4> make_bias(Arena* arena, const Dims& d, uint64_t salt, hipStream_t stream) {
  return bias_impl(arena, d, salt, /*zero=*/false, stream);
}

TensorView<4> make_bias_grad(Arena* arena, const Dims& d, uint64_t salt, hipStream_t stream) {
  if (d.bias_shape == perfmon::flash::BiasShape::BHQK) {
    return bias_impl(arena, d, salt, /*zero=*/true, stream);
  }
  return TensorView<4>::get_null_tensor(to_aotriton_dtype(d.dtype));
}

TensorView<4> make_dq_acc(Arena* arena, const Dims& d) {
  const uint64_t b = d.batch, h = d.heads_q, s = d.seqlen_q, dd = d.hdim;
  void* p = arena->alloc(static_cast<size_t>(b * h * s * dd) * sizeof(float));
  // Zeroed by the acquire callback before every launch, not here.
  return TensorView<4>(reinterpret_cast<intptr_t>(p), {b, h, s, dd}, {h * s * dd, s * dd, dd, 1},
                       AOTRITON_NS::kFloat32);
}

Philox make_philox(Arena* arena, const Dims& d) {
  Philox out;
  if (d.dropout_p <= 0.0) return out;
  auto scalar = [&](uint64_t value) {
    void* p = arena->alloc(sizeof(uint64_t));
    hip_check(hipMemcpy(p, &value, sizeof(value), hipMemcpyHostToDevice), "hipMemcpy");
    return TensorView<0>(reinterpret_cast<intptr_t>(p), AOTRITON_NS::kUInt64);
  };
  out.seed = scalar(d.seed);
  out.offset1 = scalar(0);
  out.seed_out = scalar(0);
  out.offset_out = scalar(0);
  out.offset2 = 0;
  return out;
}

void* make_counter(Arena* arena, hipStream_t stream) {
  void* p = arena->alloc(sizeof(int32_t));
  perfmon::fill_zero(p, sizeof(int32_t), stream);
  return p;
}

void zero_counter(void* counter, hipStream_t stream) {
  perfmon::fill_zero(counter, sizeof(int32_t), stream);
}

}  // namespace perfmon_flash
