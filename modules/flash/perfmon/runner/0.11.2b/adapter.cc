// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Flash adapter for the 0.11 line (0.11.2b; 0.11b is a symlink: the two
// flash.h and util.h are identical).
//
// What distinguishes this API generation from 0.13b (../0.13b):
//   - LazyTensor has no `eager` member and its callbacks take the cookie:
//     acquire(void* cookie). So D and DQ_ACC are both supplied by callbacks
//     whose cookie is the context; `dispose` stays null because the arena owns
//     the buffers.
//   - no seq_strides_* tensors, and attn_options has only force_backend_index.
//   - the backends come from backends.csv, written by hand: these releases
//     publish no backends.h. build_subject.sh checks it against the internal
//     BackendEnum of the build. aiter_bwd also exists here as a separate entry
//     point, but forcing backend 2 of attn_bwd reaches the same kernels, and
//     that is the path every other release measures.
//
// AOTRITON_NS, never `aotriton::`: every subject is built with
// AOTRITON_NAME_SUFFIX=_pmon, so the namespace is aotriton_pmon.
//
// Measurement choices (the same in every adapter):
//   - only the two matmul inputs and outputs matter, no accuracy is checked, so
//     inputs are finite uniform noise and L and D are not consistent with them;
//   - the backward pass is measured alone, with generic Out, DO, L and D.

#include "../lib/common.h"

#include <perfmon/fill.h>

#include <aotriton/flash.h>
#include <aotriton/runtime.h>

#include <memory>
#include <string>

namespace {

using AOTRITON_NS::Stream;
using AOTRITON_NS::TensorView;
namespace flash = AOTRITON_NS::v3::flash;
using namespace perfmon_flash;

struct Context final : perfmon::Measurement {
  void launch(hipStream_t stream) override;

  Arena arena;
  int32_t iface = kIfaceFwd;
  flash::attn_options options;

  flash::attn_fwd_params fwd;
  // The causal tile counter. The persistent forward kernel does an atomic add
  // on it, so it must be real whenever the mask is not None, and zero at every
  // launch: a stale value makes later launches find no tiles and finish
  // instantly, a silent wrong time. Null when there is no mask: the signature
  // derives the argument away then.
  void* counter = nullptr;

  flash::attn_bwd_params bwd;
  // dq_acc is what the AITER backward accumulates into, so it is zeroed on the
  // launch stream by the acquire callback: only a launch that uses the tensor
  // pays for the memset.
  TensorView<4> dq_acc = TensorView<4>::get_null_tensor(AOTRITON_NS::kFloat32);
  size_t dq_acc_bytes = 0;
  hipStream_t launch_stream = nullptr;
  TensorView<2> delta = TensorView<2>::get_null_tensor(AOTRITON_NS::kFloat32);
};

TensorView<2> acquire_delta(void* cookie) {
  return static_cast<Context*>(cookie)->delta;
}

TensorView<4> acquire_dq_acc(void* cookie) {
  auto* ctx = static_cast<Context*>(cookie);
  if (ctx->dq_acc) perfmon::fill_zero(ctx->dq_acc.data_ptr(), ctx->dq_acc_bytes, ctx->launch_stream);
  return ctx->dq_acc;
}

void set_causal(const Causal& c, int8_t* type, int32_t* left, int32_t* right) {
  switch (c.mode) {
    case CausalMode::None:
      *type = flash::CausalType::None;
      *left = 0;
      *right = 0;
      break;
    case CausalMode::TopLeft:
      *type = flash::CausalType::WindowedAttention;
      *left = flash::WindowValue::TopLeftAligned;
      *right = flash::WindowValue::TopLeftAligned;
      break;
    case CausalMode::Window:
      *type = flash::CausalType::WindowedAttention;
      *left = c.left;
      *right = c.right;
      break;
  }
}

void build_fwd(Context* ctx, const Dims& d, const Causal& causal, hipStream_t s) {
  auto& p = ctx->fwd;
  auto* a = &ctx->arena;
  p.Q = make_bhsd(a, d, d.heads_q, d.seqlen_q, 1, false, s);
  p.K = make_bhsd(a, d, d.heads_kv, d.seqlen_k, 2, false, s);
  p.V = make_bhsd(a, d, d.heads_kv, d.seqlen_k, 3, false, s);
  p.B = d.bias_shape == perfmon::flash::BiasShape::None
            ? TensorView<4>::get_null_tensor(p.Q.dtype())
            : make_bias(a, d, 4, s);
  // p.A (varlen) stays default-constructed, as in the in-repo tuning caller.
  p.Sm_scale = d.sm_scale;
  p.L = make_lse(a, d, 5, /*zero=*/true, s);
  p.Out = make_bhsd(a, d, d.heads_q, d.seqlen_q, 6, /*zero=*/true, s);
  p.dropout_p = static_cast<float>(d.dropout_p);
  const Philox ph = make_philox(a, d);
  p.philox_seed_ptr = ph.seed;
  p.philox_offset1 = ph.offset1;
  p.philox_offset2 = ph.offset2;
  p.philox_seed_output = ph.seed_out;
  p.philox_offset_output = ph.offset_out;
  p.encoded_softmax = TensorView<4>::get_null_tensor(p.Q.dtype());
  set_causal(causal, &p.causal_type, &p.window_left, &p.window_right);
  if (p.causal_type != flash::CausalType::None) {
    ctx->counter = make_counter(a, s);
    p.persistent_atomic_counter =
        TensorView<0>(reinterpret_cast<intptr_t>(ctx->counter), AOTRITON_NS::kInt32);
  } else {
    p.persistent_atomic_counter = TensorView<0>::get_null_tensor(AOTRITON_NS::kInt32);
  }
  p.varlen_type = flash::VarlenType::None;  // the cu_seqlens_* tensors stay disabled
}

void build_bwd(Context* ctx, const Dims& d, const Causal& causal, int backend, hipStream_t s) {
  auto& p = ctx->bwd;
  auto* a = &ctx->arena;
  p.Q = make_bhsd(a, d, d.heads_q, d.seqlen_q, 1, false, s);
  p.K = make_bhsd(a, d, d.heads_kv, d.seqlen_k, 2, false, s);
  p.V = make_bhsd(a, d, d.heads_kv, d.seqlen_k, 3, false, s);
  p.B = d.bias_shape == perfmon::flash::BiasShape::None
            ? TensorView<4>::get_null_tensor(p.Q.dtype())
            : make_bias(a, d, 4, s);
  p.DB = d.bias_shape == perfmon::flash::BiasShape::None
             ? TensorView<4>::get_null_tensor(p.Q.dtype())
             : make_bias_grad(a, d, 5, s);
  p.Sm_scale = d.sm_scale;
  p.Out = make_bhsd(a, d, d.heads_q, d.seqlen_q, 6, false, s);
  p.DO = make_bhsd(a, d, d.heads_q, d.seqlen_q, 7, false, s);
  p.DK = make_bhsd(a, d, d.heads_kv, d.seqlen_k, 8, /*zero=*/true, s);
  p.DV = make_bhsd(a, d, d.heads_kv, d.seqlen_k, 9, /*zero=*/true, s);
  p.DQ = make_bhsd(a, d, d.heads_q, d.seqlen_q, 10, /*zero=*/true, s);
  p.L = make_lse(a, d, 11, false, s);
  ctx->delta = make_lse(a, d, 12, false, s);
  p.D.cookie = ctx;
  p.D.acquire = &acquire_delta;

  // Only a backend that reads dq_acc is given one; for the others the lazy
  // tensor is never acquired.
  const char* name = backend >= 0 ? backend_name(kIfaceBwd, backend) : nullptr;
  const bool needs_dq_acc = backend == kBackendUnforced || (name && std::string(name) == "aiter");
  if (needs_dq_acc) {
    ctx->dq_acc = make_dq_acc(a, d);
    ctx->dq_acc_bytes = static_cast<size_t>(d.batch * d.heads_q * d.seqlen_q * d.hdim) * sizeof(float);
  }
  p.DQ_ACC.cookie = ctx;
  p.DQ_ACC.acquire = &acquire_dq_acc;

  p.dropout_p = static_cast<float>(d.dropout_p);
  const Philox ph = make_philox(a, d);
  p.philox_seed_ptr = ph.seed;
  p.philox_offset1 = ph.offset1;
  p.philox_offset2 = ph.offset2;
  set_causal(causal, &p.causal_type, &p.window_left, &p.window_right);
  p.varlen_type = flash::VarlenType::None;
}

void Context::launch(hipStream_t stream) {
  if (iface == kIfaceFwd) {
    if (counter) zero_counter(counter, stream);
    launch_check(flash::attn_fwd(fwd, flash::attn_fwd_params::kVersion, Stream(stream),
                                 &options));
    return;
  }
  launch_stream = stream;
  launch_check(flash::attn_bwd(bwd, flash::attn_bwd_params::kVersion, Stream(stream),
                               &options));
}

}  // namespace

namespace perfmon_flash {

std::unique_ptr<perfmon::Measurement> adapter_prepare(const Request& req,
                                                          int backend,
                                                          hipStream_t scratch) {
  const Dims d = dims_of(req);
  const Causal causal = resolve_causal(req, /*has_window=*/true);
  auto ctx = std::make_unique<Context>();
  ctx->iface = req.iface;
  ctx->options.force_backend_index = backend;  // -1 is "not forced"
  if (req.iface == kIfaceFwd) {
    build_fwd(ctx.get(), d, causal, scratch);
  } else {
    build_bwd(ctx.get(), d, causal, backend, scratch);
  }
  return ctx;
}

}  // namespace perfmon_flash
