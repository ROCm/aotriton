// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Flash adapter for 0.10b: the first release with the v3 API, before backends
// could be forced.
//
// What distinguishes this API generation from 0.11.2b (../0.11.2b):
//   - attn_options is empty: no backend can be forced, so backends.csv has no
//     rows and only the unforced operator is measured;
//   - no LazyTensor and no DQ_ACC: D is a plain tensor, and there is no
//     accumulator to zero;
//   - kVersion is 1 for both params structs.
//
// The v2 entry points still exist in this release's header and would compile
// here. They are not used: v3 is what users get, and what the later releases
// are measured through.

#include "../lib/common.h"

#include <perfmon/fill.h>

#include <aotriton/flash.h>
#include <aotriton/runtime.h>

#include <memory>

namespace {

using AOTRITON_NS::Stream;
using AOTRITON_NS::TensorView;
namespace flash = AOTRITON_NS::v3::flash;
using namespace perfmon_flash;

struct Context final : perfmon::Measurement {
  void launch(hipStream_t stream) override;

  Arena arena;
  int32_t iface = kIfaceFwd;
  flash::attn_fwd_params fwd;
  // The causal tile counter, real and re-zeroed at every launch whenever the
  // mask is not None: see head/adapter.cc.
  void* counter = nullptr;
  flash::attn_bwd_params bwd;
};

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
  p.varlen_type = flash::VarlenType::None;
}

void build_bwd(Context* ctx, const Dims& d, const Causal& causal, hipStream_t s) {
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
  p.D = make_lse(a, d, 12, false, s);
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
    launch_check(flash::attn_fwd(fwd, flash::attn_fwd_params::kVersion, Stream(stream)));
    return;
  }
  launch_check(flash::attn_bwd(bwd, flash::attn_bwd_params::kVersion, Stream(stream)));
}

}  // namespace

namespace perfmon_flash {

std::unique_ptr<perfmon::Measurement> adapter_prepare(const Request& req,
                                                          int /*backend: always unforced here*/,
                                                          hipStream_t scratch) {
  const Dims d = dims_of(req);
  const Causal causal = resolve_causal(req, /*has_window=*/true);
  auto ctx = std::make_unique<Context>();
  ctx->iface = req.iface;
  if (req.iface == kIfaceFwd) {
    build_fwd(ctx.get(), d, causal, scratch);
  } else {
    build_bwd(ctx.get(), d, causal, scratch);
  }
  return ctx;
}

}  // namespace perfmon_flash
