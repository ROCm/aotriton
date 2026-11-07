// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Flash adapter for 0.9.2b: the only release before the v3 API.
//
// What distinguishes it:
//   - no params struct: the entry points are long positional functions in
//     AOTRITON_NS::v2::flash, and the arguments are held as members here;
//   - the mask is a plain `bool is_causal`: there is no windowed attention, so
//     an entry with a window is unsupported;
//   - no backend can be forced (backends.csv has no rows), so only the
//     unforced operator is measured. attn_bwd is that operator; attn_bwd_fused
//     is a second entry point that users reach only by calling it directly;
//   - `atomic_for_causal` is this release's persistent_atomic_counter, with the
//     same rule: real and re-zeroed at every launch whenever causal;
//   - D is a plain tensor ("empty_like(softmax_lse)").
//
// GQA is supported by the kernels, but their source notes that a bias is
// incompatible with it, so that combination is reported unsupported.
//
// Do not reuse this adapter for a later release. It compiles against 0.10b and
// 0.11.2b too, because they still ship the v2 API next to v3, but it would then
// silently measure the v2 entry point of a release whose v3 path is what users
// get.

#include "../lib/common.h"

#include <perfmon/fill.h>

#include <aotriton/flash.h>
#include <aotriton/runtime.h>

#include <memory>

namespace {

using AOTRITON_NS::Stream;
using AOTRITON_NS::TensorView;
namespace flash = AOTRITON_NS::v2::flash;
using namespace perfmon_flash;

struct Context final : perfmon::Measurement {
  void launch(hipStream_t stream) override;

  Arena arena;
  int32_t iface = kIfaceFwd;
  bool is_causal = false;
  float sm_scale = 1.0f;
  float dropout_p = 0.0f;
  Philox philox;

  TensorView<4> q, k, v, b, out;
  TensorView<2> lse;
  // forward only
  TensorView<4> encoded_softmax;
  void* counter = nullptr;
  TensorView<0> atomic_for_causal = TensorView<0>::get_null_tensor(AOTRITON_NS::kInt32);
  // backward only
  TensorView<4> dout, dq, dk, dv, db;
  TensorView<2> delta;
};

void build_fwd(Context* ctx, const Dims& d, hipStream_t s) {
  auto* a = &ctx->arena;
  ctx->q = make_bhsd(a, d, d.heads_q, d.seqlen_q, 1, false, s);
  ctx->k = make_bhsd(a, d, d.heads_kv, d.seqlen_k, 2, false, s);
  ctx->v = make_bhsd(a, d, d.heads_kv, d.seqlen_k, 3, false, s);
  ctx->b = d.bias_shape == perfmon::flash::BiasShape::None
               ? TensorView<4>::get_null_tensor(ctx->q.dtype())
               : make_bias(a, d, 4, s);
  ctx->lse = make_lse(a, d, 5, /*zero=*/true, s);
  ctx->out = make_bhsd(a, d, d.heads_q, d.seqlen_q, 6, /*zero=*/true, s);
  ctx->encoded_softmax = TensorView<4>::get_null_tensor(ctx->q.dtype());
  if (ctx->is_causal) {
    ctx->counter = make_counter(a, s);
    ctx->atomic_for_causal =
        TensorView<0>(reinterpret_cast<intptr_t>(ctx->counter), AOTRITON_NS::kInt32);
  }
}

void build_bwd(Context* ctx, const Dims& d, hipStream_t s) {
  auto* a = &ctx->arena;
  ctx->q = make_bhsd(a, d, d.heads_q, d.seqlen_q, 1, false, s);
  ctx->k = make_bhsd(a, d, d.heads_kv, d.seqlen_k, 2, false, s);
  ctx->v = make_bhsd(a, d, d.heads_kv, d.seqlen_k, 3, false, s);
  ctx->b = d.bias_shape == perfmon::flash::BiasShape::None
               ? TensorView<4>::get_null_tensor(ctx->q.dtype())
               : make_bias(a, d, 4, s);
  ctx->db = d.bias_shape == perfmon::flash::BiasShape::None
                ? TensorView<4>::get_null_tensor(ctx->q.dtype())
                : make_bias_grad(a, d, 5, s);
  ctx->out = make_bhsd(a, d, d.heads_q, d.seqlen_q, 6, false, s);
  ctx->dout = make_bhsd(a, d, d.heads_q, d.seqlen_q, 7, false, s);
  ctx->dk = make_bhsd(a, d, d.heads_kv, d.seqlen_k, 8, /*zero=*/true, s);
  ctx->dv = make_bhsd(a, d, d.heads_kv, d.seqlen_k, 9, /*zero=*/true, s);
  ctx->dq = make_bhsd(a, d, d.heads_q, d.seqlen_q, 10, /*zero=*/true, s);
  ctx->lse = make_lse(a, d, 11, false, s);
  ctx->delta = make_lse(a, d, 12, false, s);
}

void Context::launch(hipStream_t stream) {
  if (iface == kIfaceFwd) {
    if (counter) zero_counter(counter, stream);
    launch_check(flash::attn_fwd(q, k, v, b, sm_scale, lse, out, dropout_p,
                                 philox.seed, philox.offset1,
                           static_cast<int64_t>(philox.offset2), philox.seed_out,
                           philox.offset_out, encoded_softmax, is_causal,
                           atomic_for_causal, Stream(stream)));
    return;
  }
  launch_check(flash::attn_bwd(q, k, v, b, sm_scale, out, dout, dq, dk,
                               dv, db, lse, delta, dropout_p, philox.seed,
                         philox.offset1, static_cast<int64_t>(philox.offset2),
                         is_causal, Stream(stream)));
}

}  // namespace

namespace perfmon_flash {

std::unique_ptr<perfmon::Measurement> adapter_prepare(const Request& req,
                                                          int /*backend: always unforced here*/,
                                                          hipStream_t scratch) {
  const Dims d = dims_of(req);
  const Causal causal = resolve_causal(req, /*has_window=*/false);
  if (d.heads_q != d.heads_kv && d.bias_shape != perfmon::flash::BiasShape::None) {
    throw Unsupported("this release's kernels do not support a bias together with GQA");
  }
  auto ctx = std::make_unique<Context>();
  ctx->iface = req.iface;
  ctx->is_causal = causal.mode != CausalMode::None;
  ctx->sm_scale = d.sm_scale;
  ctx->dropout_p = static_cast<float>(d.dropout_p);
  ctx->philox = make_philox(&ctx->arena, d);
  if (req.iface == kIfaceFwd) {
    build_fwd(ctx.get(), d, scratch);
  } else {
    build_bwd(ctx.get(), d, scratch);
  }
  return ctx;
}

}  // namespace perfmon_flash
