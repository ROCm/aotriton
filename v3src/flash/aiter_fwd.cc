// Copyright © 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/config.h>
#include <aotriton/_internal/util.h>
#include <aotriton/_internal/flash/aiter.h>
#include <aotriton/flash.h>
#include <aotriton/util.h>
#include <aotriton/_internal/lazy_tensor_internal.h>
#include <flash/iface.op_attn_fwd.h>
#include <flash/affine.aiter_fmha_v3_fwd.h>
#include <algorithm>
#include <limits>
#ifndef NDEBUG
#include <iostream>
#include <stdio.h>
#endif

namespace AOTRITON_NS::v3::flash {

static aiter::mha_fwd_args
construct_mha_fwd_args(const AiterFmhaV3FwdContext& ctx);

const char*
AiterFmhaV3FwdContext::check_inputs_are_supported(Gpu gpu) {
  const auto& args = *params;
#define RETURN_IF(COND)                                               \
  do {                                                                \
    if (COND) {                                                       \
      return "Input unsupported due to " STRINGIFICATION(COND);       \
    }                                                                 \
  } while(0)
  // AITER ASM does not support dropout
  RETURN_IF(args.ENABLE_DROPOUT);
  // No bias support
  RETURN_IF(args.BIAS_TYPE);
  // FIXME: Varlen support disabled for now
  RETURN_IF(args.Num_seqlens != 0);
  // GQA only supported in varlen (aka. group mode)
  RETURN_IF(args.Num_head_q != args.Num_head_k);
  // GQA only supported in varlen (aka. group mode)
  if (args.Num_head_q != args.Num_head_k) {
    RETURN_IF(!args.cu_seqlens_q || !*args.cu_seqlens_q);
    RETURN_IF(!args.cu_seqlens_k || !*args.cu_seqlens_k);
  }
  // Must provide cu_seqlens_q/k at the same time
  if (args.cu_seqlens_q && *args.cu_seqlens_q) {
    RETURN_IF(!args.cu_seqlens_k || !*args.cu_seqlens_k);
  }
  // Only support unpadded (192, 128) and (128, 128)
  RETURN_IF(args.Hdim_qk != 128 && args.Hdim_qk != 192);
  RETURN_IF(args.Hdim_vo != 128);
  // ASM FWD only support TopLeftAligned
  if (args.CAUSAL_TYPE != CausalType::None) {
    // Invalid assignment
    RETURN_IF(args.Window_left != args.Window_right);
    if (args.Window_left != WindowValue::TopLeftAligned) {
#ifndef NDEBUG
      std::cerr << "Input unsupported due to args.CAUSAL_TYPE = " << int(args.CAUSAL_TYPE) << " and "
                << " args.Window_left = " << args.Window_left
                << " args.Window_right = " << args.Window_right
                << std::endl;
#endif
      return "Input unsupported due to BottomRightAligned/SWA";
    }
  }
#undef RETURN_IF
  // AITER ASM kernel only reads u32 strides.
#define CHECK_STRIDE(T)                                               \
  do {                                                                \
    auto strides = T->strides();                                      \
    size_t max_e = *std::max_element(strides.begin(), strides.end()); \
    if (max_e * 2 > std::numeric_limits<uint32_t>::max()) {           \
      return "Input unsupported due to large tensor " STRINGIFICATION(T);           \
    }                                                                 \
  } while(0)
#if 0
      std::cerr << "Input unsupported due to large tensor " << #T << std::endl;
      std::cerr << "strides: "; for (auto s : strides) std::cerr << s << " "; std::cerr << std::endl;
      std::cerr << "max_e * 2: " << max_e * 2 << std::endl;
#endif
  CHECK_STRIDE(args.Q);
  CHECK_STRIDE(args.K);
  CHECK_STRIDE(args.V);
  CHECK_STRIDE(args.Out);
#undef CHECK_STRIDE
  cookie = construct_mha_fwd_args(*this);
  cookie.v3_api_check = true;
  AOTRITON_NS::v3::aiter::ck_tile::stream_config sc {
    .gpu_ = gpu,
  };
  if (fmha_fwd_v3(cookie, sc) != 0)
    return "v3_api_check report failure";

  return nullptr;
}

// Too many narrowing warning here.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnarrowing"

aiter::mha_fwd_args
static construct_mha_fwd_args(const AiterFmhaV3FwdContext& ctx) {
  const auto& args = *ctx.params;
  bool has_lse = *args.L;
  bool is_group_mode = args.cu_seqlens_q->data_ptr();
  auto batch = args.Q->size(0);
  auto nhead_q = args.Q->size(1);
  auto nhead_k = args.K->size(1);
  auto hdim_qk = args.Q->size(3);
  auto hdim_vo = args.V->size(3);
  auto scale = args.Sm_scale;
  auto stride_q = args.Q->stride(2);
  auto stride_k = args.K->stride(2);
  auto stride_v = args.V->stride(2);
  auto stride_o = args.Out->stride(2);

  auto nhead_stride_q = args.Q->stride(1);
  auto nhead_stride_k = args.K->stride(1);
  auto nhead_stride_v = args.V->stride(1);
  auto nhead_stride_o = args.Out->stride(1);
  // FIXME: Use Rank-3 tensor for LSE
  // then:
  // nhead_stride_lsed = args.L->stride(1);
  // batch_stride_lsed = args.L->stride(0);
  auto seqlen_q = args.Q->size(2);
  auto nhead_stride_lsed = seqlen_q;

  auto batch_stride_q = args.Q->stride(0);
  auto batch_stride_k = args.K->stride(0);
  auto batch_stride_v = args.V->stride(0);
  auto batch_stride_o = args.Out->stride(0);
  auto batch_stride_lse = nhead_q * seqlen_q;

  auto data_type = [&args]() {
    if (args.Q->dtype() == DType::kFloat16)
      return "fp16";
    return "bf16";
  };
  auto [mask_type, window_size_left, window_size_right] = [&args]() -> std::tuple<int, int, int> {
    if (args.CAUSAL_TYPE == CausalType::None)
      return {0, -1, -1};
    if (args.Window_left == WindowValue::TopLeftAligned)
      return {1, -1, 0};
    if (args.Window_left == WindowValue::BottomRightAligned)
      return {2, -1, 0};
    return {3, args.Window_left, args.Window_right};
  }();

  auto pointer_with_default = [](const void* pref, const void* def) {
    return pref ? pref : def;
  };
  auto seqstart_q_ptr = pointer_with_default(args.seq_strides_q->data_ptr(), args.cu_seqlens_q->data_ptr());
  auto seqstart_k_ptr = pointer_with_default(args.seq_strides_k->data_ptr(), args.cu_seqlens_k->data_ptr());

  // TODO: use .v3_api_check for lookup_optimal
  aiter::mha_fwd_args ret = {
    // aiter args
    .use_asm_v3           = true,                                               // bool
    .v3_api_check         = false,                                              // bool
    .how_v3_bf16_cvt      = 0,                                                  // int, 0/1/2: rtne/rtna/rtz
    // from ck fmha_fwd_traits
    .data_type            = data_type(),                                        // std::string
    .is_group_mode        = is_group_mode,                                      // bool
    .bias_type            = args.BIAS_TYPE,                                     // int
    .has_lse              = has_lse,                                            // bool
    .qscale_type          = 0,                                                  // bool
    // .has_sink          = false,                                              // bool
    // From ck  fmha_fwd_args
    .q_ptr                = args.Q->data_ptr(),                                 // const void*
    .k_ptr                = args.K->data_ptr(),                                 // const void*
    .v_ptr                = args.V->data_ptr(),                                 // const void*
    .bias_ptr             = args.B->data_ptr(),                                 // const void*
    // .q_descale_ptr     = nullptr,                                            // const void*
    // .k_descale_ptr     = nullptr,                                            // const void*
    // .v_descale_ptr     = nullptr,                                            // const void*
    .rand_val_ptr         = nullptr,                                            // void*
    .lse_ptr              = args.L->data_ptr(),                                 // const void*
    .o_ptr                = args.Out->data_ptr(),                               // const void*
    // Key precedence difference from triton kernel:
    // Triton:
    //   logical cu_seqlen -> physical cu_seqlen if supplied
    // ASM:
    //   physical cu_seqlen -> logical cu_seqlen if supplied
    .seqstart_q_ptr       = seqstart_q_ptr,                                     // const void*  // "Physical" cu_seqlen
    .seqstart_k_ptr       = seqstart_k_ptr,                                     // const void*  // "Physical" cu_seqlen
    .seqlen_q_ptr         = nullptr,                                            // const void*  // unused in ASM
    .seqlen_k_ptr         = nullptr,                                            // const void*  // unused in ASM
    .cu_seqlen_q_ptr      = args.cu_seqlens_q->data_ptr(),                      // const void*  // "Logical" cu_seqlen
    .cu_seqlen_k_ptr      = args.cu_seqlens_k->data_ptr(),                      // const void*  // "Logical" cu_seqlen
    // .block_scale_seqstart_q_ptr     = nullptr,                               // const void*
    // .block_scale_seqstart_k_ptr     = nullptr,                               // const void*
    // .sink_ptr          = nullptr,                                            // const void*
    .seqlen_q             = args.Max_seqlen_q,                                  // int
    .seqlen_k             = args.Max_seqlen_k,                                  // int
    .batch                = batch,                                              // int
    .max_seqlen_q         = args.Max_seqlen_q,                                  // int
    .hdim_q               = hdim_qk,                                            // int
    .hdim_v               = hdim_vo,                                            // int
    .nhead_q              = nhead_q,                                            // int
    .nhead_k              = nhead_k,                                            // int
    .scale_s              = scale,                                              // float
    .logits_soft_cap      = 0.0,                                                // float
    .stride_q             = stride_q,                                           // int
    .stride_k             = stride_k,                                           // int
    .stride_v             = stride_v,                                           // int
    .stride_bias          = 0,                                                  // int
    .stride_randval       = 0,                                                  // int
    .stride_o             = stride_o,                                           // int
    .nhead_stride_q       = nhead_stride_q,                                     // int
    .nhead_stride_k       = nhead_stride_k,                                     // int
    .nhead_stride_v       = nhead_stride_v,                                     // int
    .nhead_stride_bias    = 0,                                                  // int
    .nhead_stride_randval = 0,                                                  // int
    .nhead_stride_lse     = nhead_stride_lsed,                                  // int
    .nhead_stride_o       = nhead_stride_o,                                     // int
    .batch_stride_q       = batch_stride_q,                                     // int
    .batch_stride_k       = batch_stride_k,                                     // int
    .batch_stride_v       = batch_stride_v,                                     // int
    .batch_stride_bias    = 0,                                                  // int
    .batch_stride_randval = 0,                                                  // int
    .batch_stride_lse     = batch_stride_lse,                                   // int
    .batch_stride_o       = batch_stride_o,                                     // int
    .batch_stride_q_descale = 0,                                                // int
    .batch_stride_k_descale = 0,                                                // int
    .batch_stride_v_descale = 0,                                                // int
    .window_size_left     = window_size_left,                                   // int
    .window_size_right    = window_size_right,                                  // int
    .mask_type            = mask_type,                                          // int
    .min_seqlen_q         = 0,                                                  // int
    .p_drop               = 0.0,                                                // float
    .s_randval            = false,                                              // bool
    .drop_seed_offset     = std::make_pair<uint64_t, uint64_t>(0, 0),
    // .block_scale_size_q   = 0,
    // .block_scale_size_kv  = 0,
  };

  return ret;
}
#pragma GCC diagnostic pop

hipError_t
AiterFmhaV3FwdContext::launch(hipStream_t stream) const {
  cookie.v3_api_check = false;
  AOTRITON_NS::v3::aiter::ck_tile::stream_config sc {
    .stream_id_ = stream,
  };
  // FIXME: hipErrorPeerAccessUnsupported
  return fmha_fwd_v3(cookie, sc) == 0 ? hipSuccess : hipErrorPeerAccessUnsupported;
}

}
