// Copyright © 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/config.h>
#include <aotriton/_internal/util.h>
#include <aotriton/_internal/flash/aiter.h>
#include <aotriton/flash.h>
#include <aotriton/util.h>
#include <aotriton/_internal/lazy_tensor_internal.h>
#include <flash/iface.op_attn_bwd.h>
#include <flash/affine.aiter_fmha_v3_bwd.h>
#include <algorithm>
#include <limits>
#ifndef NDEBUG
#include <iostream>
#include <stdio.h>
#endif

namespace AOTRITON_NS::v3::flash {

const char*
AiterFmhaV3BwdContext::check_inputs_are_supported() {
  const auto& args = *params;
#define RETURN_IF(COND)                                               \
  do {                                                                \
    if (COND) {                                                       \
      return "Input unsupported due to " STRINGIFICATION(COND);       \
    }                                                                 \
  } while(0)
  // No bias support
  RETURN_IF(args.BIAS_TYPE);
  // No Varlen support
  RETURN_IF(args.cu_seqlens_q && *args.cu_seqlens_q);
  RETURN_IF(args.cu_seqlens_k && *args.cu_seqlens_k);
  // Only support hdim <= 192
  RETURN_IF(args.hdim_qk > 192 || args.hdim_vo > 192);
  // Only support hdim_qk == hdim_vo
  RETURN_IF(args.hdim_qk != args.hdim_vo);
  // TODO: support dropout kernel. fwd and bwd should have identical PRNG
  RETURN_IF(args.ENABLE_DROPOUT);
  RETURN_IF(args.num_head_q != args.num_head_k);
  RETURN_IF(!args.DQ_ACC);
#undef RETURN_IF
  // We do not have test suite to validate SWA at the moment.
  if (args.CAUSAL_TYPE != CausalType::None) {
      if (args.Window_left != WindowValue::TopLeftAligned ||
          args.Window_right != WindowValue::TopLeftAligned) {
#ifndef NDEBUG
        std::cerr << "Input unsupported due to args.CAUSAL_TYPE = " << int(args.CAUSAL_TYPE) << " and "
                  << " args.Window_left = " << args.Window_left
                  << " args.Window_right = " << args.Window_right
                  << std::endl;
#endif
        return "Input unsupported due to SWA";
      }
  }
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
  CHECK_STRIDE(args.DO);
  CHECK_STRIDE(args.DK);
  CHECK_STRIDE(args.DV);
  CHECK_STRIDE(args.DB);
#undef CHECK_STRIDE

  return nullptr;
}

// Too many narrowing warning here.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnarrowing"

aiter::mha_bwd_args
construct_mha_bwd_args(const AiterFmhaV3BwdContext& ctx) {
  const auto& args = *ctx.params;
  auto batch = args.Q->size(0);
  auto nhead_q = args.Q->size(1);
  auto nhead_k = args.K->size(1);
  auto hdim_qk = args.Q->size(3);
  auto hdim_vo = args.V->size(3);
  auto scale = args.sm_scale;
  auto stride_q = args.Q->stride(2);
  auto stride_k = args.K->stride(2);
  auto stride_v = args.V->stride(2);
  auto stride_o = args.Out->stride(2);
  auto stride_do = args.DO->stride(2);
  auto stride_dq_acc = args.DQ_ACC->stride(2);
  auto stride_dq = args.DQ->stride(2);
  auto stride_dk = args.DK->stride(2);
  auto stride_dv = args.DV->stride(2);

  auto nhead_stride_q = args.Q->stride(1);
  auto nhead_stride_k = args.K->stride(1);
  auto nhead_stride_v = args.V->stride(1);
  auto nhead_stride_o = args.Out->stride(1);
  auto nhead_stride_do = args.DO->stride(1);
  auto nhead_stride_dq_acc = args.DQ_ACC->stride(1);
  auto nhead_stride_dq = args.DQ->stride(1);
  auto nhead_stride_dk = args.DK->stride(1);
  auto nhead_stride_dv = args.DV->stride(1);
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
  auto batch_stride_do = args.DO->stride(0);
  auto batch_stride_dq_acc = args.DQ_ACC->stride(0);
  auto batch_stride_dq = args.DQ->stride(0);
  auto batch_stride_dk = args.DK->stride(0);
  auto batch_stride_dv = args.DV->stride(0);
  auto batch_stride_lsed = nhead_q * seqlen_q;

  auto data_type = [&args]() {
    if (args.Q->dtype() == DType::kFloat16)
      return "fp16";
    return "bf16";
  };
  // AITER ASM TopLeftAligned/BottomRightAligned logic:
  // 1. a.mask_type == None: fast accept
  // 2. a.mask_type == Window: fast reject
  // 3. Check window_size == (-1, 0), then use a.mask_type to tell TL/BR
  // 4. Return none when window_size == (-1, -1)
  // 4. Return 3 elsewhere
  auto [mask_type, window_size_left, window_size_right] = [&args]() -> std::tuple<int, int, int> {
    if (args.CAUSAL_TYPE == CausalType::None)
      return {0, -1, -1};
    if (args.Window_left == WindowValue::TopLeftAligned)
      return {1, -1, 0};
    if (args.Window_left == WindowValue::BottomRightAligned)
      return {2, -1, 0};
    return {3, args.Window_left, args.Window_right};
  }();

  // TODO: use .v3_api_check for lookup_optimal
  aiter::mha_bwd_args ret = {
    // aiter args
    .use_asm_v3           = true,                                               // bool
    .v3_atomic_fp32       = true,                                               // bool
    .v3_bf16_cvt          = 0,                                                  // int
    .v3_api_check         = false,                                              // bool
    // From ck  fmha_bwd_traits                                                 
    .hdim_q               = hdim_qk,                                            // int
    .hdim_v               = hdim_vo,                                            // int
    .data_type            = data_type(),                                        // std::string
    .is_group_mode        = nhead_q != nhead_k,                                 // bool
    .mask_type            = mask_type,                                          // int
    .bias_type            = args.BIAS_TYPE,                                     // int
    .has_dbias            = 0,                                                  // bool
    .has_dropout          = args.ENABLE_DROPOUT,                                // bool
    .is_store_randval     = false,                                              // bool
    .is_deterministic     = false,                                              // bool
    // From ck  fmha_bwd_args                                                   
    .q_ptr                = args.Q->data_ptr(),                                 // const void*
    .k_ptr                = args.K->data_ptr(),                                 // const void*
    .v_ptr                = args.V->data_ptr(),                                 // const void*
    .bias_ptr             = args.B->data_ptr(),                                 // const void*
    .o_ptr                = args.Out->data_ptr(),                               // const void*
    .lse_ptr              = args.L->data_ptr(),                                 // const void*
    .do_ptr               = args.DO->data_ptr(),                                // const void*
    .d_ptr                = args.D->data_ptr(),                                 // void*
    .rand_val_ptr         = nullptr,                                            // void*
    .dq_ptr               = args.DQ->data_ptr(),                                // void*
    .dk_ptr               = args.DK->data_ptr(),                                // void*
    .dv_ptr               = args.DV->data_ptr(),                                // void*
    .dbias_ptr            = nullptr,                                            // void*
    .dq_acc_ptr           = args.DQ_ACC->data_ptr(),                            // void*
    .seqstart_q_ptr       = args.seq_strides_q->data_ptr(),                     // const void*
    .seqstart_k_ptr       = args.seq_strides_k->data_ptr(),                     // const void*
    .seqlen_q_ptr         = nullptr,                                            // const void*
    .seqlen_k_ptr         = nullptr,                                            // const void*
    .cu_seqlen_q_ptr      = args.cu_seqlens_q->data_ptr(),                      // const void*
    .cu_seqlen_k_ptr      = args.cu_seqlens_k->data_ptr(),                      // const void*
    .seqlen_q             = args.max_seqlen_q,                                  // int
    .seqlen_k             = args.max_seqlen_k,                                  // int
    .batch                = batch,                                              // int
    .max_seqlen_q         = args.max_seqlen_q,                                  // int
    .max_seqlen_k         = args.max_seqlen_k,                                  // int
    .nhead_q              = nhead_q,                                            // int
    .nhead_k              = nhead_k,                                            // int
    .scale                = scale,                                              // float
    .stride_q             = stride_q,                                           // int
    .stride_k             = stride_k,                                           // int
    .stride_v             = stride_v,                                           // int
    .stride_bias          = 0,                                                  // int
    .stride_o             = stride_o,                                           // int
    .stride_randval       = 0,                                                  // int
    .stride_do            = stride_do,                                          // int
    .stride_dq_acc        = stride_dq_acc,                                      // int
    .stride_dq            = stride_dq,                                          // int
    .stride_dk            = stride_dk,                                          // int
    .stride_dv            = stride_dv,                                          // int
    .stride_dbias         = 0,                                                  // int
    .nhead_stride_q       = nhead_stride_q,                                     // int
    .nhead_stride_k       = nhead_stride_k,                                     // int
    .nhead_stride_v       = nhead_stride_v,                                     // int
    .nhead_stride_bias    = 0,                                                  // int
    .nhead_stride_o       = nhead_stride_o,                                     // int
    .nhead_stride_randval = 0,                                                  // int
    .nhead_stride_do      = nhead_stride_do,                                    // int
    .nhead_stride_lsed    = nhead_stride_lsed,                                  // int
    .nhead_stride_dq_acc  = nhead_stride_dq_acc,                                // int64_t
    .nhead_stride_dq      = nhead_stride_dq,                                    // int
    .nhead_stride_dk      = nhead_stride_dk,                                    // int
    .nhead_stride_dv      = nhead_stride_dv,                                    // int
    .nhead_stride_dbias   = 0,                                                  // int
    .batch_stride_q       = batch_stride_q,                                     // int
    .batch_stride_k       = batch_stride_k,                                     // int
    .batch_stride_v       = batch_stride_v,                                     // int
    .batch_stride_bias    = 0,                                                  // int
    .batch_stride_o       = batch_stride_o,                                     // int
    .batch_stride_randval = 0,                                                  // int
    .batch_stride_do      = batch_stride_do,                                    // int
    .batch_stride_lsed    = batch_stride_lsed,                                  // int
    .batch_stride_dq_acc  = batch_stride_dq_acc,                                // int64_t
    .batch_stride_dq      = batch_stride_dq,                                    // int
    .batch_stride_dk      = batch_stride_dk,                                    // int
    .batch_stride_dv      = batch_stride_dv,                                    // int
    .batch_stride_dbias   = 0,                                                  // int
    .split_stride_dq_acc  = 0,                                                  // int
    .window_size_left     = window_size_left,                                   // int
    .window_size_right    = window_size_right,                                  // int
    .p_drop               = 0.0,                                                // float
    .p_undrop             = 0.0,                                                // float
    .drop_seed_offset     = std::make_pair<uint64_t, uint64_t>(0, 0),
  };

  return ret;
}
#pragma GCC diagnostic pop

hipError_t
AiterFmhaV3BwdContext::launch(hipStream_t stream) const {
  auto a = construct_mha_bwd_args(*this);
  AOTRITON_NS::v3::aiter::ck_tile::stream_config sc {
    .stream_id_ = stream,
  };
  // FIXME: hipErrorPeerAccessUnsupported
  return fmha_v3_bwd(a, sc) == 0 ? hipSuccess : hipErrorPeerAccessUnsupported;
}

}
