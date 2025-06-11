// Copyright © 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/config.h>
#include <aotriton/_internal/util.h>
#include <aotriton/flash.h>
#include <aotriton/util.h>
#include <flash/shim.attn_bwd.h>
#include <flash/iface.op_attn_bwd.h>
#include <iostream>

namespace AOTRITON_NS::v3::flash {

void BwdDqDkDvV3::calculate_residual_func_fields() const {
    const auto& args = *params;
    auto check_if_uniform = [&]() -> bool {
        // Reject varlen
        if (args.cu_seqlens_q || args.cu_seqlens_k) return false;
        // TODO: GQA support
        if (args.num_head_q != args.num_head_k) return false;
#define CMP_TENSOR(X, Y)
        do {                                                            \
            if (args.X->strides() != args.Y->strides()) return false;   \
        } while(0)
        CMP_TENSOR(Q, K);
        CMP_TENSOR(Q, DO);
        CMP_TENSOR(K, V);
        // This is more restrict than AITER kernel
        CMP_TENSOR(K, DK);
        CMP_TENSOR(V, DV);
#undef CMP_TENSOR
        // Tensor Memory layout must be BHSD or BSHD
        // D-last is ensured by caller
        if (args.Q->stride(0) < args.Q->stride(2)) return false;

        if (args.max_seqlen_q != args.max_seqlen_k) return false;
        if (args.max_seqlen_q % 64) return false;
        return true;
    };
    auto check_hdim_regular = [&]() -> bool {
        if (args.head_dim == 64)
            return true;
        if (args.head_dim == 128)
            return true;
        if (args.head_dim == 192)
            return true;
        return false;
    };
    kIsUniformStride = check_if_uniform();
    kIsSEQPad = (args.max_seqlen_q % 64 != 0);
    kIsHDPad = !check_hdim_regular();
    kIsGroupMode = (args.num_head_q != args.num_head_k);
}

dim3 BwdDqDkDvV3::grid_calculator() const {
}

hipError_t AOTRITON_API
aiter_bwd(const attn_bwd_params& in,
          int32_t params_version,
          AOTRITON_NS::Stream stream_wrap,
          const attn_options* options) {
  if (params_version != attn_bwd_params::kVersion) {
    return hipErrorInvalidSymbol; // params_version mismatch
  }
  hipError_t err;
  auto stream = stream_wrap.native();
  auto gpu = getGpuFromStream(stream);
  int batch = in.Q.size(0);
  int head_dim = in.Q.size(3);
  int num_head_q = in.Q.size(1);
  int num_head_k = in.K.size(1);
  int max_seqlen_q = in.Q.size(2);
  int max_seqlen_k = in.K.size(2);
  int num_seqlens = 0;
  if (in.cu_seqlens_q) {
    // Compact varlen, num_seqlens > 0
    num_seqlens = in.cu_seqlens_q.size(0) - 1;
    max_seqlen_q = in.Max_seqlen_q;
  }
  if (in.cu_seqlens_k) {
    max_seqlen_k = in.Max_seqlen_k;
  }
  const auto& compiled_head_dims = BwdKernelDkDvMetadata::get_BLOCK_DMODEL_choices();
  int16_t head_dim_rounded = round_value(head_dim, compiled_head_dims);
  OpAttnBwdParams params = {
    .Q = &in.Q,
    .K = &in.K,
    .V = &in.V,
    .B = &in.B,
    .sm_scale = in.Sm_scale,
    .Out = &in.Out,
    .DO = &in.DO,
    .DK = &in.DK,
    .DV = &in.DV,
    .DQ = &in.DQ,
    .DB = &in.DB,
    .L = &in.L,
    .D = &in.D,
    .num_head_q = num_head_q,
    .num_head_k = num_head_k,
    .cu_seqlens_q = &in.cu_seqlens_q,
    .cu_seqlens_k = &in.cu_seqlens_k,
    .num_seqlens = num_seqlens,
    .max_seqlen_q = max_seqlen_q,
    .max_seqlen_k = max_seqlen_k,
    .head_dim = head_dim,
    .dropout_p = in.dropout_p,
    .philox_seed_ptr  = &in.philox_seed_ptr,
    .philox_offset1   = &in.philox_offset1,
    .philox_offset2   = in.philox_offset2,
    .Window_left = in.window_left,
    .Window_right = in.window_left,
    .BLOCK_DMODEL = head_dim_rounded,
    .CAUSAL_TYPE = in.causal_type,
    .ENABLE_DROPOUT = in.dropout_p > 0.0,
    .PADDED_HEAD = head_dim != head_dim_rounded,
    .BIAS_TYPE = bool(in.B) ? 1 : 0,
  };
  BwdDqDkDvV3Context context;
  context.params = &params;
  err = context.lookup_optimal(gpu);
  if (err != hipSuccess) {
    return err;
  }
  return context.launch(gpu, stream);
}

}
