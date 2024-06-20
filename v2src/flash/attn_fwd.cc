// Copyright © 2023-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/_internal/util.h>
#include <aotriton/flash.h>
#include <aotriton/util.h>
#include <flash/shim.attn_fwd.h>
#include <iostream>

#ifdef NDEBUG
#define AOTRITON_VERBOSE 0
#else
#define AOTRITON_VERBOSE 1
#endif

namespace aotriton::v2::flash {

hipError_t
_attn_fwd_common(T4 q,
                 T4 k,
                 T4 v,
                 T1 cu_seqlens_q,
                 T1 cu_seqlens_k,
                 int32_t num_seqlens,
                 int32_t max_seqlen_q,
                 int32_t max_seqlen_k,
                 T4 b,
                 float sm_scale,
                 T2 softmax_lse,
                 T4 out,
                 float dropout_p,
                 uint64_t philox_seed,
                 uint64_t philox_offset,
                 T4 encoded_softmax,
                 bool is_causal,
                 aotriton::Stream stream_wrap) {
  hipError_t err;
  auto stream = stream_wrap.native();
  auto arch = getArchFromStream(stream);
  auto grid_calculator = [](const AttnFwdParams& params) -> dim3 {
#if AOTRITON_VERBOSE
    std::cerr << "Selected Kernel "
              << " BLOCK_M = " << params.BLOCK_M << " BLOCK_N = " << params.BLOCK_N
              << " pre_load_v = " << params.pre_load_v << std::endl;
#endif
    dim3 grid {
      aotriton::cdiv<uint32_t>(params.max_seqlen_q, params.BLOCK_M),
      uint32_t(params.Q->size(1)),
      params.num_seqlens == 0 ? uint32_t(params.Q->size(0)) : params.num_seqlens,
    };
#if AOTRITON_VERBOSE
    std::cerr << "Grid conf " << grid.x << " " << grid.y << " " << grid.z << std::endl;
#endif
    return grid;
  };
  int head_size = q.size(3);
  int head_dim_rounded = std::max<int>(16, aotriton::bit_ceil(head_size));
  int bias_type = 0;
  if (b) {
    bias_type = 1;
  }
  // Requires C++ 20
  AttnFwdParams params = {
    .Q = &q,
    .K = &k,
    .V = &v,
    .B = &b,
    .Out = &out,
    .encoded_softmax = &encoded_softmax,
    .sm_scale = sm_scale,
    .M = &softmax_lse,
    .cu_seqlens_q = &cu_seqlens_q,
    .cu_seqlens_k = &cu_seqlens_k,
    .num_seqlens = num_seqlens,
    .max_seqlen_q = max_seqlen_q,
    .max_seqlen_k = max_seqlen_k,
    .head_dim = static_cast<int32_t>(head_size),
    .dropout_p = dropout_p,
    .philox_seed = philox_seed,
    .philox_offset_base = static_cast<uint32_t>(philox_offset),
    .CAUSAL = is_causal,
    .BLOCK_DMODEL = head_dim_rounded,
    .ENABLE_DROPOUT = dropout_p > 0.0,
    .RETURN_ENCODED_SOFTMAX = bool(encoded_softmax),
    .PADDED_HEAD = head_dim_rounded != head_size,
    .BIAS_TYPE = bias_type,
  };
  AttnFwdContext context;
  context.grid_calculator = grid_calculator;
  // .grid_calculator = grid_calculator
  err = context.lookup_optimal(params, arch);
  if (err != hipSuccess) {
    return err;
  }
  err = context.launch(params, stream);
  return err;
}

hipError_t
attn_fwd(T4 q,
         T4 k,
         T4 v,
         T4 b,
         float sm_scale,
         T2 softmax_lse,
         T4 out,
         float dropout_p,
         uint64_t philox_seed,
         uint64_t philox_offset,
         T4 encoded_softmax,
         bool is_causal,
         aotriton::Stream stream_wrap) {
  auto null_t1 = T1::get_null_tensor(DType::kInt32);
  return _attn_fwd_common(q,
                          k,
                          v,
                          null_t1,
                          null_t1,
                          0,
                          q.size(2),
                          k.size(2),
                          b,
                          sm_scale,
                          softmax_lse,
                          out,
                          dropout_p,
                          philox_seed,
                          philox_offset,
                          encoded_softmax,
                          is_causal,
                          stream_wrap);
}

hipError_t
attn_fwd_compact_varlen(T4 q,            // 1 x num_heads x total_q x head_size, total_q := \sum_{i=0}^{b} s_i
                        T4 k,            // 1 x num_heads x total_k x head_size, total_k := \sum_{i=0}^{b} s_i
                        T4 v,            // 1 x num_heads x total_v x head_size, total_, := \sum_{i=0}^{b} s_i
                        T1 cu_seqlens_q, // b+1, i64
                        T1 cu_seqlens_k, // b+1, i64
                        int32_t max_seqlen_q,
                        int32_t max_seqlen_k,
                        T4 b, // reserved, note this b is "bias", not "batch"
                        float sm_scale,
                        T2 softmax_lse,
                        T4 out, // 1 x num_heads x total_q x head_size
                        float dropout_p,
                        uint64_t philox_seed,
                        uint64_t philox_offset,
                        T4 encoded_softmax,
                        bool is_causal,
                        aotriton::Stream stream_wrap) {
  return _attn_fwd_common(q,
                          k,
                          v,
                          cu_seqlens_q,
                          cu_seqlens_k,
                          cu_seqlens_q.size(0) - 1,
                          max_seqlen_q,
                          max_seqlen_k,
                          b,
                          sm_scale,
                          softmax_lse,
                          out,
                          dropout_p,
                          philox_seed,
                          philox_offset,
                          encoded_softmax,
                          is_causal,
                          stream_wrap);
}

}
