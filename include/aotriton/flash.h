// Copyright © 2023-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AOTRITON_V2_API_FLASH_ATTN_H
#define AOTRITON_V2_API_FLASH_ATTN_H

#include "runtime.h"
#include "util.h"
#include "cpp_tune.h"

namespace aotriton::v2::flash {

hipError_t
check_gpu(aotriton::Stream stream);

using T4 = aotriton::TensorView<4>;
using T2 = aotriton::TensorView<2>;
using T1 = aotriton::TensorView<1>;
using T0 = aotriton::TensorView<0>;

struct FwdExtraArguments : public CppTune {
};

struct BwdExtraArguments {
#if AOTRITON_BUILD_FOR_TUNING
  FwdExtraArguments dkdv, dqdb;
#endif
};

hipError_t
attn_fwd(T4 q, // batch_size x num_heads x seqlen_q x head_size
         T4 k, // batch_size x num_heads x seqlen_k x head_size
         T4 v, // batch_size x num_heads x seqlen_k x head_size
         T4 b, // batch_size x num_heads x seqlen_k x head_size
         float sm_scale,
         T2 softmax_lse,
         T4 Out, // batch_size x num_heads x seqlen_q x head_size
         float dropout_p,
         T0 philox_seed,
         T0 philox_offset1,
         int64_t philox_offset2,
         T0 philox_seed_output,
         T0 philox_offset_output,
         T4 encoded_softmax,
         bool is_causal,
         aotriton::Stream stream,
         FwdExtraArguments* extargs = nullptr);

hipError_t
attn_fwd_compact_varlen(T4 q, // 1 x num_heads x total_q x head_size, total_q := \sum_{i=0}^{b} s_i
                        T4 k, // 1 x num_heads x total_k x head_size, total_k := \sum_{i=0}^{b} s_i
                        T4 v, // 1 x num_heads x total_v x head_size, total_, := \sum_{i=0}^{b} s_i
                        T1 cu_seqlens_q, // b+1, i64
                        T1 cu_seqlens_k, // b+1, i64
                        int32_t max_seqlen_q, // FIXME: Switch to Tensor
                        int32_t max_seqlen_k,
                        T4 b, // reserved, note this b is "bias", not "batch"
                        float sm_scale,
                        T2 softmax_lse,
                        T4 Out, // 1 x num_heads x total_q x head_size
                        float dropout_p,
                        T0 philox_seed,
                        T0 philox_offset1,
                        int64_t philox_offset2,
                        T0 philox_seed_output,
                        T0 philox_offset_output,
                        T4 encoded_softmax,
                        bool is_causal,
                        aotriton::Stream stream,
                        FwdExtraArguments* extargs = nullptr);

hipError_t
attn_bwd(T4 q, // batch_size x num_heads x seqlen_q x head_size
         T4 k, // batch_size x num_heads x seqlen_k x head_size
         T4 v, // batch_size x num_heads x seqlen_k x head_size
         T4 b, // batch_size x num_heads x seqlen_q x seqlen_k
         float sm_scale,
         T4 out,  // batch_size x num_heads x seqlen_q x head_size
         T4 dout, // batch_size x num_heads x seqlen_q x head_size
         T4 dq,   // batch_size x num_heads x seqlen_q x head_size
         T4 dk,   // batch_size x num_heads x seqlen_k x head_size
         T4 dv,   // batch_size x num_heads x seqlen_k x head_size
         T4 db,   // batch_size x num_heads x seqlen_q x seqlen_k
         T2 softmax_lse,
         T2 delta, // buffer, empty_like(softmax_lse)
         float dropout_p,
         T0 philox_seed,
         T0 philox_offset1,
         int64_t philox_offset2,
         bool is_causal,
         aotriton::Stream stream,
         BwdExtraArguments* extargs = nullptr);

hipError_t
attn_bwd_compact_varlen(T4 q, // 1 x num_heads x total_q x head_size, total_q := \sum_{i=0}^{b}
                        T4 k, // 1 x num_heads x total_k x head_size, total_k := \sum_{i=0}^{b}
                        T4 v, // 1 x num_heads x total_v x head_size, total_, := \sum_{i=0}^{b}
                        T1 cu_seqlens_q, // b+1, i64
                        T1 cu_seqlens_k, // b+1, i64
                        int32_t max_seqlen_q,
                        int32_t max_seqlen_k,
                        T4 b, // reserved
                        float sm_scale,
                        T4 out,  // batch_size x num_heads x seqlen_q x head_size
                        T4 dout, // batch_size x num_heads x seqlen_q x head_size
                        T4 dq,   // batch_size x num_heads x seqlen_q x head_size
                        T4 dk,   // batch_size x num_heads x seqlen_k x head_size
                        T4 dv,   // batch_size x num_heads x seqlen_k x head_size
                        T4 db,   // batch_size x num_heads x seqlen_q x seqlen_k
                        T2 softmax_lse,
                        T2 delta, // buffer, empty_like(softmax_lse)
                        float dropout_p,
                        T0 philox_seed,
                        T0 philox_offset1,
                        int64_t philox_offset2,
                        bool is_causal,
                        aotriton::Stream stream,
                        BwdExtraArguments* extargs = nullptr);

hipError_t
debug_fill_dropout_rng(T4 r,
                       uint64_t philox_seed,
                       uint64_t philox_offset,
                       aotriton::Stream stream);

hipError_t
debug_fill_dropout_rng_tensor(T4 r,
                              T0 philox_seed,
                              T0 philox_offset,
                              aotriton::Stream stream);

} // aotriton::v2::flash

#endif
