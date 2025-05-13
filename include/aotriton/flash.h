// Copyright © 2023-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AOTRITON_V2_API_FLASH_ATTN_H
#define AOTRITON_V2_API_FLASH_ATTN_H

#include <aotriton/config.h>
#include "runtime.h"
#include "util.h"
#include "cpp_tune.h"

namespace AOTRITON_NS::v2::flash {

hipError_t AOTRITON_API
check_gpu(AOTRITON_NS::Stream stream);

using T4 = AOTRITON_NS::TensorView<4>;
using T2 = AOTRITON_NS::TensorView<2>;
using T1 = AOTRITON_NS::TensorView<1>;
using T0 = AOTRITON_NS::TensorView<0>;

struct AOTRITON_API FwdExtraArguments : public CppTune {
};

struct AOTRITON_API BwdExtraArguments {
#if AOTRITON_BUILD_FOR_TUNING
  FwdExtraArguments dkdv, dqdb;
#endif
};

struct AOTRITON_API FusedBwdExtraArguments : public CppTune {
};

hipError_t AOTRITON_API
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
         T0 atomic_for_causal,
         AOTRITON_NS::Stream stream,
         FwdExtraArguments* extargs = nullptr);

hipError_t AOTRITON_API
attn_fwd_compact_varlen(T4 q, // 1 x num_heads x total_q x head_size, total_q := \sum_{i=0}^{b} s_i
                        T4 k, // 1 x num_heads x total_k x head_size, total_k := \sum_{i=0}^{b} s_i
                        T4 v, // 1 x num_heads x total_v x head_size, total_, := \sum_{i=0}^{b} s_i
                        T4 b, // reserved, note this b is "bias", not "batch"
                        T1 cu_seqlens_q, // b+1, i64
                        T1 cu_seqlens_k, // b+1, i64
                        int32_t max_seqlen_q, // FIXME: Switch to Tensor
                        int32_t max_seqlen_k,
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
                        T0 atomic_for_causal,
                        AOTRITON_NS::Stream stream,
                        FwdExtraArguments* extargs = nullptr);

hipError_t AOTRITON_API
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
         AOTRITON_NS::Stream stream,
         BwdExtraArguments* extargs = nullptr);

hipError_t AOTRITON_API
attn_bwd_fused(T4 q, // batch_size x num_heads x seqlen_q x head_size
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
               float dropout_p,
               T0 philox_seed,
               T0 philox_offset1,
               int64_t philox_offset2,
               bool is_causal,
               AOTRITON_NS::Stream stream,
               FusedBwdExtraArguments* extargs = nullptr);

hipError_t AOTRITON_API
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
                        AOTRITON_NS::Stream stream,
                        BwdExtraArguments* extargs = nullptr);

// varlen should use len(cu_seqlens_q) - 1 for the batch size
hipError_t AOTRITON_API
debug_simulate_encoded_softmax(T4 r,  // batch_size x num_heads x max_seqlen_q x max_seqlen_k
                               float dropout_p,
                               T0 philox_seed,
                               T0 philox_offset1,
                               uint64_t philox_offset2,
                               AOTRITON_NS::Stream stream);

} // AOTRITON_NS::v2::flash


namespace AOTRITON_NS::v3::flash {

using T4 = AOTRITON_NS::TensorView<4>;
using T2 = AOTRITON_NS::TensorView<2>;
using T1 = AOTRITON_NS::TensorView<1>;
using T0 = AOTRITON_NS::TensorView<0>;

// For debugging and profiling purpose
struct attn_options {
};

struct attn_fwd_params {
  T4       Q;
  T4       K;
  T4       V;
  T4       B;
  float    Sm_scale;
  T2       L;
  T4       Out;
  int32_t  Num_head_q;
  int32_t  Num_head_k;
  int32_t  Num_seqlens;
  T1       cu_seqlens_q;
  T1       cu_seqlens_k;
  int32_t  Max_seqlen_q;
  int32_t  Max_seqlen_k;
  int32_t  Head_dim;
  float    dropout_p;
  T0       philox_seed_ptr;
  T0       philox_offset1;
  uint64_t philox_offset2;
  T0       philox_seed_output;
  T0       philox_offset_output;
  T4       encoded_softmax;
  int8_t   causal_type;
  T0       persistent_atomic_counter;

  static constexpr int32_t kVersion = 1;
};

hipError_t AOTRITON_API
attn_fwd(const attn_fwd_params& params,
         int32_t params_version,
         AOTRITON_NS::Stream stream,
         const attn_options* options = nullptr);

struct attn_bwd_params {
  T4        Q;
  T4        K;
  T4        V;
  T4        B;
  float     Sm_scale;
  T4        Out;
  T4        DO;
  T4        DK;
  T4        DV;
  T4        DQ;
  T4        DB;
  T2        L;
  T2        D;
  int32_t   Num_head_q;
  int32_t   Num_head_k;
  int32_t   Num_seqlens;
  T1        cu_seqlens_q;
  T1        cu_seqlens_k;
  int32_t   Max_seqlen_q;
  int32_t   Max_seqlen_k;
  int32_t   Head_dim;
  float     dropout_p;
  T0        philox_seed_ptr;
  T0        philox_offset1;
  uint64_t  philox_offset2;
  int8_t    causal_type;

  static constexpr int32_t kVersion = 1;
};

hipError_t AOTRITON_API
attn_bwd(const attn_bwd_params& params,
         int32_t params_version,
         AOTRITON_NS::Stream stream,
         const attn_options* options = nullptr);

} // AOTRITON_NS::v3::flash

#endif
