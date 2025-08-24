// Copyright © 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AOTRITON_V3_API_FLASH_ATTN_H
#define AOTRITON_V3_API_FLASH_ATTN_H

#include <aotriton/config.h>
#include "runtime.h"
#include "util.h"
#include "cpp_tune.h"
#include "v2/flash.h"

namespace AOTRITON_NS::v3::flash {

using T4 = AOTRITON_NS::TensorView<4>;
using T2 = AOTRITON_NS::TensorView<2>;
using T1 = AOTRITON_NS::TensorView<1>;
using T0 = AOTRITON_NS::TensorView<0>;
using LT2 = AOTRITON_NS::LazyTensor<2>;
using LT4 = AOTRITON_NS::LazyTensor<4>;

// For debugging and profiling purpose
struct AOTRITON_API attn_options {
  int force_backend_index = -1;
};

// Note: DO NOT declare enums as enum class : int8_t. Enum class cannot be cased to
// underlying types directly. Compiler complains:
//   error: cannot convert ‘WindowValue’ to ‘int32_t’ {aka ‘int’} in initialization
// etc.
//
// There is no plan to support enum in shim code generator, and hence the cast is unavoidable.

// TopLeftAligned and BottomRightAligned are supported in Triton kernel, but
// not compiled into the binary GPU kernels
struct AOTRITON_API CausalType {
  static constexpr int8_t None = 0;
  // static constexpr int8_t TopLeftAligned = 1;
  // static constexpr int8_t BottomRightAligned = 2;
  static constexpr int8_t WindowedAttention = 3;
};

struct AOTRITON_API WindowValue {
  static constexpr int32_t TopLeftAligned = -2147483647;      // 0x80000001. Special value for varlen
  static constexpr int32_t BottomRightAligned = -2147483646;  // 0x80000002. Special value for varlen
};

struct AOTRITON_API VarlenType {
  static constexpr int8_t None = 0;
  static constexpr int8_t CompactVarlen = 1;
  static constexpr int8_t PaddedVarlen = 2;
};

struct AOTRITON_API attn_fwd_params {
  T4       Q;
  T4       K;
  T4       V;
  T4       B;
  T2       A;
  float    Sm_scale;
  T2       L;
  T4       Out;
  // int32_t  Num_head_q;       // Inferred from Q.size()
  // int32_t  Num_head_k;       // Inferred from Q.size()
  // int32_t  Num_seqlens;      // Inferred from cu_seqlens_q
  T1       cu_seqlens_q;
  T1       cu_seqlens_k;
  int32_t  Max_seqlen_q = 0;    // Unused if cu_seqlens_q is empty
  int32_t  Max_seqlen_k = 0;    // Unused if cu_seqlens_k is empty
  // int32_t  Head_dim;
  float    dropout_p;
  T0       philox_seed_ptr;
  T0       philox_offset1;
  uint64_t philox_offset2;
  T0       philox_seed_output;
  T0       philox_offset_output;
  T4       encoded_softmax;
  T0       persistent_atomic_counter;
  int8_t   causal_type;
  int8_t   varlen_type = 0;
  int32_t  window_left;
  int32_t  window_right;

  static constexpr int32_t kVersion = 1;
  attn_fwd_params();
};

hipError_t AOTRITON_API
attn_fwd(const attn_fwd_params& params,
         int32_t params_version,
         AOTRITON_NS::Stream stream,
         const attn_options* options = nullptr);

struct AOTRITON_API attn_bwd_params {
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
  mutable LT2       D;              // Lazy Tensor must be mutable
  // int32_t   Num_head_q;          // Inferred from Q.size()
  // int32_t   Num_head_k;          // Inferred from Q.size()
  // int32_t   Num_seqlens;         // Inferred from cu_seqlens_q
  T1        cu_seqlens_q;
  T1        cu_seqlens_k;
  int32_t   Max_seqlen_q = 0;       // Unused if cu_seqlens_q is empty
  int32_t   Max_seqlen_k = 0;       // Unused if cu_seqlens_k is empty
  // int32_t   Head_dim;            // Inferred from Q.size()
  float     dropout_p;
  T0        philox_seed_ptr;
  T0        philox_offset1;
  uint64_t  philox_offset2;
  int8_t    causal_type;
  int8_t    varlen_type = 0;
  int32_t   window_left;
  int32_t   window_right;
  mutable LT4       DQ_ACC;          // fp32 accumulator of dq

  static constexpr int32_t kVersion = 3;
  attn_bwd_params();
};

hipError_t AOTRITON_API
attn_bwd(const attn_bwd_params& params,
         int32_t params_version,
         AOTRITON_NS::Stream stream,
         const attn_options* options = nullptr);

// NOTE: DEFERRED TO NEXT RELEASE
//
// hipError_t AOTRITON_API
// aiter_fwd(const attn_fwd_params& params,
//           int32_t params_version,
//           AOTRITON_NS::Stream stream,
//           const attn_options* options = nullptr);

hipError_t AOTRITON_API
aiter_bwd(const attn_bwd_params& params,
          int32_t params_version,
          AOTRITON_NS::Stream stream,
          const attn_options* options = nullptr);

} // AOTRITON_NS::v3::flash

#endif // AOTRITON_V3_API_FLASH_ATTN_H
