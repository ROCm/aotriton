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
  bool deterministic = false;

#if AOTRITON_BUILD_FOR_TUNING
  // Special kernel control values for selective execution and querying
  struct KernelControlValue {
    static constexpr int Auto = -1;                    // Use autotuned kernel (default)
    static constexpr int SkipAndQueryKernelNumber = -2;  // Skip kernel and query total count (writes back to array)
    // Values >= 0 mean force specific kernel index
  };

  // Kernel slot assignments in kernel_fine_control array
  // Automatically generated from kernel NAMEs
  // See v3python/rules/flash/__init__.py for kernel definitions
  enum KernelSlot {
    // Forward pass kernels (from attn_fwd, etc.)
    attn_fwd = 0,
    debug_simulate_encoded_softmax = 1,

    // Backward pass kernels (from bwd_preprocess, bwd_kernel_*, etc.)
    bwd_preprocess = 2,
    bwd_preprocess_varlen = 3,
    bwd_kernel_dk_dv = 4,
    bwd_kernel_dq = 5,
    bwd_kernel_fuse = 6,

    MaxKernels = 7
  };

  // Fine-grained kernel control within Metro backends
  // Use KernelSlot enum to index into this array
  // Use KernelControlValue constants for special values
  // Mutable to support querying kernel numbers via QueryKernelNumber
  mutable std::array<int, KernelSlot::MaxKernels> kernel_fine_control = {
    KernelControlValue::Auto,  // attn_fwd
    KernelControlValue::Auto,  // debug_simulate_encoded_softmax
    KernelControlValue::Auto,  // bwd_preprocess
    KernelControlValue::Auto,  // bwd_preprocess_varlen
    KernelControlValue::Auto,  // bwd_kernel_dk_dv
    KernelControlValue::Auto,  // bwd_kernel_dq
    KernelControlValue::Auto   // bwd_kernel_fuse
  };
#endif
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
  static constexpr int8_t StridedVarlen = 3;
};

struct AOTRITON_API attn_fwd_params {
  T4       Q;
  T4       K;
  T4       V;
  T4       B;
  T2       A;
  float    Sm_scale;
  T2       L;                   // Can be T2::get_null_tensor()
  T4       Out;
  // int32_t  Num_head_q;       // Inferred from Q.size()
  // int32_t  Num_head_k;       // Inferred from Q.size()
  // int32_t  Num_seqlens;      // Inferred from cu_seqlens_q
  T1       cu_seqlens_q;
  T1       cu_seqlens_k;
  int32_t  Max_seqlen_q = 0;    // Unused if cu_seqlens_q is empty
  int32_t  Max_seqlen_k = 0;    // Unused if cu_seqlens_k is empty
  T1       seq_strides_q;       // See cu_seqlens_padded parameter in
  T1       seq_strides_k;       // Transformer Engine API nvte_fused_attn_fwd_qkvpacked
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

  static constexpr int32_t kVersion = 3;
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
  T1        seq_strides_q;          // See cu_seqlens_padded parameter in
  T1        seq_strides_k;          // Transformer Engine API nvte_fused_attn_fwd_qkvpacked
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

  static constexpr int32_t kVersion = 6;
  attn_bwd_params();
};

hipError_t AOTRITON_API
attn_bwd(const attn_bwd_params& params,
         int32_t params_version,
         AOTRITON_NS::Stream stream,
         const attn_options* options = nullptr);

} // AOTRITON_NS::v3::flash

#endif // AOTRITON_V3_API_FLASH_ATTN_H
