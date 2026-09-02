// Copyright © 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AOTRITON_V3_API_FLASH_ATTN_H
#define AOTRITON_V3_API_FLASH_ATTN_H

#include <aotriton/config.h>
#include "runtime.h"
#include "util.h"
#include "cpp_tune.h"

namespace AOTRITON_NS::v2::flash {

// check_gpu will be preserved for backward compatibility
hipError_t AOTRITON_API
check_gpu(AOTRITON_NS::Stream stream);

hipError_t AOTRITON_API
debug_simulate_encoded_softmax(AOTRITON_NS::TensorView<4> r,  // batch_size x num_heads x max_seqlen_q x max_seqlen_k
                               float dropout_p,
                               AOTRITON_NS::TensorView<0> philox_seed,
                               AOTRITON_NS::TensorView<0> philox_offset1,
                               uint64_t philox_offset2,
                               AOTRITON_NS::Stream stream);

}

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
  // Kernel slot assignments in kernel_fine_control array
  // Automatically generated from kernel NAMEs
  // See modules/flash/aot/__init__.py for kernel definitions
  enum KernelSlot {
    // Forward pass kernels (from attn_fwd, etc.)
    attn_fwd = 0,
    debug_simulate_encoded_softmax = 1,

    // Backward pass kernels (from bwd_preprocess, bwd_kernel_*, etc.)
    // bwd_preprocess_varlen was merged into bwd_preprocess (one kernel under
    // varlen_bits == 0), so the dk_dv/dq/fuse slots shifted down by one. Safe
    // because every consumer resolves this enum BY NAME: the shim codegen
    // builds KERNEL_SLOT_INDEX from the kernel name, and modules/flash/tune/
    // spells `c.KernelSlot.<name>`.
    bwd_preprocess = 2,
    bwd_kernel_dk_dv = 3,
    bwd_kernel_dq = 4,
    bwd_kernel_fuse = 5,

    MaxKernels = 6
  };

  // Fine-grained kernel control within Metro backends
  // Use KernelSlot enum to index into this array-like container
  // Returns shared_ptr<KernelControl> for reference semantics in Python
  mutable KernelFineControl kernel_fine_control{KernelSlot::MaxKernels};
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

// ---------------------------------------------------------------------------
// varlen_bits: the layout descriptor the GPU kernels decode.
//
// Variable-length attention is now implemented as a product of three
// independent, PER-SIDE choices:
//
// A) Is the token axis stacked;
// B) how a sequence's length is given;
// C) where the sequence starts
//
// The public uses a bit-field struct rather than an opaque integer for ease of use.
// Note even if the the host compiler's bit-field allocation is
// implementation-defined, for a given compiler and given processor, the
// allocation is determined and thus will not break the ABI.
//
// Struct-of-constants rather than `enum class`, for the reason recorded above
// this block: an enum class does not convert to its underlying type without a
// cast, and these values are assigned straight into bit-fields.

// A. Is the token axis stacked?
struct AOTRITON_API VarlenStacked {
  static constexpr uint32_t BHSD = 0;   // rank-4, sequence selected by batch index
  static constexpr uint32_t THD  = 1;   // Still rank-4, 1THD shape
};

// B. How is the length of sequence z given?  (info_0 is seqinfo_q0/seqinfo_k0)
struct AOTRITON_API VarlenLength {
  static constexpr uint32_t MAX        = 0;   // every sequence is Max_seqlen
  static constexpr uint32_t CUMULATIVE = 1;   // info_0[z+1] - info_0[z], (N+1,)
  static constexpr uint32_t INDIVIDUAL = 2;   // info_0[z], (N,)
};

// C. Where does sequence z start along the token axis?  (info_1 is
// seqinfo_q1/seqinfo_k1;
struct AOTRITON_API VarlenPosition {
  static constexpr uint32_t IMPLIED = 0;  // 0 if BHSD, else z * Max_seqlen
  static constexpr uint32_t REUSE   = 1;  // re-use info_0[z], already loaded
  static constexpr uint32_t ARRAY   = 2;  // info_1[z]
};

// LSE/Delta memory arrangement. HT is AOTriton's and the default; TH is what
// Transformer Engine requires. Two bits, not one for possible extension.
struct AOTRITON_API VarlenLseLayout {
  static constexpr uint32_t HT = 0;   // (H, T), offset (b*H + h)*S + s
  static constexpr uint32_t TH = 1;   // (T, H), offset (b*S + s)*H + h
};

// One side's three axes. uint8_t, not uint32_t, so the struct really is the one
// byte the encoding gives it rather than a 4-byte storage unit holding 8 bits.
struct AOTRITON_API VarlenMode {
  uint8_t stacked  : 1;
  uint8_t length   : 2;
  uint8_t position : 2;
  uint8_t reserved : 3;
};

static_assert(sizeof(VarlenMode) == 1, "a side's mode is one byte on the wire");

// alignas because the byte-sized members would otherwise align this to 2, and
// the whole point is that it occupies exactly the space -- and the slot -- an
// int32_t would.
struct AOTRITON_API alignas(uint32_t) VarlenBits {
  // Q and K decode independently, so mixed addressing modes are expressible.
  // Only the seqused_k pairing is exercised through this API; the rest of the
  // combination space is expected to work but untested.
  VarlenMode qmode;          // bits 7:0
  VarlenMode kmode;          // bits 15:8 -- same layout, so one decoder twice
  uint16_t lse_layout : 2;   // bits 17:16
  uint16_t reserved   : 14;  // bits 31:18
};

// Occupies exactly the space an int32_t would, so appending it to a params
// struct perturbs nothing. A compiler that disagreed would shift every field a
// consumer adds after it.
static_assert(sizeof(VarlenBits) == 4, "varlen_bits is a u32 on the wire");
static_assert(alignof(VarlenBits) == 4, "VarlenBits must align like an int32_t");

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
  // int32_t  Num_seqlens;      // Inferred from seqinfo_q0 and varlen_bits
  T1       seqinfo_q0;
  T1       seqinfo_k0;
  int32_t  Max_seqlen_q = 0;
  int32_t  Max_seqlen_k = 0;
  T1       seqinfo_q1;
  T1       seqinfo_k1;
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
  // int8_t varlen_type;        // Superseded by varlen_bits
  int32_t  window_left;
  int32_t  window_right;
  VarlenBits varlen_bits = {};

  static constexpr int32_t kVersion = 4;
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
  // int32_t   Num_seqlens;         // Inferred from seqinfo_q0 and varlen_bits
  T1        seqinfo_q0;
  T1        seqinfo_k0;
  int32_t   Max_seqlen_q = 0;
  int32_t   Max_seqlen_k = 0;
  T1        seqinfo_q1;
  T1        seqinfo_k1;
  // int32_t   Head_dim;            // Inferred from Q.size()
  float     dropout_p;
  T0        philox_seed_ptr;
  T0        philox_offset1;
  uint64_t  philox_offset2;
  int8_t    causal_type;
  // int8_t varlen_type;            // Superseded by varlen_bits
  int32_t   window_left;
  int32_t   window_right;
  mutable LT4       DQ_ACC;         // fp32 accumulator of dq
  VarlenBits varlen_bits = {};

  static constexpr int32_t kVersion = 7;
  attn_bwd_params();
};

hipError_t AOTRITON_API
attn_bwd(const attn_bwd_params& params,
         int32_t params_version,
         AOTRITON_NS::Stream stream,
         const attn_options* options = nullptr);

} // AOTRITON_NS::v3::flash

#endif // AOTRITON_V3_API_FLASH_ATTN_H
