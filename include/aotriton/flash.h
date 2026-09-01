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
// Variable-length attention is not one feature but a product of three
// independent, PER-SIDE choices (A: is the token axis stacked, B: how a
// sequence's length is given, C: where the sequence starts). The retired
// `VarlenType` enum sampled four points of that space and could not spell the
// rest -- `seqused_k` paired with cumulative offsets, or packed queries against
// a dense KV cache, both of which real callers ship.
//
// The public spelling is a bit-field struct rather than an opaque integer: a
// caller writes `v.q_length = VarlenLength::CUMULATIVE` and can see what it
// asked for, where `0x0B0B` in a debugger cannot say that.
//
// The word the kernel decodes has a FIXED bit allocation; the host compiler's
// bit-field allocation is implementation-defined. The two are therefore never
// equated by reinterpretation: AOTriton builds the wire word from these fields
// with explicit shifts, so it comes out identical on every platform because
// nothing is reinterpreted. That conversion is internal -- a caller fills the
// struct in and never sees the word. It is also why there is no
// <bit>/std::bit_cast here: this header includes no standard library of its
// own, so a consumer built against a different C++ runtime can still use it.
//
// Struct-of-constants rather than `enum class`, for the reason recorded above
// this block: an enum class does not convert to its underlying type without a
// cast, and these values are assigned straight into bit-fields.

// A. Is the token axis stacked?
struct AOTRITON_API VarlenStacked {
  static constexpr uint32_t BHSD = 0;   // rank-4, sequence selected by batch index
  // Every sequence packed along one token axis, selected by a row offset rather
  // than by a batch index. The tensor is still rank 4 -- leave its batch size at
  // 1, since the batch axis no longer distinguishes sequences and the kernel
  // addresses batch slice 0 unconditionally under this mode.
  static constexpr uint32_t THD  = 1;
};

// B. How is the length of sequence z given?  (info_0 is seqinfo_q0/seqinfo_k0)
struct AOTRITON_API VarlenLength {
  static constexpr uint32_t MAX        = 0;   // every sequence is Max_seqlen
  static constexpr uint32_t CUMULATIVE = 1;   // info_0[z+1] - info_0[z], (N+1,)
  static constexpr uint32_t INDIVIDUAL = 2;   // info_0[z], (N,)
};

// C. Where does sequence z start along the token axis?  (info_1 is
// seqinfo_q1/seqinfo_k1; REUSE needs LENGTH == CUMULATIVE, since only then does
// info_0 hold positions as well as lengths, and it passes no info_1 at all.)
struct AOTRITON_API VarlenPosition {
  static constexpr uint32_t IMPLIED = 0;  // 0 if BHSD, else z * Max_seqlen
  static constexpr uint32_t REUSE   = 1;  // re-use info_0[z], already loaded
  static constexpr uint32_t ARRAY   = 2;  // info_1[z]
};

// LSE/Delta memory arrangement. HT is AOTriton's and the default; TH is what
// Transformer Engine requires. Two bits, not one, so that a padded or blocked
// arrangement has somewhere to go without another ABI change.
struct AOTRITON_API VarlenLseLayout {
  static constexpr uint32_t HT = 0;   // (H, T), offset (b*H + h)*S + s
  static constexpr uint32_t TH = 1;   // (T, H), offset (b*S + s)*H + h
};

struct AOTRITON_API VarlenBits {
  // Q side, bits 7:0
  uint32_t q_stacked  : 1;
  uint32_t q_length   : 2;
  uint32_t q_position : 2;
  uint32_t q_reserved : 3;
  // Q and K decode independently, so mixed addressing modes are expressible.
  // Only the seqused_k pairing is exercised through this API; the rest of the
  // combination space is expected to work but untested.
  uint32_t k_stacked  : 1;   // K side, bits 15:8 -- the same byte layout, so
  uint32_t k_length   : 2;   // the kernel has ONE decoder called twice
  uint32_t k_position : 2;
  uint32_t k_reserved : 3;
  uint32_t lse_layout : 2;   // bits 17:16
  uint32_t reserved18 : 6;   // bits 23:18
  uint32_t reserved24 : 8;   // bits 31:24, reserved for paged KV
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
  // Named by ROLE, matching the kernels. seqinfo_?0 is the LENGTH source, read
  // at [z] and additionally at [z+1] under CUMULATIVE; seqinfo_?1 is the
  // POSITION source, read at [z] and only under POSITION == ARRAY. varlen_bits
  // says which is read and how, so either may be a null tensor when its side's
  // mode needs no array -- classical packed varlen passes neither ?1.
  //
  // The roles are fixed and never swapped, which is what lets the two come from
  // DIFFERENT tensors: torch's seqused_k pairs an INDIVIDUAL length array in
  // seqinfo_k0 with a CUMULATIVE position array in seqinfo_k1.
  //
  // These were cu_seqlens_q/k and seq_strides_q/k up to kVersion 3/6. Renaming
  // in place is safe precisely because compatibility is a per-version
  // translation (csrc/params_abi_compat.h) and not a layout promise; the old
  // names survive in versioned_attn_*_params<N>, which is now the only place
  // they describe anything.
  T1       seqinfo_q0;
  T1       seqinfo_k0;
  // Read when that side's LENGTH is MAX, or its POSITION is IMPLIED under THD.
  // The host needs Max_seqlen_q regardless, to size the grid.
  int32_t  Max_seqlen_q = 0;
  int32_t  Max_seqlen_k = 0;
  // Transformer Engine's cu_seqlens_padded (nvte_fused_attn_fwd_qkvpacked) maps
  // onto seqinfo_?1, as does torch's cu_seq_k when seqused_k supplies lengths.
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
  // NOTE: `int8_t varlen_type` used to sit here, up to kVersion 3 / 6. It has
  // been replaced by varlen_bits below, and removing it moved NOTHING: it lived
  // inside the padding that precedes the 4-aligned window_left, so every later
  // field keeps its offset and sizeof is unchanged. That is deliberate, and it
  // is what lets a binary compiled against the older header keep running when
  // this library is dropped in beneath it -- the byte it wrote is still there,
  // and the shim reads it when params_version says the caller predates
  // varlen_bits. Do not fill this padding with a new field.
  int32_t  window_left;
  int32_t  window_right;
  // APPEND ONLY, from kVersion 4 onwards. Every field added here goes at the
  // END of the struct, so the previous version's layout stays a strict prefix
  // of this one -- that is what lets the shim upgrade an old caller's params
  // with a version test instead of a shadow struct per version.
  //
  // kVersion 4. The whole varlen space, superseding the VarlenType enum. A
  // caller below kVersion 4 does not have this field at all, so the shim must
  // not read it there. Zero-initialized is the dense case.
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
  // Roles and read conditions exactly as attn_fwd_params documents them.
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
  // See attn_fwd_params: the removed `varlen_type` byte stays padding on
  // purpose, so this struct is ABI-compatible with kVersion 6.
  int32_t   window_left;
  int32_t   window_right;
  mutable LT4       DQ_ACC;          // fp32 accumulator of dq
  // APPEND ONLY, from kVersion 7 onwards -- see attn_fwd_params above for why.
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
