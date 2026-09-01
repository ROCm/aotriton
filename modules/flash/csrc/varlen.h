// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AOTRITON_MODULES_FLASH_CSRC_VARLEN_H
#define AOTRITON_MODULES_FLASH_CSRC_VARLEN_H

// Host-side glue between the two spellings of the varlen layout: the public
// `VarlenType` enum (four sampled configurations, the only thing a caller
// could say before kVersion 4/7) and the public `VarlenBits` struct (the whole
// space).
//
// <aotriton/flash.h> carries only what a CALLER needs to fill a VarlenBits in:
// the three axis constants, the struct, and its size. Everything about turning
// that struct into the word the kernel decodes is AOTriton's own business and
// lives here -- the bit positions, the encoder, and the asserts pinning them to
// the spec. A consumer never builds the wire word, so putting any of it in the
// public header only widened what a change could break.
//
// The historical params layouts and their translation live next door in
// params_abi_compat.h. They are not varlen: a v3 -> v4 copy carries
// philox_offset2 and DQ_ACC too, and varlen is merely the field that prompted
// the version bump. What that header takes FROM here is varlen_bits_of().
//
// Deliberately internal, and deliberately free functions taking plain scalars
// rather than a params struct: the forward and backward params structs are
// different types spelling the same field differently, so neither a shared
// base nor a template over the struct would bind both. Scalars also make this
// directly unit-testable (modules/flash/tests/test_varlen_translation.cc)
// without constructing a context.

#include <aotriton/config.h>
#include <aotriton/flash.h>

#include <cstddef>
#include <cstdint>

// A nested namespace, not AOTRITON_NS::v3::flash itself. Everything here is
// AOTriton's own: the wire encoding, the legacy enum, the per-version params
// layouts. None of it is part of the user API, and declaring it in the same
// namespace as attn_fwd_params would put it in front of every caller that
// writes `using namespace aotriton::v3::flash`. Follows the existing
// ::aiter sub-namespace.
namespace AOTRITON_NS::v3::flash::internal {

// The retired public enum. It is no longer in <aotriton/flash.h> because no
// caller should reach for it any more -- but a binary compiled against the
// kVersion 3/6 header still WRITES it, so the names have to survive somewhere
// to interpret those bytes. Internal is exactly the right place: needed to read
// the past, not offered for the future.
struct VarlenType {
  static constexpr int8_t None = 0;
  static constexpr int8_t CompactVarlen = 1;
  static constexpr int8_t PaddedVarlen = 2;
  static constexpr int8_t StridedVarlen = 3;
};

// Bit positions of the wire word. The single place the layout is stated as
// numbers; VarlenBits states the same thing as declaration order, and
// varlen_to_wire() is the bridge that never assumes the two agree.
struct VarlenShift {
  static constexpr uint32_t STACKED    = 0;   // within a side byte
  static constexpr uint32_t LENGTH     = 1;
  static constexpr uint32_t POSITION   = 3;
  static constexpr uint32_t K_SIDE     = 8;   // K's byte, relative to Q's
  static constexpr uint32_t LSE_LAYOUT = 16;
};

// Host struct -> the fixed-allocation word the kernel shifts apart.
// UNCONDITIONAL explicit shifts: no platform branch, no endianness #ifdef, no
// bit_cast. Reading a bit-field member inside a constexpr function is fine --
// it is constexpr *bit_cast* over bit-fields that clang rejects -- so the
// spec's hex table stays compile-time checked below.
constexpr uint32_t
varlen_to_wire(VarlenBits v) {
  return (uint32_t(v.q_stacked)  << VarlenShift::STACKED)
       | (uint32_t(v.q_length)   << VarlenShift::LENGTH)
       | (uint32_t(v.q_position) << VarlenShift::POSITION)
       | (uint32_t(v.k_stacked)  << (VarlenShift::K_SIDE + VarlenShift::STACKED))
       | (uint32_t(v.k_length)   << (VarlenShift::K_SIDE + VarlenShift::LENGTH))
       | (uint32_t(v.k_position) << (VarlenShift::K_SIDE + VarlenShift::POSITION))
       | (uint32_t(v.lse_layout) << VarlenShift::LSE_LAYOUT);
}

// The four VarlenType rows, decomposed onto the three axes. This is the whole
// compatibility path: an old caller can express exactly these four
// configurations, so nothing is lost in the upgrade.
//
// It takes THREE inputs and not just the enum, because the enum was never the
// whole classification. Before varlen_bits the host passed a signed
// `Num_seqlens` -- zero when seqinfo_q0 was absent, negated when
// varlen_type said PaddedVarlen -- and the KERNEL then picked strided over
// compact by testing seqinfo_q1 for null. So:
//
//   * varlen_type == None with seqinfo_q0 supplied meant COMPACT, not dense;
//   * varlen_type == CompactVarlen with seqinfo_q1 supplied meant STRIDED;
//   * varlen_type == PaddedVarlen without seqinfo_q0 meant dense, because
//     negating a zero count leaves it zero.
//
// Reproducing only the enum would silently change all three, and the first is
// not hypothetical -- it is what any caller that never set varlen_type but did
// pass seqinfo_q0 has been getting.
constexpr VarlenBits
varlen_bits_of(int8_t varlen_type, bool cu_seqlens_q_present, bool seq_strides_q_present) {
  if (!cu_seqlens_q_present) {
    return {};                                  // dense; Num_seqlens == 0
  }
  if (varlen_type == VarlenType::PaddedVarlen) {  // Num_seqlens < 0
    // BHSD, one sequence per batch slot: lengths still come from a cumulative
    // array, but the position is implied by the batch index.
    return {
      .q_length = VarlenLength::CUMULATIVE,
      .k_length = VarlenLength::CUMULATIVE,
    };
  }
  if (seq_strides_q_present) {
    // THD with a dedicated position array per side.
    return {
      .q_stacked = VarlenStacked::THD,
      .q_length = VarlenLength::CUMULATIVE,
      .q_position = VarlenPosition::ARRAY,
      .k_stacked = VarlenStacked::THD,
      .k_length = VarlenLength::CUMULATIVE,
      .k_position = VarlenPosition::ARRAY,
    };
  }
  // Classical packed varlen: THD, cumulative lengths, position REUSEd out
  // of that same array, so no position array is passed on either side.
  return {
    .q_stacked = VarlenStacked::THD,
    .q_length = VarlenLength::CUMULATIVE,
    .q_position = VarlenPosition::REUSE,
    .k_stacked = VarlenStacked::THD,
    .k_length = VarlenLength::CUMULATIVE,
    .k_position = VarlenPosition::REUSE,
  };
}

// The spec's hex table (sdpa-varlen-plan.md section 2), pinned to the mapping
// itself rather than to a separate set of named constants -- those would be a
// second spelling of these same four rows, and only the mapping is what a
// launch actually goes through. Evaluated by every compiler that reads this
// header, so an implementation-defined bit allocation that packed the other way
// is a build error rather than a plausible wrong address.
static_assert(varlen_to_wire(varlen_bits_of(VarlenType::None, false, false))
              == 0x0000u, "dense must encode as 0x0000");
static_assert(varlen_to_wire(varlen_bits_of(VarlenType::CompactVarlen, true, false))
              == 0x0B0Bu, "compact varlen must encode as 0x0B0B");
static_assert(varlen_to_wire(varlen_bits_of(VarlenType::PaddedVarlen, true, false))
              == 0x0202u, "padded varlen must encode as 0x0202");
static_assert(varlen_to_wire(varlen_bits_of(VarlenType::StridedVarlen, true, true))
              == 0x1313u, "strided varlen must encode as 0x1313");

// The addressing bytes, i.e. everything except LSE_LAYOUT. Two configurations
// with the same addressing describe the same memory traversal and differ only
// in where the logsumexp rows land.
constexpr uint32_t
varlen_addressing(uint32_t wire) {
  return wire & 0xFFFFu;
}

// One side's byte. `k_side` picks K's, which is Q's shifted by K_SIDE.
constexpr uint32_t
varlen_side(uint32_t wire, bool k_side) {
  return (wire >> (k_side ? VarlenShift::K_SIDE : 0u)) & 0xFFu;
}

// Whether the shim must take Max_seqlen from the CALLER rather than from the
// tensor's own extent.
//
// Tensor presence is the wrong predicate and was the bug this replaced: a THD
// side with LENGTH == MAX carries no length array, so keying off seqinfo_?0
// left max_seqlen at K.size(2) -- the total packed token count, not the
// per-sequence maximum -- and decode_addressing then read seqlen = that total
// and row_off = z * that total, i.e. out of bounds. It also let dense bits with
// a stray seqinfo_?0 replace a correct tensor extent with an unset zero.
//
// Only a fully dense side (all three axes at their zero value) may trust the
// tensor: there BHSD + MAX means the extent IS the per-sequence length.
constexpr bool
varlen_uses_caller_max_seqlen(uint32_t wire, bool k_side) {
  return varlen_side(wire, k_side) != 0u;
}

// Cheap well-formedness check on the bits and the arrays they claim to need.
// No device reads, so it runs on every launch -- unlike the prefix-sum and
// non-overlap preconditions, which would cost a sync and stay documented only.
//
// Catches the combinations that would otherwise reach the kernel and fault or
// silently misaddress: an out-of-range field, REUSE without CUMULATIVE (only
// CUMULATIVE makes seqinfo_?0 hold positions as well as lengths), and a mode
// whose array is absent -- the kernel would tl.load from null.
constexpr bool
varlen_side_valid(uint32_t side, bool has_info0, bool has_info1) {
  const uint32_t length = (side >> VarlenShift::LENGTH) & 3u;
  const uint32_t position = (side >> VarlenShift::POSITION) & 3u;
  if (length > VarlenLength::INDIVIDUAL || position > VarlenPosition::ARRAY) {
    return false;                                   // 3 is not a value
  }
  if (side & 0xE0u) {
    return false;                                   // reserved bits 7:5
  }
  if (position == VarlenPosition::REUSE && length != VarlenLength::CUMULATIVE) {
    return false;                                   // seqinfo_?0 holds no position
  }
  if (length != VarlenLength::MAX && !has_info0) {
    return false;                                   // length source missing
  }
  if (position == VarlenPosition::ARRAY && !has_info1) {
    return false;                                   // position source missing
  }
  if (length == VarlenLength::MAX && has_info0) {
    return false;                                   // array supplied but unread
  }
  return true;
}

constexpr bool
varlen_valid(uint32_t wire, bool has_q0, bool has_q1, bool has_k0, bool has_k1) {
  if (wire & 0xFFFC0000u) {
    return false;                                   // reserved bits 31:18
  }
  return varlen_side_valid(varlen_side(wire, false), has_q0, has_q1)
      && varlen_side_valid(varlen_side(wire, true), has_k0, has_k1);
}

// N, the sequence count, read off the Q side of the wire word.
//
// Returns a NEGATIVE value when the bits do not determine it: STACKED with
// LENGTH == MAX is a well-defined uniform stacking for the kernel but hands
// the host no array to count, and Q's batch axis is 1 under THD rather than
// N. The shims reject that rather than guessing, because guessing produces a
// grid of the wrong z extent -- in-bounds addresses of the wrong rows.
//
// `seqinfo_q0_len` must be 0 for an absent tensor. TensorView's default
// constructor leaves its sizes INDETERMINATE (only get_null_tensor zeroes
// them), so `size(0)` on an unset operand is not merely zero, it is garbage;
// use varlen_seqinfo_len() below rather than calling size(0) unguarded.
constexpr int32_t
varlen_seq_count(uint32_t wire, int32_t seqinfo_q0_len, int32_t q_batch) {
  const uint32_t q_stacked = (wire >> VarlenShift::STACKED) & 1u;
  const uint32_t q_length = (wire >> VarlenShift::LENGTH) & 3u;
  if (q_stacked == VarlenStacked::BHSD) {
    return q_batch;                        // one sequence per batch slot
  }
  if (q_length == VarlenLength::CUMULATIVE) {
    return seqinfo_q0_len - 1;             // the array is (N+1,)
  }
  if (q_length == VarlenLength::INDIVIDUAL) {
    return seqinfo_q0_len;                 // the array is (N,)
  }
  return -1;                               // STACKED + MAX: undetermined here
}

inline int32_t
varlen_seqinfo_len(const T1& t) {
  return t ? static_cast<int32_t>(t.size(0)) : 0;
}

// N, for the z extent of every BACKWARD grid. The backward kernels take no
// sequence-count argument -- they read tl.num_programs(2) -- so the grid IS the
// only place N is communicated to them, and the four grid calculators that
// exist (preprocess, dk_dv, dq, fuse) must all produce the same number.
//
// Templated because OpAttnBwdParams lives in a generated header this one does
// not include, and because there is nothing type-specific about it beyond the
// three fields it reads. attn_bwd() validates the count before launching, so a
// negative return cannot reach a grid calculator.
template<typename BwdParams>
inline uint32_t
varlen_bwd_seq_count(const BwdParams* params) {
  return static_cast<uint32_t>(
      varlen_seq_count(static_cast<uint32_t>(params->varlen_bits),
                       varlen_seqinfo_len(*params->seqinfo_q0),
                       static_cast<int32_t>(params->Q->size(0))));
}

// The same count derived WITHOUT consulting the bits: off Q's batch axis and
// the size of whatever length array was handed in, which is how the shims
// computed it before varlen_bits existed. Keyed on tensor PRESENCE where
// varlen_seq_count() is keyed on the STACKED bit, so the two disagree exactly
// when the operands do not match the layout that was declared -- a padded call
// whose seqinfo_q0 is the wrong length, say. That failure is not a crash: it
// is a kernel launched over the wrong number of programs, addressing in-bounds
// rows that are simply the wrong ones, so it is worth one comparison.
//
// Returns -1 when the Q side is INDIVIDUAL, where "array size minus one" is
// not the count and there is nothing independent to compare; the caller then
// skips the comparison rather than inventing one.
constexpr int32_t
varlen_seq_count_independent(uint32_t wire, int32_t seqinfo_q0_len, int32_t q_batch) {
  const uint32_t q_length = (wire >> VarlenShift::LENGTH) & 3u;
  if (q_length == VarlenLength::INDIVIDUAL) {
    return -1;
  }
  return seqinfo_q0_len > 0 ? seqinfo_q0_len - 1 : q_batch;
}

} // AOTRITON_NS::v3::flash::internal

#endif
