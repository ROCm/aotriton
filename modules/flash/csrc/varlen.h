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

// One side's byte. Q and K encode identically, so the whole-word encoder below
// is this called twice; see it for why the shifts are spelled out rather than
// bit_cast'd.
constexpr uint32_t
varlen_mode_to_wire(VarlenMode m) {
  return (uint32_t(m.stacked)  << VarlenShift::STACKED)
       | (uint32_t(m.length)   << VarlenShift::LENGTH)
       | (uint32_t(m.position) << VarlenShift::POSITION)
       | (uint32_t(m.reserved) << 5);
}

// Host struct -> the fixed-allocation word the kernel shifts apart.
// UNCONDITIONAL explicit shifts: no platform branch, no endianness #ifdef, no
// bit_cast. Reading a bit-field member inside a constexpr function is fine --
// it is constexpr *bit_cast* over bit-fields that clang rejects -- so the
// spec's hex table stays compile-time checked below.
//
// Reserved fields are carried, not dropped, so this and varlen_from_wire() are
// a total inverse; varlen_valid() is what requires them to be zero.
constexpr uint32_t
varlen_to_wire(VarlenBits v) {
  return varlen_mode_to_wire(v.qmode)
       | (varlen_mode_to_wire(v.kmode) << VarlenShift::K_SIDE)
       | (uint32_t(v.lse_layout) << VarlenShift::LSE_LAYOUT)
       | (uint32_t(v.reserved) << 18);
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
      .qmode = {.length = VarlenLength::CUMULATIVE},
      .kmode = {.length = VarlenLength::CUMULATIVE},
    };
  }
  if (seq_strides_q_present) {
    // THD with a dedicated position array per side.
    return {
      .qmode = {.stacked = VarlenStacked::THD, .length = VarlenLength::CUMULATIVE, .position = VarlenPosition::ARRAY},
      .kmode = {.stacked = VarlenStacked::THD, .length = VarlenLength::CUMULATIVE, .position = VarlenPosition::ARRAY},
    };
  }
  // Classical packed varlen: THD, cumulative lengths, position REUSEd out
  // of that same array, so no position array is passed on either side.
  return {
    .qmode = {.stacked = VarlenStacked::THD, .length = VarlenLength::CUMULATIVE, .position = VarlenPosition::REUSE},
    .kmode = {.stacked = VarlenStacked::THD, .length = VarlenLength::CUMULATIVE, .position = VarlenPosition::REUSE},
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

// The inverse of varlen_to_wire(). Needed because the CONTEXT helpers -- grid
// calculators -- see only the int32 the generated params struct carries, and
// everything above them reasons in VarlenBits.
constexpr VarlenMode
varlen_mode_from_wire(uint32_t side) {
  VarlenMode m{};
  m.stacked  = (side >> VarlenShift::STACKED) & 1u;
  m.length   = (side >> VarlenShift::LENGTH) & 3u;
  m.position = (side >> VarlenShift::POSITION) & 3u;
  m.reserved = (side >> 5) & 7u;
  return m;
}

constexpr VarlenBits
varlen_from_wire(uint32_t wire) {
  VarlenBits v{};
  v.qmode = varlen_mode_from_wire(wire & 0xFFu);
  v.kmode = varlen_mode_from_wire((wire >> VarlenShift::K_SIDE) & 0xFFu);
  v.lse_layout = (wire >> VarlenShift::LSE_LAYOUT) & 3u;
  v.reserved = (wire >> 18) & 0x3FFFu;
  return v;
}

static_assert(varlen_to_wire(varlen_from_wire(0x1150Bu)) == 0x1150Bu,
              "varlen_from_wire must invert varlen_to_wire");
static_assert(varlen_to_wire(varlen_from_wire(0x1313u)) == 0x1313u,
              "varlen_from_wire must invert varlen_to_wire");
static_assert(varlen_to_wire(varlen_from_wire(0xFFFFFFFFu)) == 0xFFFFFFFFu,
              "the inverse must be total, reserved bits included");

// The addressing bytes, i.e. everything except LSE_LAYOUT. Two configurations
// with the same addressing describe the same memory traversal and differ only
// in where the logsumexp rows land.
//
// Takes the WIRE, not VarlenBits, and is the one predicate that should: its
// caller is a grid calculator, which sees only the int32 the generated params
// struct carries. It also has to agree with the kernel's own
// `(Varlen_bits & 0xFFFF) != 0` mask, and stating that mask once here is what
// makes "agree" checkable.
constexpr uint32_t
varlen_addressing(uint32_t wire) {
  return wire & 0xFFFFu;
}

// One side's addressing: its mode, and the operands that mode selects.
//
// Built once per side, so nothing below takes a `k_side` flag -- the side is
// decided at construction and every question after that is a member. That was
// the point: threading a bool through seven predicates made every call site
// restate which half it meant, and made the two halves look like different
// operations when they are the same one twice.
//
// Holds extracted numbers rather than tensors, so it stays constexpr and can be
// tested without constructing a TensorView. varlen_addressing_of() below builds
// one from the operands a shim actually has.
class VarlenAddressing {
 public:
  constexpr VarlenAddressing(VarlenMode mode,
                             int32_t batch_extent, int32_t token_extent,
                             int32_t info0_len, int32_t info1_len,
                             int32_t caller_max_seqlen, int32_t tensor_max_seqlen)
    : mode_(mode),
      batch_extent_(batch_extent), token_extent_(token_extent),
      info0_len_(info0_len), info1_len_(info1_len),
      caller_max_seqlen_(caller_max_seqlen), tensor_max_seqlen_(tensor_max_seqlen) {}

  // Whether Max_seqlen comes from the CALLER rather than the tensor's extent.
  //
  // Tensor presence is the wrong predicate and was the bug this replaced: a THD
  // side with LENGTH == MAX carries no length array, so keying off seqinfo_?0
  // left max_seqlen at K.size(2) -- the total packed token count, not the
  // per-sequence maximum -- and decode_addressing then read seqlen = that total
  // and row_off = z * that total, i.e. out of bounds. It also let dense bits
  // with a stray seqinfo_?0 replace a correct extent with an unset zero.
  //
  // Only a fully dense side may trust the tensor: there BHSD + MAX means the
  // extent IS the per-sequence length.
  constexpr bool uses_caller_max_seqlen() const {
    return mode_.stacked != VarlenStacked::BHSD
        || mode_.length != VarlenLength::MAX
        || mode_.position != VarlenPosition::IMPLIED;
  }

  // Is the caller's Max_seqlen actually set, where the mode reads one?
  //
  // Zero is the header's default, so "never set it" and "meant zero" arrive as
  // the same input -- and the result is silent rather than merely wrong. The
  // forward grid's x extent is cdiv(max_seqlen_q, BLOCK_M), so zero launches NO
  // workgroups, leaves Out at whatever was allocated, and returns hipSuccess.
  //
  // Nothing else catches it: zero satisfies every lower bound in extents_ok()
  // by construction, valid() sees well-formed bits, and both sequence counts
  // agree. Kept separate from those two because it is not a property of the
  // bits or of the arrays -- it is the one scalar operand that has no
  // recognisable "unset" value other than a wrong answer.
  constexpr bool max_seqlen_ok() const {
    return !uses_caller_max_seqlen() || caller_max_seqlen_ > 0;
  }

  constexpr int32_t max_seqlen() const {
    return uses_caller_max_seqlen() ? caller_max_seqlen_ : tensor_max_seqlen_;
  }

  // Cheap well-formedness: the bits, and whether the arrays this mode reads
  // were supplied. No device reads, so it runs on every launch -- unlike the
  // prefix-sum and non-overlap preconditions, which cost a sync.
  //
  // Catches what would otherwise reach the kernel and fault or misaddress: an
  // out-of-range field, REUSE without CUMULATIVE (only CUMULATIVE makes
  // seqinfo_?0 hold positions as well as lengths), and a mode whose array is
  // absent -- the kernel would tl.load from null.
  constexpr bool valid() const {
    if (mode_.length > VarlenLength::INDIVIDUAL
        || mode_.position > VarlenPosition::ARRAY) {
      return false;                                 // 3 is not a value
    }
    if (mode_.reserved) {
      return false;
    }
    if (mode_.position == VarlenPosition::REUSE
        && mode_.length != VarlenLength::CUMULATIVE) {
      return false;                                 // info0 holds no position
    }
    if (mode_.length != VarlenLength::MAX && !info0_len_) {
      return false;                                 // length source missing
    }
    if (mode_.position == VarlenPosition::ARRAY && !info1_len_) {
      return false;                                 // position source missing
    }
    if (mode_.length == VarlenLength::MAX && info0_len_) {
      return false;                                 // supplied but never read
    }
    return true;
  }

  // Do the arrays and the tensor hold N sequences' worth?
  //
  // LOWER BOUNDS, not equality: a caller may legitimately hand over a bigger
  // buffer than it needs -- a KV cache sized for the worst-case batch, or a
  // view into a larger allocation. Too SHORT is the only error, because that is
  // what the kernel reads past.
  constexpr bool extents_ok(int32_t n) const {
    if (mode_.length == VarlenLength::CUMULATIVE && info0_len_ < n + 1) {
      return false;                        // read at [z] and [z+1]
    }
    if (mode_.length == VarlenLength::INDIVIDUAL && info0_len_ < n) {
      return false;                        // read at [z]
    }
    if (mode_.position == VarlenPosition::ARRAY && info1_len_ < n) {
      return false;                        // read at [z]
    }
    if (mode_.stacked == VarlenStacked::BHSD && batch_extent_ < n) {
      // batch_index = z on a BHSD side, so its tensor needs N slots. N comes
      // off the Q side alone, which is how mixed 0x000B reached K.size(0) with
      // a z the K tensor never had.
      return false;
    }
    // The TOKEN axis, where the host can know how far the kernel will reach.
    //
    // THD + IMPLIED puts sequence z at row z * max_seqlen, so the last row read
    // is N * max_seqlen and a shorter axis is an out-of-bounds launch. The Q
    // side cannot trip this -- seq_count() derives N by dividing that very
    // extent -- but K's N comes from Q, so a stacked K with MAX (side 0x01)
    // against a short K.size(2) reaches past the end with nothing to stop it.
    //
    // int64 because N * max_seqlen is a whole-batch quantity: at N = 64K and
    // max_seqlen = 64K the product leaves int32 while both factors are
    // unremarkable.
    const int64_t per_seq = max_seqlen();
    if (mode_.stacked == VarlenStacked::THD
        && mode_.position == VarlenPosition::IMPLIED
        && static_cast<int64_t>(token_extent_) < static_cast<int64_t>(n) * per_seq) {
      return false;
    }
    // A BHSD side under IMPLIED reads rows 0..seqlen of its own slice, and
    // seqlen <= max_seqlen. Under REUSE/ARRAY it starts at a nonzero row_off
    // within that slice, so this stays a lower bound there rather than the
    // reach -- see the note below on why the contents are not read.
    if (mode_.stacked == VarlenStacked::BHSD
        && static_cast<int64_t>(token_extent_) < per_seq) {
      return false;
    }
    // Under REUSE or ARRAY the row offsets come from an array whose CONTENTS
    // decide the reach, and reading them costs a device sync. Those stay
    // documented preconditions (the prefix-sum and non-overlap assumptions)
    // rather than checks, which is why this is not a complete bound.
    return true;
  }

  // The logsumexp token pitch, which reads slot [N] of whichever array supplies
  // POSITION -- the length array under REUSE, the position array under ARRAY.
  // A stacked LSE runs to the batch's total token count instead of padding each
  // row-group, and that total lives in the prefix sum's last slot.
  //
  // Q SIDE ONLY, and that is why it is not folded into extents_ok(). LSE is
  // indexed by Q's addressing, so the [N] slot is a property of the OUTPUT, not
  // of a side's mode -- the two modes are symmetric and neither needs it for
  // its own addressing. K's position array needs N entries, not N+1, and
  // demanding otherwise would reject a legitimate one.
  constexpr bool lse_pitch_ok(int32_t n) const {
    if (mode_.stacked != VarlenStacked::THD) {
      return true;                         // batched: the pitch is max_seqlen
    }
    if (mode_.position == VarlenPosition::REUSE) {
      return info0_len_ >= n + 1;
    }
    if (mode_.position == VarlenPosition::ARRAY) {
      return info1_len_ >= n + 1;
    }
    return true;                           // IMPLIED: N * max_seqlen, no read
  }

  // N off this side. Q's is the launch's; K's is not consulted.
  //
  // STACKED + MAX is uniform stacking: every sequence is exactly max_seqlen
  // rows, so the token axis holds N of them. There is no array to count, but
  // the count is not undetermined either.
  //
  // -1 where it genuinely cannot be derived: a max_seqlen of zero, or a token
  // axis that is not a whole multiple of it, which is not a uniform stacking
  // and must not be rounded into one. The shims reject that rather than
  // guessing, because a wrong z extent gives in-bounds addresses of the wrong
  // rows.
  constexpr int32_t seq_count() const {
    if (mode_.stacked == VarlenStacked::BHSD) {
      return batch_extent_;                  // one sequence per batch slot
    }
    if (mode_.length == VarlenLength::CUMULATIVE) {
      return info0_len_ - 1;                 // the array is (N+1,)
    }
    if (mode_.length == VarlenLength::INDIVIDUAL) {
      return info0_len_;                     // the array is (N,)
    }
    const int32_t per_seq = max_seqlen();
    if (per_seq <= 0 || token_extent_ < 0 || token_extent_ % per_seq != 0) {
      return -1;                             // not a uniform stacking
    }
    return token_extent_ / per_seq;
  }

  // The same count WITHOUT consulting the bits: off the batch axis and whatever
  // length array was handed in, which is how the shims computed it before
  // varlen_bits existed. Keyed on array PRESENCE where seq_count() is keyed on
  // STACKED, so the two disagree exactly when the operands do not match the
  // declared layout -- a padded call whose seqinfo_q0 is the wrong length, say.
  // Not a crash: a kernel over the wrong number of programs, addressing
  // in-bounds rows that are the wrong ones. Worth one comparison.
  //
  // -1 where there is nothing independent to compare and the caller skips it:
  // INDIVIDUAL, where "array size minus one" is not the count; and stacked MAX,
  // where the only other source is a batch axis that is 1 under THD and would
  // disagree with every correct count above one.
  constexpr int32_t independent_seq_count() const {
    if (mode_.length == VarlenLength::INDIVIDUAL) {
      return -1;
    }
    if (mode_.stacked == VarlenStacked::THD && mode_.length == VarlenLength::MAX) {
      return -1;
    }
    return info0_len_ > 0 ? info0_len_ - 1 : batch_extent_;
  }

 private:
  VarlenMode mode_;
  int32_t batch_extent_;
  int32_t token_extent_;
  int32_t info0_len_;
  int32_t info1_len_;
  int32_t caller_max_seqlen_;
  int32_t tensor_max_seqlen_;
};

// Whole-word checks, the ones that are not per side.
constexpr bool
varlen_valid(VarlenBits v, const VarlenAddressing& q, const VarlenAddressing& k) {
  if (v.reserved) {
    return false;
  }
  // Two bits for future room, but only HT and TH defined; the kernel branches
  // `layout == 0 ? HT : TH`, so 2 and 3 would become TH silently.
  if (v.lse_layout > VarlenLseLayout::TH) {
    return false;
  }
  return q.valid() && k.valid();
}

inline int32_t
varlen_seqinfo_len(const T1& t) {
  return t ? static_cast<int32_t>(t.size(0)) : 0;
}

// Build one side's addressing from the operands a shim holds.
//
// `base` is that side's data tensor: its batch axis feeds the BHSD count and
// the BHSD extent check, its token axis the uniform-stacking division. The
// TensorView default constructor leaves sizes INDETERMINATE (only
// get_null_tensor zeroes them), so array lengths go through
// varlen_seqinfo_len() rather than an unguarded size(0).
inline VarlenAddressing
varlen_addressing_of(VarlenMode mode, const T4& base,
                     const T1& info0, const T1& info1, int32_t caller_max_seqlen) {
  return VarlenAddressing(mode,
                          static_cast<int32_t>(base.size(0)),
                          static_cast<int32_t>(base.size(2)),
                          varlen_seqinfo_len(info0),
                          varlen_seqinfo_len(info1),
                          caller_max_seqlen,
                          static_cast<int32_t>(base.size(2)));
}

// N, for the z extent of every BACKWARD grid. The backward kernels take no
// sequence-count argument -- they read tl.num_programs(2) -- so the grid IS the
// only place N is communicated to them, and the four grid calculators that
// exist (preprocess, dk_dv, dq, fuse) must all produce the same number.
//
// Templated because OpAttnBwdParams lives in a generated header this one does
// not include, and because there is nothing type-specific about it beyond the
// fields it reads. This is one of the two places the WIRE is legitimate: a grid
// calculator sees only the int32 the generated struct carries. attn_bwd()
// validates the count before launching, so a negative cannot reach a grid.
template<typename BwdParams>
inline uint32_t
varlen_bwd_seq_count(const BwdParams* params) {
  const VarlenBits v = varlen_from_wire(static_cast<uint32_t>(params->varlen_bits));
  // Every operand, not just the ones seq_count() happens to read: an object
  // that under-reports its own position array would answer valid() and
  // extents_ok() wrongly the moment a grid calculator asked either.
  const VarlenAddressing q(v.qmode,
                           static_cast<int32_t>(params->Q->size(0)),
                           static_cast<int32_t>(params->Q->size(2)),
                           varlen_seqinfo_len(*params->seqinfo_q0),
                           varlen_seqinfo_len(*params->seqinfo_q1),
                           params->max_seqlen_q,
                           static_cast<int32_t>(params->Q->size(2)));
  return static_cast<uint32_t>(q.seq_count());
}

} // AOTRITON_NS::v3::flash::internal

#endif
