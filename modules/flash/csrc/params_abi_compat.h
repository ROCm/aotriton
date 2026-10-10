// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AOTRITON_MODULES_FLASH_CSRC_PARAMS_ABI_COMPAT_H
#define AOTRITON_MODULES_FLASH_CSRC_PARAMS_ABI_COMPAT_H

// The params structs as PAST RELEASES declared them, plus their translation
// into the current ones. Nothing here is used by a current caller; it exists so
// that a binary compiled against an older AOTriton header keeps working when
// this library is dropped in beneath it. The promise is a library swap, not
// source compatibility.
//
// That promise is the reason for the file name. A reader who opens it because
// something looks redundant -- a struct that duplicates one in flash.h, a copy
// function that seems to do nothing -- should learn from the include line alone
// that the redundancy IS the feature, and that deleting it breaks software
// nobody in this repository can rebuild.
//
// WHY THIS AND NOT A LAYOUT RULE. Compatibility is not maintained by keeping
// the current structs shaped like the old ones -- no field is pinned, no
// padding is reserved, nothing must stay a prefix of anything. Each historical
// layout is DESCRIBED here instead, so fields are free to be added and removed
// from the live structs at will. That freedom is the entire return on the cost
// of this file.
//
// MAINTAINING IT.
//   * Adding a version: add its layout as a versioned_attn_*_params<N>
//     specialisation with a translate_to_current(), and add X(N) to that
//     family's list.
//   * Retiring a version: delete both. The list IS the supported set, so a
//     caller at a retired version is refused rather than mistranslated; there
//     is no separate minimum-version constant to update in step.
//
// The floor is 0.13b (fwd kVersion 3, bwd kVersion 6), verified against the
// 0.13b tag field-for-field. There is no plan to reach further back.
//
// THE TWO FAMILIES HAVE INDEPENDENT LINEUPS, DELIBERATELY. Forward is at 3 -> 4
// while backward is at 6 -> 7; the numbers have never been in step and are not
// meant to be, because the two structs change for unrelated reasons. Hence two
// separate version lists rather than one shared constant -- a single "supported
// version" would force a bump on the family that did not change.
//
// PER-FAMILY PRACTICE, NOT SHARED MACHINERY. Another kernel family wanting this
// should copy the pattern, not include this header. The macro below looks
// generic and is not: the layouts, the field-by-field copies and the translation
// rules are specific to these structs, and hoisting the ten lines that are
// reusable would buy a dependency in exchange for nothing.

#include <aotriton/config.h>
#include <aotriton/flash.h>

// varlen_bits_of(): the one field that does not survive translation as itself.
// Everything else is copied across; varlen_type becomes varlen_bits.
#include "varlen.h"

#include <cstdint>

namespace AOTRITON_NS::v3::flash::internal {

// The older versions this build can still read, one entry per described
// layout. This list IS the compatibility window: a version in it can be
// translated, a version below it cannot, and there is no separate
// "minimum supported" constant to drift out of step with the specialisations
// that actually exist.
//
// Deliberately no names like kVersionWithVarlenBits. Naming ABI versions after
// whatever feature prompted them stops paying off as they pile up -- nobody
// recalls which Fedora or Firefox release a codename stands for either. The
// current version is `attn_*_params::kVersion` and the old ones are numbers.
#define AOTRITON_FLASH_FWD_LEGACY_VERSIONS(X) X(3)
#define AOTRITON_FLASH_BWD_LEGACY_VERSIONS(X) X(6)

// ---------------------------------------------------------------------------
// Per-version params translation.
//
// ABI compatibility here is NOT the accident of a field happening to land in
// padding -- it is an explicit translation of the caller's params object,
// selected by params_version. That is what makes fields free to add and remove
// across kVersions: the older layout is described verbatim below, so nothing
// about the current struct's shape has to be preserved to keep an old binary
// working. The goal is a drop-in library swap (a binary built against 0.13
// headers, run against this library), not source compatibility.
//
// Adding a kVersion later means adding another struct here and another arm to
// the upgrade; it does not mean constraining what the current struct may look
// like.

// The family. Primary template is DECLARED and not defined, so naming a version
// nobody described is a compile error rather than a silent empty struct.
//
// Fully internal: no AOTRITON_API, never in a public header. These types exist
// to read bytes a past release's header laid out, and offering them to callers
// would only invite someone to keep writing that layout.
template<int Version> struct versioned_attn_fwd_params;
template<int Version> struct versioned_attn_bwd_params;

// attn_fwd_params exactly as kVersion 3 declared it. Field-for-field, in order,
// including `varlen_type`, which the current struct no longer has.
template<>
struct versioned_attn_fwd_params<3> {
  T4       Q;
  T4       K;
  T4       V;
  T4       B;
  T2       A;
  float    Sm_scale;
  T2       L;
  T4       Out;
  T1       cu_seqlens_q;
  T1       cu_seqlens_k;
  int32_t  Max_seqlen_q = 0;
  int32_t  Max_seqlen_k = 0;
  T1       seq_strides_q;
  T1       seq_strides_k;
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

  static attn_fwd_params translate_to_current(const versioned_attn_fwd_params<3>& old);
};

// attn_bwd_params exactly as kVersion 6 declared it.
template<>
struct versioned_attn_bwd_params<6> {
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
  mutable LT2       D;
  T1        cu_seqlens_q;
  T1        cu_seqlens_k;
  int32_t   Max_seqlen_q = 0;
  int32_t   Max_seqlen_k = 0;
  T1        seq_strides_q;
  T1        seq_strides_k;
  float     dropout_p;
  T0        philox_seed_ptr;
  T0        philox_offset1;
  uint64_t  philox_offset2;
  int8_t    causal_type;
  int8_t    varlen_type = 0;
  int32_t   window_left;
  int32_t   window_right;
  mutable LT4       DQ_ACC;

  static attn_bwd_params translate_to_current(const versioned_attn_bwd_params<6>& old);
};

// Translate a kVersion 3 params object into the current one.
//
// Every field is copied by NAME, not by layout, so the two structs are free to
// diverge in shape however they like -- which is the whole point of translating
// rather than reinterpreting one as the other.
//
// `varlen_type` is the only field that does not survive as itself: it becomes
// varlen_bits, through the tensor-presence rules above. Nothing is lost, because
// the four rows that enum could express are exactly the four varlen_bits_of()
// produces.
inline attn_fwd_params
versioned_attn_fwd_params<3>::translate_to_current(const versioned_attn_fwd_params<3>& old) {
  attn_fwd_params p;
  p.Q = old.Q;
  p.K = old.K;
  p.V = old.V;
  p.B = old.B;
  p.A = old.A;
  p.Sm_scale = old.Sm_scale;
  p.L = old.L;
  p.Out = old.Out;
  p.seqinfo_q0 = old.cu_seqlens_q;
  p.seqinfo_k0 = old.cu_seqlens_k;
  p.Max_seqlen_q = old.Max_seqlen_q;
  p.Max_seqlen_k = old.Max_seqlen_k;
  p.seqinfo_q1 = old.seq_strides_q;
  p.seqinfo_k1 = old.seq_strides_k;
  p.dropout_p = old.dropout_p;
  p.philox_seed_ptr = old.philox_seed_ptr;
  p.philox_offset1 = old.philox_offset1;
  p.philox_offset2 = old.philox_offset2;
  p.philox_seed_output = old.philox_seed_output;
  p.philox_offset_output = old.philox_offset_output;
  p.encoded_softmax = old.encoded_softmax;
  p.persistent_atomic_counter = old.persistent_atomic_counter;
  p.causal_type = old.causal_type;
  p.window_left = old.window_left;
  p.window_right = old.window_right;
  p.varlen_bits = varlen_bits_of(old.varlen_type,
                                 bool(old.cu_seqlens_q),
                                 bool(old.seq_strides_q));
  return p;
}

// Translate a kVersion 6 params object into the current one. See above.
inline attn_bwd_params
versioned_attn_bwd_params<6>::translate_to_current(const versioned_attn_bwd_params<6>& old) {
  attn_bwd_params p;
  p.Q = old.Q;
  p.K = old.K;
  p.V = old.V;
  p.B = old.B;
  p.Sm_scale = old.Sm_scale;
  p.Out = old.Out;
  p.DO = old.DO;
  p.DK = old.DK;
  p.DV = old.DV;
  p.DQ = old.DQ;
  p.DB = old.DB;
  p.L = old.L;
  p.D = old.D;
  p.seqinfo_q0 = old.cu_seqlens_q;
  p.seqinfo_k0 = old.cu_seqlens_k;
  p.Max_seqlen_q = old.Max_seqlen_q;
  p.Max_seqlen_k = old.Max_seqlen_k;
  p.seqinfo_q1 = old.seq_strides_q;
  p.seqinfo_k1 = old.seq_strides_k;
  p.dropout_p = old.dropout_p;
  p.philox_seed_ptr = old.philox_seed_ptr;
  p.philox_offset1 = old.philox_offset1;
  p.philox_offset2 = old.philox_offset2;
  p.causal_type = old.causal_type;
  p.window_left = old.window_left;
  p.window_right = old.window_right;
  p.DQ_ACC = old.DQ_ACC;
  p.varlen_bits = varlen_bits_of(old.varlen_type,
                                 bool(old.cu_seqlens_q),
                                 bool(old.seq_strides_q));
  return p;
}

// Translate a params object written by an OLDER caller into the current one.
//
// Dispatches on the version the caller actually passed. An earlier cut tested
// a range and then hard-coded `versioned_attn_fwd_params<3>`; the two agreed
// only because 3 happened to be the single legacy version, and the day a second
// one is added that code would reinterpret a v2 object as a v3 -- silently, and
// as a wrong field layout rather than a crash.
//
// Returns false when the version is older than anything described here, which
// is also how the shims spell "too old to support": the set of specialisations
// IS the supported set.
//
// `in` is a reference to the CALLER's object, laid out by the header that
// binary was built against, so it is read only through the type describing that
// layout -- never as the current struct.
#define AOTRITON_FLASH_DEFINE_TRANSLATE(NAME, PARAMS, VERSION_LIST)          \
  inline bool                                                                \
  NAME(const PARAMS& in, int32_t params_version, PARAMS* out) {              \
    switch (params_version) {                                                \
      VERSION_LIST(AOTRITON_FLASH_TRANSLATE_CASE)                            \
      default:                                                               \
        return false;                                                        \
    }                                                                        \
  }

#define AOTRITON_FLASH_TRANSLATE_CASE(N)                                     \
  case N:                                                                    \
    *out = versioned_params<N>::translate_to_current(                        \
        *reinterpret_cast<const versioned_params<N>*>(&in));                 \
    return true;

// versioned_params is the per-family alias the CASE macro expands against, so
// one case body serves both families.
#define versioned_params versioned_attn_fwd_params
AOTRITON_FLASH_DEFINE_TRANSLATE(translate_fwd_params, attn_fwd_params,
                                AOTRITON_FLASH_FWD_LEGACY_VERSIONS)
#undef versioned_params

#define versioned_params versioned_attn_bwd_params
AOTRITON_FLASH_DEFINE_TRANSLATE(translate_bwd_params, attn_bwd_params,
                                AOTRITON_FLASH_BWD_LEGACY_VERSIONS)
#undef versioned_params

#undef AOTRITON_FLASH_TRANSLATE_CASE
#undef AOTRITON_FLASH_DEFINE_TRANSLATE
#undef AOTRITON_FLASH_FWD_LEGACY_VERSIONS
#undef AOTRITON_FLASH_BWD_LEGACY_VERSIONS

} // AOTRITON_NS::v3::flash::internal

#endif
