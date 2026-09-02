// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Standalone test for modules/flash/csrc/varlen.h -- the host-side
// translation from AOTriton's `VarlenType` enum into the `varlen_bits` word the
// GPU kernels decode. There is no C++ test harness anywhere in this project (no
// gtest, no CMake add_test target), so this is a plain main() rather than an
// invented framework. It is NOT part of the CMake build: v3src/CMakeLists.txt
// globs "modules/*/csrc/*.cc" recursively, and this file deliberately lives
// under modules/flash/tests/ (no "csrc" path component) so it is never swept
// into libaotriton_v2.so.
//
// It exercises the FOUR rows of the legacy varlen table plus the version-gated
// upgrade path. A dense-only test would be worthless here: it would pass three
// of the four wrong implementations, since dense is the one row where every
// input is zero and every plausible bug still produces 0x0000.
//
// It does NOT test varlen_to_wire() against the spec's hex on its own. That is
// not an omission: csrc/varlen.h carries a static_assert table pinning every
// legacy row through varlen_bits_of(), so the property most at risk of platform
// variation is checked by every compiler that reads the header rather than by
// whichever one happens to run this binary. The prior art on the flydsl branch needed a
// runtime `varlen_selfcheck()` only because it kept a SECOND representation (a
// bitfield decoder alongside a shift encoder) that clang would not let it
// check at compile time; under the single-representation design there is
// nothing left for such a call to compare.
//
// Build -- header-only, so no generated per-kernel header and no library link:
//
//   g++ -std=c++20 -D__HIP_PLATFORM_AMD__=1
//       -I . -I include -I <any build dir>/include -I <hip headers>
//       modules/flash/tests/test_varlen_translation.cc -o /tmp/varlen_test
//   /tmp/varlen_test
//
// The build dir is only needed for aotriton/config.h.

#include <aotriton/config.h>

#include "../csrc/varlen.h"
#include "../csrc/params_abi_compat.h"

#include <cstdio>

// util.h declares `extern template class TensorView<N>`, so its member
// definitions normally come from libaotriton_v2.so. Instantiate them here
// instead -- that is what keeps this a two-command test with no library build,
// and TensorView is a header-only class template, so nothing is lost.
template class AOTRITON_NS::TensorView<1>;
template class AOTRITON_NS::TensorView<2>;
template class AOTRITON_NS::TensorView<4>;

// Same reason, for the params constructors: they are defined (empty) in the
// shim translation units that end up in the library. Constructing a params
// object is new here -- the test grew one when the compatibility path became a
// struct translation rather than a scalar mapping.
namespace AOTRITON_NS::v3::flash {
attn_fwd_params::attn_fwd_params() {}
attn_bwd_params::attn_bwd_params() {}
}

namespace {

using AOTRITON_NS::v3::flash::VarlenBits;
using AOTRITON_NS::v3::flash::VarlenLength;
using AOTRITON_NS::v3::flash::VarlenPosition;
using AOTRITON_NS::v3::flash::VarlenStacked;

// The named configurations this test needs, defined HERE rather than pulled in
// from a production header. Production carries no such constants -- the only
// place a configuration is spelled out is varlen_bits_of(), which is the thing
// under test -- and a test that reused them could not detect a change to them.
// Written out as fields so a wrong expectation looks wrong.
constexpr VarlenBits kDense = {};

constexpr VarlenBits kCompact = {
  .q_stacked = VarlenStacked::THD,
  .q_length = VarlenLength::CUMULATIVE,
  .q_position = VarlenPosition::REUSE,
  .k_stacked = VarlenStacked::THD,
  .k_length = VarlenLength::CUMULATIVE,
  .k_position = VarlenPosition::REUSE,
};

// torch.nn.attention.varlen's `seqused_k` on a packed KV cache: the K side
// takes its LENGTH from an individual array and its POSITION from a cumulative
// one. Two different tensors, which is why seqinfo_k0/k1 are named by role.
// No VarlenType can spell this, so it only ever arrives through the struct.
constexpr VarlenBits kSequsedKOnPacked = {
  .q_stacked = VarlenStacked::THD,
  .q_length = VarlenLength::CUMULATIVE,
  .q_position = VarlenPosition::REUSE,
  .k_stacked = VarlenStacked::THD,
  .k_length = VarlenLength::INDIVIDUAL,
  .k_position = VarlenPosition::ARRAY,
};

int g_failures = 0;

void
check(bool cond, const char* what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++g_failures;
  } else {
    std::fprintf(stderr, "  ok: %s\n", what);
  }
}

void
check_eq(uint32_t got, uint32_t want, const char* what) {
  if (got != want) {
    std::fprintf(stderr, "FAIL: %s -- got 0x%04x, want 0x%04x\n", what, got, want);
    ++g_failures;
  } else {
    std::fprintf(stderr, "  ok: %s == 0x%04x\n", what, got);
  }
}

using namespace AOTRITON_NS::v3::flash;
// varlen.h lives in a nested namespace so it stays out of the user API; a test
// of that header is exactly who should reach into it.
using namespace AOTRITON_NS::v3::flash::internal;

// One row of the legacy table, as the triple the pre-varlen_bits shim actually
// keyed on: the enum only supplied the padded SIGN, tensor presence supplied
// the rest.
void
check_row(const char* name, int8_t varlen_type, bool cu_seqlens_q, bool seq_strides_q,
          uint32_t want_wire) {
  const uint32_t got = varlen_to_wire(varlen_bits_of(varlen_type, cu_seqlens_q, seq_strides_q));
  std::fprintf(stderr, "-- %s: varlen_type=%d cu_seqlens_q=%d seq_strides_q=%d\n",
               name, int(varlen_type), int(cu_seqlens_q), int(seq_strides_q));
  check_eq(got, want_wire, name);
}

void
check_seq_count(const char* name, uint32_t wire, int32_t q0_len, int32_t q_batch,
                int32_t want) {
  const int32_t got = varlen_seq_count(wire, q0_len, q_batch);
  if (got != want) {
    std::fprintf(stderr, "FAIL: %s -- N got %d, want %d\n", name, got, want);
    ++g_failures;
  } else {
    std::fprintf(stderr, "  ok: %s N == %d\n", name, got);
  }
}

void
test_legacy_table() {
  // Row 1: dense. No arrays at all.
  check_row("dense", VarlenType::None, false, false, 0x0000u);

  // Row 2: compact varlen. cu_seqlens_q supplied, seq_strides_q is null --
  // which is the ONLY thing separating this row from strided.
  check_row("compact", VarlenType::CompactVarlen, true, false, 0x0B0Bu);

  // Row 3: padded varlen. BHSD with one sequence per batch slot; the enum is
  // load-bearing here and nowhere else, because it is the only input that
  // distinguishes this from compact.
  check_row("padded", VarlenType::PaddedVarlen, true, false, 0x0202u);

  // Row 4: strided varlen. Position read from a dedicated array.
  check_row("strided", VarlenType::StridedVarlen, true, true, 0x1313u);
}

void
test_legacy_quirks() {
  // These three are the reason varlen_bits_of() takes tensor presence and not
  // just the enum. Each reproduces what the OLD code did, which is what
  // backward compatibility means -- not what the enum's name suggests.

  // The enum was never set by callers that predate it; cu_seqlens_q alone made
  // Num_seqlens positive, and positive meant compact.
  check_row("None + cu_seqlens_q => compact",
            VarlenType::None, true, false, 0x0B0Bu);

  // The kernel picked strided by testing seq_strides_q for null, so the enum
  // saying "compact" did not override a supplied position array.
  check_row("CompactVarlen + seq_strides_q => strided",
            VarlenType::CompactVarlen, true, true, 0x1313u);

  // Negating a zero sequence count leaves it zero, so padded without the
  // length array was dense.
  check_row("PaddedVarlen without cu_seqlens_q => dense",
            VarlenType::PaddedVarlen, false, false, 0x0000u);
}

// A kVersion 3 object, as an old binary would have filled it, translated into
// the current one. This is the whole ABI story: the old layout is DESCRIBED,
// not reinterpreted field by field, so removing varlen_type from the current
// struct costs nothing.
void
test_version_translation() {
  auto fwd_wire = [](int8_t varlen_type, bool cu_q, bool ss_q) {
    versioned_attn_fwd_params<3> old{};
    old.varlen_type = varlen_type;
    // Presence is what the old shim keyed on, and a TensorView is truthy when
    // it has a base pointer, so a fake non-null one is enough here.
    int32_t storage[2] = {0, 0};
    if (cu_q) {
      old.cu_seqlens_q = T1(reinterpret_cast<intptr_t>(storage), {2}, {1}, AOTRITON_NS::DType::kInt32);
    }
    if (ss_q) {
      old.seq_strides_q = T1(reinterpret_cast<intptr_t>(storage), {2}, {1}, AOTRITON_NS::DType::kInt32);
    }
    old.causal_type = CausalType::WindowedAttention;
    old.window_left = 17;
    old.window_right = 23;
    old.Max_seqlen_q = 128;
    const attn_fwd_params now =
        versioned_attn_fwd_params<3>::translate_to_current(old);
    // Everything else must survive the translation, not just varlen. A copy
    // that dropped a field would still pass a varlen-only assertion.
    check(now.causal_type == CausalType::WindowedAttention, "fwd v3: causal_type survives");
    check(now.window_left == 17 && now.window_right == 23, "fwd v3: window survives");
    check(now.Max_seqlen_q == 128, "fwd v3: Max_seqlen_q survives");
    return varlen_to_wire(now.varlen_bits);
  };

  check_eq(fwd_wire(VarlenType::None, false, false), 0x0000u, "fwd v3 -> dense");
  check_eq(fwd_wire(VarlenType::CompactVarlen, true, false), 0x0B0Bu, "fwd v3 -> compact");
  check_eq(fwd_wire(VarlenType::PaddedVarlen, true, false), 0x0202u, "fwd v3 -> padded");
  check_eq(fwd_wire(VarlenType::StridedVarlen, true, true), 0x1313u, "fwd v3 -> strided");
  // The quirks again, this time through the translation rather than the raw
  // mapping -- an old binary really does reach the library this way.
  check_eq(fwd_wire(VarlenType::None, true, false), 0x0B0Bu,
           "fwd v3 -> None + cu_seqlens_q is compact");

  // A translated object never carries an LSE layout: kVersion 3 had no way to
  // ask for one, so _TH can only arrive from a caller new enough to have the
  // struct. Anything else would be inventing a request the caller never made.
  versioned_attn_fwd_params<3> plain{};
  check(versioned_attn_fwd_params<3>::translate_to_current(plain).varlen_bits.lse_layout
            == VarlenLseLayout::HT,
        "fwd v3 translates to the default LSE layout");

  // Backward, same shape. DQ_ACC is checked because it is the one field that
  // sits AFTER the varlen fields in the v6 layout.
  versioned_attn_bwd_params<6> oldb{};
  oldb.varlen_type = VarlenType::CompactVarlen;
  int32_t storage[2] = {0, 0};
  oldb.cu_seqlens_q = T1(reinterpret_cast<intptr_t>(storage), {2}, {1}, AOTRITON_NS::DType::kInt32);
  oldb.causal_type = CausalType::WindowedAttention;
  oldb.window_left = 5;
  const attn_bwd_params nowb =
      versioned_attn_bwd_params<6>::translate_to_current(oldb);
  check_eq(varlen_to_wire(nowb.varlen_bits), 0x0B0Bu, "bwd v6 -> compact");
  check(nowb.causal_type == CausalType::WindowedAttention, "bwd v6: causal_type survives");
  check(nowb.window_left == 5, "bwd v6: window survives");

  // Dispatch by the version actually passed, through the same entry point the
  // shims use. The rejection case is the one that matters: it is the signature
  // of the bug this replaced, where a range test was paired with a hard-coded
  // versioned_attn_fwd_params<3> and any other old version would have been
  // reinterpreted as a v3 rather than refused.
  {
    versioned_attn_fwd_params<3> v3{};
    v3.varlen_type = VarlenType::CompactVarlen;
    int32_t seq[2] = {0, 0};
    v3.cu_seqlens_q = T1(reinterpret_cast<intptr_t>(seq), {2}, {1},
                         AOTRITON_NS::DType::kInt32);
    attn_fwd_params out;
    const auto& as_current = *reinterpret_cast<const attn_fwd_params*>(&v3);
    check(translate_fwd_params(as_current, 3, &out), "fwd: version 3 translates");
    check_eq(varlen_to_wire(out.varlen_bits), 0x0B0Bu, "fwd: version 3 -> compact");
    check(!translate_fwd_params(as_current, 2, &out),
          "fwd: version 2 is refused, not read as a v3");
    check(!translate_fwd_params(as_current, 0, &out), "fwd: version 0 is refused");

    versioned_attn_bwd_params<6> v6{};
    v6.varlen_type = VarlenType::PaddedVarlen;
    v6.cu_seqlens_q = T1(reinterpret_cast<intptr_t>(seq), {2}, {1},
                         AOTRITON_NS::DType::kInt32);
    attn_bwd_params outb;
    const auto& b_as_current = *reinterpret_cast<const attn_bwd_params*>(&v6);
    check(translate_bwd_params(b_as_current, 6, &outb), "bwd: version 6 translates");
    check_eq(varlen_to_wire(outb.varlen_bits), 0x0202u, "bwd: version 6 -> padded");
    check(!translate_bwd_params(b_as_current, 5, &outb),
          "bwd: version 5 is refused, not read as a v6");
    // The current version is not a legacy row: the shims handle it by not
    // translating at all, so the dispatcher must not claim it either.
    check(!translate_fwd_params(as_current, attn_fwd_params::kVersion, &out),
          "fwd: current version is not in the legacy table");
  }

  // A current-version caller needs no translation at all: the struct IS the
  // answer, which is what removing the enum from the API bought.
  attn_fwd_params fresh;
  fresh.varlen_bits.q_stacked = VarlenStacked::THD;
  fresh.varlen_bits.q_length = VarlenLength::CUMULATIVE;
  fresh.varlen_bits.q_position = VarlenPosition::REUSE;
  fresh.varlen_bits.k_stacked = VarlenStacked::THD;
  fresh.varlen_bits.k_length = VarlenLength::INDIVIDUAL;
  fresh.varlen_bits.k_position = VarlenPosition::ARRAY;
  check_eq(varlen_to_wire(fresh.varlen_bits), 0x150Bu,
           "current caller: seqused_k on packed KV, which no VarlenType can spell");
  fresh.varlen_bits.lse_layout = VarlenLseLayout::TH;
  check_eq(varlen_to_wire(fresh.varlen_bits), 0x1'150Bu, "current caller: + _TH");
}

// PR #222 review: the shim keyed Max_seqlen off tensor presence, so a THD side
// with LENGTH == MAX -- which carries no length array -- kept the tensor's own
// extent. Under THD that extent is the total packed token count.
void
test_max_seqlen_source() {
  // Dense: no bits set, so the tensor extent is the per-sequence length and the
  // caller's Max_seqlen must NOT override it.
  check(!varlen_uses_caller_max_seqlen(0x0000u, false), "dense Q trusts the tensor");
  check(!varlen_uses_caller_max_seqlen(0x0000u, true), "dense K trusts the tensor");

  // The reported case: compact Q against a uniformly-stacked K (side 0x01 =
  // THD, MAX, IMPLIED). Q is already refused elsewhere for STACKED+MAX, so it
  // is the K side that slipped through.
  check(varlen_uses_caller_max_seqlen(0x010Bu, true),
        "THD+MAX K takes Max_seqlen_k from the caller");
  check(varlen_uses_caller_max_seqlen(0x010Bu, false) == true,
        "compact Q takes Max_seqlen_q from the caller");

  // Every shipped mode already did the right thing, via presence; they must
  // keep doing it now that the predicate is the bits.
  for (uint32_t wire : {0x0B0Bu, 0x0202u, 0x1313u, 0x150Bu, 0x040Bu}) {
    check(varlen_uses_caller_max_seqlen(wire, false), "non-dense Q uses caller max");
    check(varlen_uses_caller_max_seqlen(wire, true), "non-dense K uses caller max");
  }
}

// PR #222 review: the shim validated only the Q-derived sequence count, so
// combinations that make the kernel tl.load from null reached the launch.
void
test_bits_validation() {
  // The shipped configurations, with exactly the arrays each mode needs.
  check(varlen_valid(0x0000u, false, false, false, false), "dense is valid");
  check(varlen_valid(0x0B0Bu, true, false, true, false), "compact is valid");
  check(varlen_valid(0x0202u, true, false, true, false), "padded is valid");
  check(varlen_valid(0x1313u, true, true, true, true), "strided is valid");
  check(varlen_valid(0x150Bu, true, false, true, true), "seqused on packed is valid");
  check(varlen_valid(0x040Bu, true, false, true, false), "seqused on BHSD is valid");

  // Missing arrays -- each of these would have been a load from null.
  check(!varlen_valid(0x0B0Bu, true, false, false, false), "compact without seqinfo_k0");
  check(!varlen_valid(0x1313u, true, false, true, true), "ARRAY Q without seqinfo_q1");
  check(!varlen_valid(0x150Bu, true, false, true, false), "seqused without seqinfo_k1");

  // A stray array the mode never reads: dense + seqinfo_q0 used to pass and
  // replace the dense max length with an unset Max_seqlen_q.
  check(!varlen_valid(0x0000u, true, false, false, false), "dense with a stray seqinfo_q0");

  // REUSE takes a POSITION out of the length array, which only holds positions
  // under CUMULATIVE. 0x09 = THD, MAX, REUSE; 0x0D = THD, INDIVIDUAL, REUSE.
  check(!varlen_valid(0x0009u, true, false, false, false), "REUSE with MAX length");
  check(!varlen_valid(0x000Du, true, false, false, false), "REUSE with INDIVIDUAL length");

  // Out-of-range fields and reserved bits.
  check(!varlen_valid(0x0006u, true, false, false, false), "LENGTH == 3 is not a value");
  check(!varlen_valid(0x0018u, false, true, false, false), "POSITION == 3 is not a value");
  check(!varlen_valid(0x0020u, false, false, false, false), "reserved bit 5 set");
  check(!varlen_valid(0x00040000u, false, false, false, false), "reserved bit 18 set");

  // LSE_LAYOUT is not addressing and must not be rejected.
  check(varlen_valid(0x10000u, false, false, false, false), "dense + TH is valid");
  check(varlen_valid(0x1150Bu, true, false, true, true), "seqused + TH is valid");

  // ... and only those two: the kernel maps every nonzero value to TH.
  check(!varlen_valid(0x20000u, false, false, false, false), "lse_layout == 2 is refused");
  check(!varlen_valid(0x30000u, false, false, false, false), "lse_layout == 3 is refused");
  check(!varlen_valid(0x2150Bu, true, false, true, true), "lse_layout == 2 on a valid mode");
}

// PR #222 review: presence was checked, extent was not, so an undersized array
// or a BHSD tensor with fewer batch slots than N was loaded past its end.
void
test_extent_validation() {
  const int32_t N = 7;
  // compact 0x0B0B: both sides CUMULATIVE, so both need N+1.
  check(varlen_extents_valid(0x0B0Bu, N, N + 1, 0, N + 1, 0, 1, 1),
        "compact with exactly N+1");
  check(!varlen_extents_valid(0x0B0Bu, N, N, 0, N + 1, 0, 1, 1),
        "compact with a short seqinfo_q0");
  check(!varlen_extents_valid(0x0B0Bu, N, N + 1, 0, N, 0, 1, 1),
        "compact with a short seqinfo_k0");

  // LOWER bound, not equality: a bigger buffer than the mode needs is fine.
  check(varlen_extents_valid(0x0B0Bu, N, N + 64, 0, N + 64, 0, 1, 1),
        "oversized arrays are accepted");

  // strided 0x1313: ARRAY on both. Stacked Q also reads [N], so it needs N+1
  // where K needs only N.
  check(varlen_extents_valid(0x1313u, N, N + 1, N + 1, N + 1, N, 1, 1),
        "strided at its minimum extents");
  check(!varlen_extents_valid(0x1313u, N, N + 1, N, N + 1, N, 1, 1),
        "strided Q position array missing its [N] slot");

  // seqused 0x150B: K length INDIVIDUAL needs N, K position ARRAY needs N.
  check(varlen_extents_valid(0x150Bu, N, N + 1, 0, N, N, 1, 1),
        "seqused at its minimum extents");
  check(!varlen_extents_valid(0x150Bu, N, N + 1, 0, N - 1, N, 1, 1),
        "seqused with a short seqused_k");

  // The reported case: mixed 0x000B derives N from a packed Q, so a dense K
  // with fewer batch slots than N is indexed with a z it never had.
  check(varlen_extents_valid(0x000Bu, N, N + 1, 0, 0, 0, 1, N),
        "mixed with a K batch of N");
  check(!varlen_extents_valid(0x000Bu, N, N + 1, 0, 0, 0, 1, N - 1),
        "mixed with a K batch shorter than N");
  check(varlen_extents_valid(0x000Bu, N, N + 1, 0, 0, 0, 1, N + 3),
        "mixed with a K batch larger than N");

  // padded 0x0202: BHSD both sides, so both tensors need N slots.
  check(varlen_extents_valid(0x0202u, N, N + 1, 0, N + 1, 0, N, N),
        "padded with N batch slots");
  check(!varlen_extents_valid(0x0202u, N, N + 1, 0, N + 1, 0, N - 1, N),
        "padded with a short Q batch");

  // dense reads no array at all and is indexed by batch on both sides.
  check(varlen_extents_valid(0x0000u, N, 0, 0, 0, 0, N, N), "dense with N batch slots");
  check(!varlen_extents_valid(0x0000u, N, 0, 0, 0, 0, N, N - 1), "dense with a short K batch");
}

void
test_seq_count() {
  // N off the Q side. cu_seqlens_q has N+1 entries under CUMULATIVE.
  check_seq_count("dense", 0x0000u, 0, 4, 4);
  check_seq_count("compact", 0x0B0Bu, 8, 1, 7);
  check_seq_count("strided", 0x1313u, 10, 1, 9);
  // Padded is BHSD on the Q side, so N rides in on Q's batch axis and the
  // array is not consulted -- the non-trivial row, and the one a "just read
  // the array" implementation gets wrong when the two disagree.
  check_seq_count("padded", 0x0202u, 6, 5, 5);
  // Q-side INDIVIDUAL: the array is (N,), not (N+1,).
  check_seq_count("q individual", 0x0004u | 0x0001u, 7, 1, 7);
  // STACKED + MAX determines nothing on the host.
  check(varlen_seq_count(0x0001u, 0, 1) < 0, "STACKED + MAX reports undetermined");

  // The independent derivation agrees on every shipped row, and is what
  // catches a padded caller whose cu_seqlens_q has the wrong length.
  check(varlen_seq_count_independent(0x0202u, 6, 5) == 5, "independent: padded agrees");
  check(varlen_seq_count_independent(0x0202u, 4, 5) == 3, "independent: padded disagrees");
  check(varlen_seq_count_independent(0x0B0Bu, 8, 1) == 7, "independent: compact agrees");
  check(varlen_seq_count_independent(0x0000u, 0, 4) == 4, "independent: dense agrees");
  check(varlen_seq_count_independent(0x0405u, 7, 1) < 0,
        "independent: declines Q-side INDIVIDUAL");
}

// A stand-in for the generated OpAttnBwdParams: the three fields
// varlen_bwd_seq_count() reads, and nothing else. Compiling against this is the
// point -- the real struct lives in a generated header, so without a stub the
// template body is only ever checked by a full AOT build.
struct FakeBwdParams {
  int32_t varlen_bits;
  const T1* seqinfo_q0;
  const T4* Q;
};

void
test_bwd_grid_extent() {
  // Padded: Q's batch axis IS N, and the length array is present but not
  // consulted -- the row where reading the array instead would look plausible
  // and be wrong whenever the two disagree.
  const T1 cu_seqlens{1, {6}, {1}, AOTRITON_NS::kInt32};
  const T4 q_padded{1, {5, 3, 64, 64}, {0, 0, 0, 1}, AOTRITON_NS::kFloat16};
  FakeBwdParams padded{0x0202, &cu_seqlens, &q_padded};
  check(varlen_bwd_seq_count(&padded) == 5u, "bwd grid z extent, padded");

  // Compact: Q's batch axis is 1 under THD, so the array is the only source.
  const T4 q_packed{1, {1, 3, 512, 64}, {0, 0, 0, 1}, AOTRITON_NS::kFloat16};
  FakeBwdParams compact{0x0B0B, &cu_seqlens, &q_packed};
  check(varlen_bwd_seq_count(&compact) == 5u, "bwd grid z extent, compact");

  // Dense: no array at all, and the null tensor's sizes are INDETERMINATE, so
  // this also checks that the BHSD path never reaches size(0).
  const T1 no_array;
  const T4 q_dense{1, {4, 3, 128, 64}, {0, 0, 0, 1}, AOTRITON_NS::kFloat16};
  FakeBwdParams dense{0x0000, &no_array, &q_dense};
  check(varlen_bwd_seq_count(&dense) == 4u, "bwd grid z extent, dense");
}

void
test_persistent_mask() {
  // The addressing mask is what makes a dense call asking for _TH keep the
  // persistent path. attn_fwd.cc and fwd_kernel.py compute this predicate
  // SEPARATELY and must agree; disagreeing is a grid/indexing mismatch, so
  // wrong output rather than a slowdown.
  VarlenBits dense_th = kDense;
  dense_th.lse_layout = VarlenLseLayout::TH;
  const uint32_t wire = varlen_to_wire(dense_th);
  check(wire != 0u, "dense + TH is a non-zero word");
  check(varlen_addressing(wire) == 0u, "dense + TH has zero addressing bytes");
  check(varlen_addressing(varlen_to_wire(kCompact)) != 0u,
        "compact has non-zero addressing bytes");
}

} // anonymous namespace

int
main() {
  test_legacy_table();
  test_legacy_quirks();
  test_version_translation();
  test_max_seqlen_source();
  test_bits_validation();
  test_extent_validation();
  test_seq_count();
  test_bwd_grid_extent();
  test_persistent_mask();

  if (g_failures == 0) {
    std::fprintf(stderr, "ALL PASSED\n");
    return 0;
  }
  std::fprintf(stderr, "%d CHECK(S) FAILED\n", g_failures);
  return 1;
}
