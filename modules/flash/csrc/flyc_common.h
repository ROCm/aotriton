// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AOTRITON_MODULES_FLASH_CSRC_FLYC_COMMON_H
#define AOTRITON_MODULES_FLASH_CSRC_FLYC_COMMON_H

// The host-side translations every flyc flash kernel needs: AOTriton's varlen
// layout word into the two extra numbers a FlyDSL launch wants, and AOTriton's
// float dropout probability into FlyDSL's threshold/scale pair.
//
// These were `FlycAttnFwdContext` member functions until the backward kernels
// arrived. They are free functions taking plain operands rather than a params
// struct, because the two params structs are different types that spell the
// same field differently -- `OpAttnFwdParams::Varlen_bits` against
// `OpAttnBwdParams::varlen_bits` -- so neither a shared base nor a template
// over the struct would bind.

#include <aotriton/config.h>
#include <aotriton/flash.h>
#include <aotriton/_internal/log.h>
#include <aotriton/_internal/pon.h>

#include "varlen.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace AOTRITON_NS::v3::flash {

// The two numbers a flyc kernel launch needs that AOTriton has no field for.
//
// The layout word itself is not among them: `varlen_bits` is an operand on both
// params structs and reaches the kernarg vector as a plain rename, so nothing
// here has to carry it.
//
// `batch_size` is not a kernarg either -- the FlyDSL signature dropped it --
// but it is still needed, because the grid's z extent is
// `num_seqlens != 0 ? num_seqlens : batch_size` and AOTriton computes its own
// grid. So it survives here as a grid input rather than an argument.
struct FlycVarlenRow {
  int32_t batch_size;
  int32_t num_seqlens;

  // The kernel's own z-extent expression, spelled once.
  int32_t nseq_idx() const { return num_seqlens != 0 ? num_seqlens : batch_size; }
};

// Fills that row from the layout word AOTriton now hands over directly.
//
// There is no longer a varlen encoding to translate. `varlen_bits` IS the word
// the FlyDSL kernel decodes -- FlyDSL and Triton share one varlen ABI, so
// `varlen.h` serves both backends -- and it passes straight through. This
// function exists only for the two DERIVED numbers, which AOTriton has no
// field for because its own kernels take neither.
//
// This used to infer the layout instead, from the sign of a tri-state
// `Num_seqlens` plus the nullness of `seq_strides_q`. That inference, and the
// nseq_idx cross-check that made it safe to trust, both moved upstream: the
// operator shims (attn_fwd.cc / attn_bwd.cc) run varlen_valid(), max_seqlen_ok(),
// extents_ok() and the seq_count()/independent_seq_count() comparison before
// any backend is chosen. Repeating any of it here would be a second spelling of
// a check that has already passed.
//
// Q side only: the launch's N comes off Q, and both the grid's z extent and
// FlyDSL's num_seqlens are properties of the output's addressing.
inline FlycVarlenRow
flyc_classify_varlen(int32_t varlen_bits, const T4& q,
                     const T1& seqinfo_q0, const T1& seqinfo_q1,
                     int32_t max_seqlen_q) {
  const auto v = internal::varlen_from_wire(static_cast<uint32_t>(varlen_bits));
  const auto q_addr = internal::varlen_addressing_of(v.qmode, q, seqinfo_q0,
                                                     seqinfo_q1, max_seqlen_q);
  const int32_t nseq = q_addr.seq_count();
  if (nseq < 0) {
    // A stacked side with LENGTH == MAX whose token axis is not a whole
    // multiple of max_seqlen. There is no array to count and rounding would
    // give in-bounds addresses of the wrong rows, so refuse rather than guess
    // -- the same contract seq_count() documents for its -1.
    AOTRITON_LOG(LOG_ERROR,
                 "flyc varlen: varlen_bits=0x%08x gives no derivable sequence count "
                 "(Q token axis %d is not a whole multiple of max_seqlen_q=%d) "
                 "-- refusing to launch",
                 static_cast<unsigned>(varlen_bits),
                 static_cast<int>(q.size(2)), max_seqlen_q);
    throw std::runtime_error("flyc varlen: underivable sequence count");
  }

  // FlyDSL's num_seqlens is narrower than N: it means "how many sequences are
  // packed into a 1THD tensor", so a BHSD side reports 0 however many sequences
  // it holds, and the count reaches the kernel through Q's batch axis instead.
  const bool stacked = v.qmode.stacked != VarlenStacked::BHSD;
  return FlycVarlenRow {
    static_cast<int32_t>(q.size(0)),
    stacked ? nseq : 0,
  };
}

// The i32 dropout threshold. FlyDSL's `philox.dropout_threshold` (in
// modules/flash/flyc/philox.py, reached from fmha_abi_gfx1201.py's
// dropout_args): a uniform u32 reinterpreted as i32 is uniform on
// [-2**31, 2**31), so comparing against `(p - 0.5) * 0xFFFFFFFF` keeps a
// `1 - p` fraction with one signed compare and no per-element float
// conversion. This is bit-for-bit the same formula AOTriton's own Triton
// kernels already use (modules/flash/kernel/fwd_kernel.py and siblings:
// `((dropout_p - 0.5) * 0xFFFFFFFF).to(tl.int32)`), computed here in double
// precision and clamped rather than truncated by a narrowing cast, matching
// `dropout_threshold`'s own `max(-(2**31), min(2**31 - 1, t))`.
//
// When dropout is disabled, FlyDSL's `dropout_args` returns threshold 0 and
// scale 1.0 without evaluating the formula at all; mirrored here rather than
// evaluating it at p == 0, which is not the same value.
inline int32_t
flyc_idropout_p_of(bool enable_dropout, float dropout_p) {
  if (!enable_dropout) {
    return 0;
  }
  const double p = static_cast<double>(dropout_p);
  const double raw = (p - 0.5) * static_cast<double>(0xFFFFFFFFu);
  const int64_t t = static_cast<int64_t>(raw);
  constexpr int64_t kInt32Min = -(int64_t{1} << 31);
  constexpr int64_t kInt32Max = (int64_t{1} << 31) - 1;
  return static_cast<int32_t>(std::clamp(t, kInt32Min, kInt32Max));
}

inline float
flyc_dropout_scale_of(bool enable_dropout, float dropout_p) {
  if (!enable_dropout) {
    return 1.0f;
  }
  return 1.0f / (1.0f - dropout_p);
}

// Find the smallest COMPILED rung >= `hdim` in the arch-indexed table
// codegen_compiled_rung_table_defs (python/codegen/flyc.py) generates on each
// FlycXXXContext (`compiled_block_dmodel` /
// `compiled_block_dmodel_count`, both static). Mirrors
// <aotriton/_internal/util.h>'s round_value -- used at the OPERATOR's own
// BLOCK_DMODEL binning site, attn_fwd.cc / attn_bwd.cc, BEFORE a backend is
// chosen -- but over a raw (pointer, count) pair rather than a
// std::vector<int32_t>, since that is what the generated static arrays are.
//
// Returns -1, the same "not found" sentinel round_value uses, when `table`
// is empty (this arch compiles nothing for this kernel -- e.g. gfx950 for a
// gfx1201-only flyc kernel, see codegen/flyc.py's comment on why that row is
// legitimately empty) or when `hdim` exceeds every compiled rung. -1 is never
// a valid BLOCK_DMODEL choice, so it reaches godel_number()'s digit-compare
// unchanged, is rejected there with a logged "Unsupported" message, and
// lookup_optimal() returns hipErrorNotSupported -- the same graceful
// rejection shape as any other out-of-range functional value, not a crash
// and not a silently wrong kernel.
inline int16_t
flyc_round_up_rung(int32_t hdim, const int16_t* table, int count) {
  for (int i = 0; i < count; ++i) {
    if (table[i] >= hdim) {
      return table[i];
    }
  }
  return -1;
}

// The two axis orders a flyc grid can walk, mirrored (same names, same integer
// values) from FlyDSL's fmha_tuning_gfx950.GRID_AXIS_HEAD_FASTEST /
// GRID_AXIS_TILE_FASTEST. Every flyc description puts one of
// these two values into its knobs under 'GRID_AXIS_ORDER' -- gfx1201's by a
// literal assignment (there is no upstream knob to carry it), gfx950's as a
// flat field `resolve()` already produces -- so a grid_calculator() reads one
// perf() key and switches, instead of assuming a fixed order that only held
// for gfx1201 before gfx950 arrived with a different one for dK/dV.
enum FlycGridAxisOrder : int64_t {
  kFlycGridAxisHeadFastest = 0,
  kFlycGridAxisTileFastest = 1,
};

// Reads GRID_AXIS_ORDER out of `perf()` once, so the three grid_calculator()s
// log and throw identically on a missing or unrecognised value rather than
// diverging in wording. `kernel_name` is only for the message.
inline FlycGridAxisOrder
flyc_grid_axis_order(const Pon& perf, const char* kernel_name) {
  const auto opt = perf.get_int("GRID_AXIS_ORDER");
  if (!opt) {
    AOTRITON_LOG(LOG_ERROR,
                 "%s grid_calculator: perf() is missing the 'GRID_AXIS_ORDER' key",
                 kernel_name);
    throw std::runtime_error("flyc grid_calculator: missing 'GRID_AXIS_ORDER' in perf()");
  }
  const int64_t v = *opt;
  if (v != kFlycGridAxisHeadFastest && v != kFlycGridAxisTileFastest) {
    AOTRITON_LOG(LOG_ERROR,
                 "%s grid_calculator: unrecognised GRID_AXIS_ORDER=%lld",
                 kernel_name, static_cast<long long>(v));
    throw std::runtime_error("flyc grid_calculator: unrecognised GRID_AXIS_ORDER");
  }
  return static_cast<FlycGridAxisOrder>(v);
}

}  // namespace AOTRITON_NS::v3::flash

#endif
