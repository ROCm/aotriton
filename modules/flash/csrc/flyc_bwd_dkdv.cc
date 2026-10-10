// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/config.h>
#include <aotriton/_internal/util.h>
#include <aotriton/_internal/log.h>
#include <aotriton/_internal/pon.h>
#include <aotriton/flash.h>
#include <aotriton/util.h>
#include <flash/flyc.flyc_bwd_dkdv.h>
#include <flash/iface.op_attn_bwd.h>

#include "flyc_common.h"

#include <cstdint>
#include <stdexcept>

namespace AOTRITON_NS::v3::flash {

namespace {

// Note the lowercase field names: op_attn_bwd spells `varlen_bits` where
// op_attn_fwd spells `Varlen_bits`, which is why flyc_classify_varlen takes
// operands rather than a params struct.
FlycVarlenRow
classify(const OpAttnBwdParams& params) {
  return flyc_classify_varlen(params.varlen_bits,
                              *params.Q,
                              *params.seqinfo_q0,
                              *params.seqinfo_q1,
                              params.max_seqlen_q);
}

} // anonymous namespace

int32_t
FlycBwdDkdvContext::flyc_num_seqlens() const {
  return classify(*params).num_seqlens;
}

// The backward's dropout must reproduce the forward's mask exactly, so these
// two are the same translation the forward uses, from the same header -- not a
// backward-specific variant.
int32_t
FlycBwdDkdvContext::flyc_idropout_p() const {
  return flyc_idropout_p_of(params->ENABLE_DROPOUT, params->dropout_p);
}

float
FlycBwdDkdvContext::flyc_dropout_scale() const {
  return flyc_dropout_scale_of(params->ENABLE_DROPOUT, params->dropout_p);
}

// See flyc_attn_fwd.cc for the full rationale (same
// mechanism: round the TRUE head dims against this kernel's own compiled
// ladder, not against params->BLOCK_DMODEL, which already holds the
// OPERATOR's rounding from attn_bwd.cc's binning site).
int16_t
FlycBwdDkdvContext::flyc_block_dmodel() const {
  const auto [arch_number, mod_number] = get_archmod_number(current_gpu);
  (void)mod_number;
  const int32_t hdim = std::max(params->hdim_qk, params->hdim_vo);
  if (arch_number < 0) {
    return static_cast<int16_t>(hdim);
  }
  return flyc_round_up_rung(hdim, compiled_block_dmodel[arch_number], compiled_block_dmodel_count[arch_number]);
}

// Must follow flyc_block_dmodel()'s own rounding decision -- see
// flyc_attn_fwd.cc for why this recomputes rather than reading
// scratch_params.flyc_block_dmodel.
bool
FlycBwdDkdvContext::flyc_padded_head() const {
  const int32_t hdim_qk = params->hdim_qk;
  const int32_t hdim_vo = params->hdim_vo;
  const int16_t rounded = flyc_block_dmodel();
  return rounded != hdim_qk || rounded != hdim_vo;
}

// Grid shape. Unlike the forward and dQ, the two arches genuinely DISAGREE on
// the axis order here -- this is not a formality, both are real, measured
// launchers:
//
// gfx1201 (`launch_bwd_dkdv` in fmha_bwd_dkdv_gfx1201_kernel.py), TILE_FASTEST:
//
//   nseq_idx     = num_seqlens != 0 ? num_seqlens : batch_size
//   num_kv_tiles = cdiv(max_seqlen_k, BLOCK_N)
//   launcher.launch(grid=(num_kv_tiles, num_head_k, nseq_idx), block=(BLOCK_SIZE, 1, 1))
//
// gfx950 (`fmha_bwd_dkdv_gfx950_kernel`'s own `.launch(...)`), HEAD_FASTEST --
// per that call's own comment, this was TRIED the other way and measured
// 12-15% SLOWER at every rung (the eight XCDs have separate L2s, so sharing a
// KV slab across concurrent workgroups duplicates it instead of spreading
// distinct work):
//
//   num_kv_blocks = cdiv(max_seqlen_k, BLOCK_KV)
//   launcher.launch(grid=(num_head_k, num_kv_blocks, bs_idx), block=(BLOCK_SIZE, 1, 1))
//
// So both orders below are real, not a forward-compatibility stub -- see
// flyc_common.h's flyc_grid_axis_order for why this reads perf() rather than
// assuming either.
//
// The KV tile size is also NOT resolved the same way on both arches -- on
// gfx1201 it is not a resolve_knobs output at all, but BLOCK_N derived by
// this file's own description as `ROWS_PER_WAVE * NUM_TEAMS` (see the
// knob note in modules/flash/aot/flyc_bwd_dkdv.py); on gfx950 it IS a flat
// resolved field, `BwdDkDvKnobs.block_kv`. Both land under the same knob
// key, 'block_kv', normalised in Python (the derivation differs; the wire
// concept and its name do not), so this reads exactly one key.
dim3
FlycBwdDkdvContext::grid_calculator() const {
  const auto row = classify(*params);
  const auto axis_order = flyc_grid_axis_order(perf(), "flyc dkdv");

  const auto block_kv_opt = perf().get_int("block_kv");
  if (!block_kv_opt) {
    AOTRITON_LOG(LOG_ERROR,
                 "flyc dkdv grid_calculator: perf() is missing the 'block_kv' key");
    throw std::runtime_error("flyc dkdv grid_calculator: missing 'block_kv' in perf()");
  }
  const auto block_kv = static_cast<uint32_t>(*block_kv_opt);
  const auto num_kv_tiles =
      AOTRITON_NS::cdiv<uint32_t>(static_cast<uint32_t>(params->max_seqlen_k), block_kv);
  const auto num_head_k = static_cast<uint32_t>(params->num_head_k);
  const auto nseq_idx = static_cast<uint32_t>(row.nseq_idx());

  switch (axis_order) {
    case kFlycGridAxisTileFastest:
      return dim3 { num_kv_tiles, num_head_k, nseq_idx };
    case kFlycGridAxisHeadFastest:
      return dim3 { num_head_k, num_kv_tiles, nseq_idx };
  }
  throw std::runtime_error("flyc dkdv grid_calculator: unreachable GRID_AXIS_ORDER");
}

} // namespace AOTRITON_NS::v3::flash
