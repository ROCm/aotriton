// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/config.h>
#include <aotriton/_internal/util.h>
#include <aotriton/_internal/log.h>
#include <aotriton/_internal/pon.h>
#include <aotriton/flash.h>
#include <aotriton/util.h>
#include <flash/flyc.flyc_bwd_dq.h>
#include <flash/iface.op_attn_bwd.h>

#include "flyc_common.h"

#include <cstdint>
#include <stdexcept>

namespace AOTRITON_NS::v3::flash {

namespace {

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
FlycBwdDqContext::flyc_num_seqlens() const {
  return classify(*params).num_seqlens;
}

int32_t
FlycBwdDqContext::flyc_idropout_p() const {
  return flyc_idropout_p_of(params->ENABLE_DROPOUT, params->dropout_p);
}

float
FlycBwdDqContext::flyc_dropout_scale() const {
  return flyc_dropout_scale_of(params->ENABLE_DROPOUT, params->dropout_p);
}

// See flyc_attn_fwd.cc for the full rationale (same
// mechanism: round the TRUE head dims against this kernel's own compiled
// ladder, not against params->BLOCK_DMODEL, which already holds the
// OPERATOR's rounding from attn_bwd.cc's binning site).
int16_t
FlycBwdDqContext::flyc_block_dmodel() const {
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
FlycBwdDqContext::flyc_padded_head() const {
  const int32_t hdim_qk = params->hdim_qk;
  const int32_t hdim_vo = params->hdim_vo;
  const int16_t rounded = flyc_block_dmodel();
  return rounded != hdim_qk || rounded != hdim_vo;
}

// Grid shape, derived from the vendored kernel's own launcher -- `launch_bwd_dq`
// in modules/flash/flyc/fmha_bwd_dq_gfx1201_kernel.py:
//
//   nseq_idx    = num_seqlens != 0 ? num_seqlens : batch_size
//   num_q_tiles = cdiv(max_seqlen_q, BLOCK_M)
//   launcher.launch(grid=(num_head_q, num_q_tiles, nseq_idx), block=(BLOCK_SIZE, 1, 1))
//
// This one DOES match the forward's shape -- (head, q_tile, seq) with the query
// head count -- because like the forward it walks Q tiles. Its sibling dK/dV
// does not; see the note there.
//
// `block_m` is a resolve_knobs output here, so unlike dK/dV's block_n it needs
// no derivation on the description side.
//
// As in the forward, the axis order is read from perf() rather than
// assumed -- both arches' descriptions put HEAD_FASTEST in the knobs for
// this kernel today, but the switch below is what makes that a fact this
// function checks rather than one it silently depends on.
dim3
FlycBwdDqContext::grid_calculator() const {
  const auto row = classify(*params);
  const auto axis_order = flyc_grid_axis_order(perf(), "flyc dq");

  const auto block_m_opt = perf().get_int("block_m");
  if (!block_m_opt) {
    AOTRITON_LOG(LOG_ERROR, "flyc dq grid_calculator: perf() is missing the 'block_m' key");
    throw std::runtime_error("flyc dq grid_calculator: missing 'block_m' in perf()");
  }
  const auto block_m = static_cast<uint32_t>(*block_m_opt);
  const auto num_q_tiles =
      AOTRITON_NS::cdiv<uint32_t>(static_cast<uint32_t>(params->max_seqlen_q), block_m);
  const auto num_head_q = static_cast<uint32_t>(params->num_head_q);
  const auto nseq_idx = static_cast<uint32_t>(row.nseq_idx());

  switch (axis_order) {
    case kFlycGridAxisHeadFastest:
      return dim3 { num_head_q, num_q_tiles, nseq_idx };
    case kFlycGridAxisTileFastest:
      AOTRITON_LOG(LOG_ERROR, "flyc dq grid_calculator: TILE_FASTEST is not implemented for this kernel");
      throw std::runtime_error("flyc dq grid_calculator: unsupported GRID_AXIS_ORDER (TILE_FASTEST)");
  }
  throw std::runtime_error("flyc dq grid_calculator: unreachable GRID_AXIS_ORDER");
}

} // namespace AOTRITON_NS::v3::flash
