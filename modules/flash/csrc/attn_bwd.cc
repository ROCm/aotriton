// Copyright © 2023-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/config.h>
#include <aotriton/_internal/util.h>
#include <aotriton/flash.h>
#include <aotriton/util.h>
#include <flash/shim.bwd_kernel_dk_dv.h>
#include <flash/shim.bwd_kernel_dq.h>
#include <flash/shim.bwd_preprocess.h>
#include <flash/iface.op_attn_bwd.h>
#include <aotriton/_internal/log.h>

#include "varlen.h"
#include "params_abi_compat.h"

namespace AOTRITON_NS::v3::flash {

// varlen.h keeps its symbols out of the user-facing namespace; this TU opts in.
using namespace internal;

dim3 BwdPreprocessContext::grid_calculator() const {
  // One kernel now: bwd_preprocess_varlen was merged in, and the two differed
  // only in this z extent and in an addressing shuffle the decoder subsumes.
  dim3 grid {
    AOTRITON_NS::cdiv<uint32_t>(params->max_seqlen_q, this->BLOCK_M),
    uint32_t(params->Out->size(1)),
    varlen_bwd_seq_count(params),
  };
  // std::cerr << "Grid conf " << grid.x << " " << grid.y << " " << grid.z << std::endl;
  return grid;
}

dim3 BwdKernelDkDvContext::grid_calculator() const {
  auto S = AOTRITON_NS::cdiv<uint32_t>(params->max_seqlen_k, this->BLOCK_N);
  auto H = uint32_t(params->K->size(1));
  auto B = varlen_bwd_seq_count(params);
  return NUM_XCDS > 1 ? dim3 { H, S, B } : dim3 { S, H, B };
}

dim3 BwdKernelDqContext::grid_calculator() const {
  auto S = AOTRITON_NS::cdiv<uint32_t>(params->max_seqlen_q, this->BLOCK_M);
  auto H = uint32_t(params->Q->size(1));
  auto B = varlen_bwd_seq_count(params);
  return NUM_XCDS > 1 ? dim3 { H, S, B } : dim3 { S, H, B };
}

attn_bwd_params::attn_bwd_params()
{
}

hipError_t AOTRITON_API
attn_bwd(const attn_bwd_params& in,
         int32_t params_version,
         AOTRITON_NS::Stream stream_wrap,
         const attn_options* options) {
  // Newer than we know how to read: the caller was built against a header from
  // the future, and no amount of translation invents fields it added.
  if (params_version > attn_bwd_params::kVersion) {
    return hipErrorInvalidSymbol;
  }
  // Older: translate the caller's object through the type that describes ITS
  // layout, then re-enter at the current version so nothing downstream needs a
  // branch. Compatibility is this translation, not any property of the current
  // struct's shape -- which is what leaves fields free to come and go across
  // kVersions. translate_*_params() dispatches on the version actually passed
  // and reports false for anything older than it describes, so the set of
  // described layouts is exactly the supported set.
  if (params_version < attn_bwd_params::kVersion) {
    attn_bwd_params upgraded;
    if (!translate_bwd_params(in, params_version, &upgraded)) {
      return hipErrorInvalidSymbol; // too old to translate
    }
    return attn_bwd(upgraded, attn_bwd_params::kVersion, stream_wrap, options);
  }
  const uint32_t varlen_wire = varlen_to_wire(in.varlen_bits);
  hipError_t err;
  auto stream = stream_wrap.native();
  auto gpu = getGpuFromStream(stream);
  int batch = in.Q.size(0);
  int hdim_qk = in.Q.size(3);
  int hdim_vo = in.V.size(3);
  int hdim_max = std::max(hdim_qk, hdim_vo);
  int num_head_q = in.Q.size(1);
  int num_head_k = in.K.size(1);
  int max_seqlen_q = in.Q.size(2);
  int max_seqlen_k = in.K.size(2);
  // Well-formedness of the bits themselves, before anything is derived from
  // them: an out-of-range field, REUSE without CUMULATIVE, or a mode whose
  // array is absent would otherwise reach the kernel and tl.load from null.
  if (!varlen_valid(varlen_wire,
                    bool(in.seqinfo_q0), bool(in.seqinfo_q1),
                    bool(in.seqinfo_k0), bool(in.seqinfo_k1))) {
    AOTRITON_LOG(LOG_ERROR,
                 "v3::flash::attn_bwd: varlen_bits=0x%08x is not well-formed for the "
                 "seqinfo arrays supplied (q0=%d q1=%d k0=%d k1=%d) -- refusing to launch",
                 varlen_wire, int(bool(in.seqinfo_q0)), int(bool(in.seqinfo_q1)),
                 int(bool(in.seqinfo_k0)), int(bool(in.seqinfo_k1)));
    return hipErrorInvalidValue;
  }
  // N. Unlike the forward pass this never becomes a kernel argument -- the
  // backward kernels read tl.num_programs(2) -- but the three grid calculators
  // above recompute it from the same two inputs, so it is validated here once.
  const int32_t seqinfo_q0_len = varlen_seqinfo_len(in.seqinfo_q0);
  const int32_t num_seqlens = varlen_seq_count(varlen_wire, seqinfo_q0_len, batch);
  const int32_t nseq_independent =
      varlen_seq_count_independent(varlen_wire, seqinfo_q0_len, batch);
  if (num_seqlens <= 0 || (nseq_independent >= 0 && num_seqlens != nseq_independent)) {
    AOTRITON_LOG(LOG_ERROR,
                 "v3::flash::attn_bwd: varlen_bits=0x%08x gives %d sequences "
                 "(seqinfo_q0.size(0)=%d, Q.size(0)=%d) but AOTriton independently "
                 "computes %d -- refusing to launch",
                 varlen_wire, num_seqlens, seqinfo_q0_len, batch, nseq_independent);
    return hipErrorInvalidValue;
  }
  // ... and that every array and BHSD tensor actually holds that many. Lower
  // bounds only: a larger buffer than the mode needs is legitimate.
  if (!varlen_extents_valid(varlen_wire, num_seqlens,
                            seqinfo_q0_len, varlen_seqinfo_len(in.seqinfo_q1),
                            varlen_seqinfo_len(in.seqinfo_k0),
                            varlen_seqinfo_len(in.seqinfo_k1),
                            int32_t(in.Q.size(0)), int32_t(in.K.size(0)))) {
    AOTRITON_LOG(LOG_ERROR,
                 "v3::flash::attn_bwd: varlen_bits=0x%08x needs %d sequences but the "
                 "seqinfo arrays/tensors are too short (q0=%d q1=%d k0=%d k1=%d, "
                 "Q.size(0)=%d K.size(0)=%d) -- refusing to launch",
                 varlen_wire, num_seqlens, seqinfo_q0_len,
                 varlen_seqinfo_len(in.seqinfo_q1), varlen_seqinfo_len(in.seqinfo_k0),
                 varlen_seqinfo_len(in.seqinfo_k1),
                 int32_t(in.Q.size(0)), int32_t(in.K.size(0)));
    return hipErrorInvalidValue;
  }
  // Keyed on the BITS, not on tensor presence: a THD side with LENGTH == MAX
  // supplies no seqinfo_?0, and trusting the tensor extent there yields the
  // total packed token count instead of the per-sequence maximum.
  if (varlen_uses_caller_max_seqlen(varlen_wire, false)) {
    max_seqlen_q = in.Max_seqlen_q;
  }
  if (varlen_uses_caller_max_seqlen(varlen_wire, true)) {
    max_seqlen_k = in.Max_seqlen_k;
  }
  const auto& compiled_head_dims = BwdKernelDkDvMetadata::get_BLOCK_DMODEL_choices();
  int16_t hdim_rounded = round_value(hdim_max, compiled_head_dims);
  // FIXME: Remove when compiler bug fixed
  if (Gpu2VendorArch(gpu) == CAT32(GpuVendor::kAMD, 0x950)) {
    if (hdim_rounded == 48)
      hdim_rounded = 64;
    if (hdim_rounded == 80)
      hdim_rounded = 96;
  }
  LazyTensorInternal<2> lazy_delta(in.D);
  LazyTensorInternal<4> lazy_dq_acc(in.DQ_ACC);
  OpAttnBwdParams params = {
    .Q = &in.Q,
    .K = &in.K,
    .V = &in.V,
    .B = &in.B,
    .sm_scale = in.Sm_scale,
    .Out = &in.Out,
    .DO = &in.DO,
    .DK = &in.DK,
    .DV = &in.DV,
    .DQ = &in.DQ,
    .DB = &in.DB,
    .DQ_ACC = &lazy_dq_acc,
    .L = &in.L,
    .D = &lazy_delta,
    .num_head_q = num_head_q,
    .num_head_k = num_head_k,
    // Named by ROLE: ?0 is the length source, ?1 the position source. The
    // tri-state num_seqlens is gone outright -- the sign trick it used for
    // padded varlen is now a field, and its magnitude is the grid's z extent.
    .seqinfo_q0 = &in.seqinfo_q0,
    .seqinfo_k0 = &in.seqinfo_k0,
    .varlen_bits = static_cast<int32_t>(varlen_wire),
    .max_seqlen_q = max_seqlen_q,
    .max_seqlen_k = max_seqlen_k,
    .seqinfo_q1 = &in.seqinfo_q1,
    .seqinfo_k1 = &in.seqinfo_k1,
    .hdim_qk = hdim_qk,
    .hdim_vo = hdim_vo,
    .dropout_p = in.dropout_p,
    .philox_seed_ptr  = &in.philox_seed_ptr,
    .philox_offset1   = &in.philox_offset1,
    .philox_offset2   = in.philox_offset2,
    .Window_left = in.window_left,
    .Window_right = in.window_right,
    .BLOCK_DMODEL = hdim_rounded,
    .CAUSAL_TYPE = in.causal_type,
    .ENABLE_DROPOUT = in.dropout_p > 0.0,
    .PADDED_HEAD = (hdim_qk != hdim_rounded || hdim_vo != hdim_rounded),
    .BIAS_TYPE = static_cast<int8_t>(bool(in.B) ? 1 : 0),
  };
  OpAttnBwdContext context;
  context.params = &params;
  context.call_options = options;
  AOTRITON_LOG(LOG_DEBUG, "v3::flash::attn_bwd options = %p", static_cast<const void*>(options));
  if (options) {
    AOTRITON_LOG(LOG_DEBUG, "v3::flash::attn_bwd options->force_backend_index = %d",
                 int(options->force_backend_index));
  }
  bool deterministic = false;
  if (params_version >= 4 && options) {
    deterministic = options->deterministic;
  }
  if (options && options->force_backend_index >= 0) {
    context.backend_index = static_cast<OpAttnBwdContext::BackendEnum>(options->force_backend_index);
    context.disable_fallback = true;
  } else if (deterministic) {
    context.backend_index = OpAttnBwdContext::BackendEnum::kMetro_TritonSplit;
  } else {
    err = context.lookup_optimal(gpu);
    if (err != hipSuccess) {
      return err;
    }
  }
  AOTRITON_LOG(LOG_DEBUG, "v3::flash::attn_bwd context.backend_index = %d", static_cast<int>(context.backend_index));
  err = context.launch(gpu, stream);
  in.D.free();
  in.DQ_ACC.free();
  return err;
}

}
