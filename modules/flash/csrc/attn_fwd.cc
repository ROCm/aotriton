// Copyright © 2023-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/config.h>
#include <aotriton/_internal/util.h>
#include <aotriton/_internal/log.h>
#include <aotriton/flash.h>
#include <aotriton/util.h>
#include <flash/shim.attn_fwd.h>
#include <flash/iface.op_attn_fwd.h>

#include "varlen.h"
#include "params_abi_compat.h"

namespace AOTRITON_NS::v3::flash {

// varlen.h keeps its symbols out of the user-facing namespace; this TU opts in.
using namespace internal;

dim3 AttnFwdContext::grid_calculator() const {
    AOTRITON_LOG(LOG_DEBUG,
                 "Selected Kernel BLOCK_M = %d BLOCK_N = %d PRE_LOAD_V = %d",
                 int(this->BLOCK_M), int(this->BLOCK_N), int(this->PRE_LOAD_V));
    // Mask to the ADDRESSING bytes, not the whole word: LSE_LAYOUT lives in
    // bits 17:16, so a dense call asking for _TH is non-zero and must NOT take
    // the varlen fallback. The Triton kernel computes this predicate
    // independently (fwd_kernel.py's `unsupported_by_persistent`) and the two
    // must agree EXACTLY -- if they disagree the host launches a
    // persistent-shaped grid while the kernel walks tiles non-persistently, or
    // the reverse, which is a grid/indexing mismatch rather than a slowdown.
    // Keep this spelling character-identical to the kernel's.
    bool unsupported_by_persistent = (params->Varlen_bits & 0xFFFF) != 0;
    auto nblocks = AOTRITON_NS::cdiv<uint32_t>(params->Max_seqlen_q, this->BLOCK_M);
    // Use default grid if not persistent, or input is unsupported_by_persistent,
    // in which case persistent is turned off IN TRITON KERNEL
    // and this kernel will expect regular grid configs.
    //
    // Note: This fallback behavior is determined by GPU kernel at runtime.
    if (this->PERSISTENT_TYPE == 0 || unsupported_by_persistent) {
      auto S = nblocks;
      auto H = uint32_t(params->Q->size(1));
      auto B = uint32_t(params->Batch);
      return NUM_XCDS > 1 ? dim3 { H, S, B } : dim3 { S, H, B };
    }
    // PERSISTENT or PERSISTENT_DYNAMIC
    // grid = lambda META: (min(NUM_CU * META['GRID_CU_MULTIP'],
    //                      triton.cdiv(metadata.max_seqlens_q, META['BLOCK_M']) * nheads_q * batch), )
    uint32_t from_cu = params->Num_CU * this->GRID_CU_MULTIP;
    uint32_t from_in = nblocks * params->Num_head_q * params->Batch;
    dim3 grid {
      uint32_t(std::min(from_cu, from_in)),
      1,
      1,
    };
    return grid;
}

attn_fwd_params::attn_fwd_params()
{
}

hipError_t AOTRITON_API
attn_fwd(const attn_fwd_params& in,
         int32_t params_version,
         AOTRITON_NS::Stream stream_wrap,
         const attn_options* options) {
  // Newer than we know how to read: the caller was built against a header from
  // the future, and no amount of translation invents fields it added.
  if (params_version > attn_fwd_params::kVersion) {
    return hipErrorInvalidSymbol;
  }
  // Older: translate the caller's object through the type that describes ITS
  // layout, then re-enter at the current version so nothing downstream needs a
  // branch. Compatibility is this translation, not any property of the current
  // struct's shape -- which is what leaves fields free to come and go across
  // kVersions. translate_*_params() dispatches on the version actually passed
  // and reports false for anything older than it describes, so the set of
  // described layouts is exactly the supported set.
  if (params_version < attn_fwd_params::kVersion) {
    attn_fwd_params upgraded;
    if (!translate_fwd_params(in, params_version, &upgraded)) {
      return hipErrorInvalidSymbol; // too old to translate
    }
    return attn_fwd(upgraded, attn_fwd_params::kVersion, stream_wrap, options);
  }
  // Reasoned about as a VarlenBits throughout; the wire word is built once,
  // at the kernel boundary below, and otherwise only inside grid calculators.
  const VarlenBits varlen = in.varlen_bits;
  // One object per side; nothing below asks "which side" again.
  const VarlenAddressing q_addr = varlen_addressing_of(
      varlen.qmode, true, in.Q, in.seqinfo_q0, in.seqinfo_q1, in.Max_seqlen_q);
  const VarlenAddressing k_addr = varlen_addressing_of(
      varlen.kmode, false, in.K, in.seqinfo_k0, in.seqinfo_k1, in.Max_seqlen_k);
  hipError_t err;
  auto stream = stream_wrap.native();
  auto gpu = getGpuFromStream(stream);
  int batch = in.Q.size(0);
  int hdim_qk = in.Q.size(3);
  int hdim_vo = in.V.size(3);
  int hdim_max = std::max(hdim_qk, hdim_vo);
  int num_head_q = in.Q.size(1);
  int num_head_k = in.K.size(1);
  // From the side objects: the caller's value where the mode reads one, the
  // tensor's extent only where a fully dense side makes that the same thing.
  int max_seqlen_q = q_addr.max_seqlen();
  int max_seqlen_k = k_addr.max_seqlen();
  // Well-formedness of the bits themselves, before anything is derived from
  // them: an out-of-range field, REUSE without CUMULATIVE, or a mode whose
  // array is absent would otherwise reach the kernel and tl.load from null.
  if (!varlen_valid(varlen, q_addr, k_addr)) {
    AOTRITON_LOG(LOG_ERROR,
                 "v3::flash::attn_fwd: varlen_bits=0x%08x is not well-formed for the "
                 "seqinfo arrays supplied (q0=%d q1=%d k0=%d k1=%d) -- refusing to launch",
                 varlen_to_wire(varlen), int(bool(in.seqinfo_q0)), int(bool(in.seqinfo_q1)),
                 int(bool(in.seqinfo_k0)), int(bool(in.seqinfo_k1)));
    return hipErrorInvalidValue;
  }
  // N, the sequence count, is what the grid's z extent and the kernel's `[N]`
  // read are both sized by. Under the bits it comes off the Q side, and it is
  // `Batch` for the dense case by construction.
  const int32_t seqinfo_q0_len = varlen_seqinfo_len(in.seqinfo_q0);
  int num_seqlens = q_addr.seq_count();
  const int32_t nseq_independent = q_addr.independent_seq_count();
  if (num_seqlens <= 0 || (nseq_independent >= 0 && num_seqlens != nseq_independent)) {
    AOTRITON_LOG(LOG_ERROR,
                 "v3::flash::attn_fwd: varlen_bits=0x%08x gives %d sequences "
                 "(seqinfo_q0.size(0)=%d, Q.size(0)=%d) but AOTriton independently "
                 "computes %d -- refusing to launch",
                 varlen_to_wire(varlen), num_seqlens, seqinfo_q0_len, batch, nseq_independent);
    return hipErrorInvalidValue;
  }
  // ... and that every array and BHSD tensor actually holds that many. Lower
  // bounds only: a larger buffer than the mode needs is legitimate.
  if (!(q_addr.extents_ok(num_seqlens) && k_addr.extents_ok(num_seqlens))) {
    AOTRITON_LOG(LOG_ERROR,
                 "v3::flash::attn_fwd: varlen_bits=0x%08x needs %d sequences but the "
                 "seqinfo arrays/tensors are too short (q0=%d q1=%d k0=%d k1=%d, "
                 "Q.size(0)=%d K.size(0)=%d) -- refusing to launch",
                 varlen_to_wire(varlen), num_seqlens, seqinfo_q0_len,
                 varlen_seqinfo_len(in.seqinfo_q1), varlen_seqinfo_len(in.seqinfo_k0),
                 varlen_seqinfo_len(in.seqinfo_k1),
                 int32_t(in.Q.size(0)), int32_t(in.K.size(0)));
    return hipErrorInvalidValue;
  }
  const auto& compiled_head_dims = AttnFwdMetadata::get_BLOCK_DMODEL_choices();
  int16_t hdim_rounded = round_value(hdim_max, compiled_head_dims);
  // FIXME: Remove when compiler bug fixed
  if (Gpu2VendorArch(gpu) == CAT32(GpuVendor::kAMD, 0x950)) {
    if (hdim_rounded == 16)
      hdim_rounded = 32;
  }
  OpAttnFwdParams params = {
    .Q = &in.Q,
    .K = &in.K,
    .V = &in.V,
    .B = &in.B,
    .A = &in.A,
    .Sm_scale = in.Sm_scale,
    .L = &in.L,
    .Out = &in.Out,
    .Q_descale = false,
    .K_descale = false,
    .P_scale = false,
    .P_descale = false,
    .V_descale = false,
    .Num_head_q = num_head_q,
    .Num_head_k = num_head_k,
    // The sign trick on Num_seqlens (negative meant padded varlen) is gone: the
    // three axes it welded together are now independent fields of one word.
    .Varlen_bits = static_cast<int32_t>(varlen_to_wire(varlen)),
    // Named by ROLE: ?0 is the length source, ?1 the position source.
    .seqinfo_q0 = &in.seqinfo_q0,
    .seqinfo_k0 = &in.seqinfo_k0,
    .Max_seqlen_q = max_seqlen_q,
    .Max_seqlen_k = max_seqlen_k,
    .seqinfo_q1 = &in.seqinfo_q1,
    .seqinfo_k1 = &in.seqinfo_k1,
    .BLOCK_DMODEL = hdim_rounded,
    .Hdim_qk = static_cast<int32_t>(hdim_qk),
    .Hdim_vo = static_cast<int32_t>(hdim_vo),
    .PADDED_HEAD = (hdim_rounded != hdim_qk || hdim_rounded != hdim_vo),
    .ENABLE_DROPOUT = in.dropout_p > 0.0,
    .dropout_p = in.dropout_p,
    .philox_seed_ptr  = &in.philox_seed_ptr,
    .philox_offset1   = &in.philox_offset1,
    .philox_offset2 = in.philox_offset2,
    .philox_seed_output   = &in.philox_seed_output,
    .philox_offset_output = &in.philox_offset_output,
    .RETURN_ENCODED_SOFTMAX = false,
    .encoded_softmax = &in.encoded_softmax,
    .CAUSAL_TYPE = in.causal_type,
    .Window_left = in.window_left,
    .Window_right = in.window_right,
    .BIAS_TYPE = int8_t(in.B ? 1 : 0),
    .USE_ALIBI = false,
    .INT8 = false,
    .INT8_KV = false,
    .USE_P_SCALE = false,
    .persistent_atomic_counter = &in.persistent_atomic_counter,
    .Num_CU = in.causal_type != 0 ? getMultiProcessorCount(stream) : 80,
    // Batch IS N -- the kernel's `[N]` read and the grid's z extent are the
    // same number, and for the dense case N is the batch size.
    .Batch = num_seqlens,
  };
  OpAttnFwdContext context;
  context.params = &params;
  context.call_options = options;
  AOTRITON_LOG(LOG_DEBUG, "v3::flash::attn_fwd options = %p", static_cast<const void*>(options));
  if (options) {
    AOTRITON_LOG(LOG_DEBUG, "v3::flash::attn_fwd options->force_backend_index = %d",
                 int(options->force_backend_index));
  }
  if (options && options->force_backend_index >= 0) {
    context.backend_index = static_cast<OpAttnFwdContext::BackendEnum>(options->force_backend_index);
    context.disable_fallback = true;
  } else {
    err = context.lookup_optimal(gpu);
    if (err != hipSuccess) {
      return err;
    }
  }
  return context.launch(gpu, stream);
}

} // AOTRITON_NS::v3::flash
