// Copyright © 2023-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/config.h>
#include <aotriton/_internal/util.h>
#include <aotriton/_internal/log.h>
#include <aotriton/flash.h>
#include <aotriton/util.h>
#include <flash/shim.attn_fwd.h>
#include <flash/iface.op_attn_fwd.h>

namespace AOTRITON_NS::v3::flash {

dim3 AttnFwdContext::grid_calculator() const {
    AOTRITON_LOG(LOG_DEBUG,
                 "Selected Kernel BLOCK_M = %d BLOCK_N = %d PRE_LOAD_V = %d",
                 int(this->BLOCK_M), int(this->BLOCK_N), int(this->PRE_LOAD_V));
    bool unsupported_by_persistent = params->Num_seqlens != 0;
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
  if (params_version != attn_fwd_params::kVersion) {
    return hipErrorInvalidSymbol; // params_version mismatch
  }
  hipError_t err;
  auto stream = stream_wrap.native();
  auto gpu = getGpuFromStream(stream);
  int batch = in.Q.size(0);
  int hdim_qk = in.Q.size(3);
  int hdim_vo = in.V.size(3);
  int hdim_max = std::max(hdim_qk, hdim_vo);
  int num_head_q = in.Q.size(1);
  int num_head_k = in.K.size(1);
  int num_seqlens = 0;
  int max_seqlen_q = in.Q.size(2);
  int max_seqlen_k = in.K.size(2);
  if (in.cu_seqlens_q) {
    // Compact varlen, num_seqlens > 0
    num_seqlens = in.cu_seqlens_q.size(0) - 1;
    max_seqlen_q = in.Max_seqlen_q;
  }
  if (in.cu_seqlens_k) {
    max_seqlen_k = in.Max_seqlen_k;
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
    .Num_seqlens = in.varlen_type == VarlenType::PaddedVarlen ? -num_seqlens : num_seqlens,
    .cu_seqlens_q = &in.cu_seqlens_q,
    .cu_seqlens_k = &in.cu_seqlens_k,
    .Max_seqlen_q = max_seqlen_q,
    .Max_seqlen_k = max_seqlen_k,
    .seq_strides_q = &in.seq_strides_q,
    .seq_strides_k = &in.seq_strides_k,
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
    .Batch = int32_t(num_seqlens == 0 ? batch : num_seqlens),
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
