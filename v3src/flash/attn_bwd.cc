// Copyright © 2023-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/config.h>
#include <aotriton/_internal/util.h>
#include <aotriton/flash.h>
#include <aotriton/util.h>
#include <flash/shim.bwd_kernel_dk_dv.h>
#include <flash/shim.bwd_kernel_dq.h>
#include <flash/shim.bwd_preprocess.h>
#include <flash/shim.bwd_preprocess_varlen.h>
#include <flash/iface.op_attn_bwd.h>
#include <aotriton/_internal/log.h>

namespace AOTRITON_NS::v3::flash {

dim3 BwdPreprocessContext::grid_calculator() const {
  dim3 grid {
    AOTRITON_NS::cdiv<uint32_t>(params->Out->size(2), this->BLOCK_M),
    uint32_t(params->Out->size(1)),
    uint32_t(params->Out->size(0)),
  };
  // std::cerr << "Grid conf " << grid.x << " " << grid.y << " " << grid.z << std::endl;
  return grid;
}

dim3 BwdPreprocessVarlenContext::grid_calculator() const {
  dim3 grid {
    AOTRITON_NS::cdiv<uint32_t>(params->Out->size(2), this->BLOCK_M),
    uint32_t(params->Out->size(1)),
    uint32_t(params->cu_seqlens_q->size(0) - 1),
  };
  // std::cerr << "Grid conf " << grid.x << " " << grid.y << " " << grid.z << std::endl;
  return grid;
}

dim3 BwdKernelDkDvContext::grid_calculator() const {
  auto S = AOTRITON_NS::cdiv<uint32_t>(params->max_seqlen_k, this->BLOCK_N);
  auto H = uint32_t(params->K->size(1));
  auto B = params->num_seqlens == 0 ? uint32_t(params->Q->size(0)) : std::abs(params->num_seqlens);
  return NUM_XCDS > 1 ? dim3 { H, S, B } : dim3 { S, H, B };
}

dim3 BwdKernelDqContext::grid_calculator() const {
  auto S = AOTRITON_NS::cdiv<uint32_t>(params->max_seqlen_q, this->BLOCK_M);
  auto H = uint32_t(params->Q->size(1));
  auto B = params->num_seqlens == 0 ? uint32_t(params->Q->size(0)) : std::abs(params->num_seqlens);
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
  if (params_version != attn_bwd_params::kVersion) {
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
  int max_seqlen_q = in.Q.size(2);
  int max_seqlen_k = in.K.size(2);
  int num_seqlens = 0;
  if (in.cu_seqlens_q) {
    // Compact varlen, num_seqlens > 0
    num_seqlens = in.cu_seqlens_q.size(0) - 1;
    max_seqlen_q = in.Max_seqlen_q;
  }
  if (in.cu_seqlens_k) {
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
    .cu_seqlens_q = &in.cu_seqlens_q,
    .cu_seqlens_k = &in.cu_seqlens_k,
    .num_seqlens = in.varlen_type == VarlenType::PaddedVarlen ? -num_seqlens : num_seqlens,
    .max_seqlen_q = max_seqlen_q,
    .max_seqlen_k = max_seqlen_k,
    .seq_strides_q = &in.seq_strides_q,
    .seq_strides_k = &in.seq_strides_k,
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
  AOTRITON_LOG(LOG_DEBUG, "v3::flash::attn_bwd options = {}", static_cast<const void*>(options));
  if (options) {
    AOTRITON_LOG(LOG_DEBUG, "v3::flash::attn_bwd options->force_backend_index = {}",
                 options->force_backend_index);
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
  AOTRITON_LOG(LOG_DEBUG, "v3::flash::attn_bwd context.backend_index = {}", context.backend_index);
  err = context.launch(gpu, stream);
  in.D.free();
  in.DQ_ACC.free();
  return err;
}

}
