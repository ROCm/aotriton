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
#include <iostream>

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
  dim3 grid {
    AOTRITON_NS::cdiv<uint32_t>(params->max_seqlen_k, this->BLOCK_N),
    uint32_t(params->K->size(1)),
    params->num_seqlens == 0 ? uint32_t(params->Q->size(0)) : std::abs(params->num_seqlens),
  };
  // std::cerr << "bwd_kernel_dk_dv grid conf " << grid.x << " " << grid.y << " " << grid.z << std::endl;
  return grid;
}

dim3 BwdKernelDqContext::grid_calculator() const {
  dim3 grid {
    AOTRITON_NS::cdiv<uint32_t>(params->max_seqlen_q, this->BLOCK_M),
    uint32_t(params->Q->size(1)),
    params->num_seqlens == 0 ? uint32_t(params->Q->size(0)) : std::abs(params->num_seqlens),
  };
  // std::cerr << "bwd_kernel_dq grid conf " << grid.x << " " << grid.y << " " << grid.z << std::endl;
  return grid;
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
#ifndef NDEBUG
  std::cerr << "v3::flash::attn_bwd options = " << options << std::endl;
  if (options) {
    std::cerr << "v3::flash::attn_bwd options->force_backend_index = "
              << options->force_backend_index
              << std::endl;
  }
#endif
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
#ifndef NDEBUG
  std::cerr << "v3::flash::attn_bwd context.backend_index = " << context.backend_index << std::endl;
#endif
  err = context.launch(gpu, stream);
  in.D.free();
  in.DQ_ACC.free();
  return err;
}

}

// V2 API for backward compatibility

namespace AOTRITON_NS::v2::flash {

using BwdPreprocessParams = AOTRITON_NS::v3::flash::OpAttnBwdParams;
using BwdPreprocessVarlenParams = AOTRITON_NS::v3::flash::OpAttnBwdParams;
using BwdKernelDkDvParams = AOTRITON_NS::v3::flash::OpAttnBwdParams;
using BwdKernelDqParams = AOTRITON_NS::v3::flash::OpAttnBwdParams;

using BwdPreprocessContext        = AOTRITON_NS::v3::flash::BwdPreprocessContext;
using BwdPreprocessVarlenContext  = AOTRITON_NS::v3::flash::BwdPreprocessVarlenContext;
using BwdKernelDkDvContext        = AOTRITON_NS::v3::flash::BwdKernelDkDvContext;
using BwdKernelDqContext          = AOTRITON_NS::v3::flash::BwdKernelDqContext;

using BwdPreprocessMetadata        = AOTRITON_NS::v3::flash::BwdPreprocessMetadata;
using BwdPreprocessVarlenMetadata  = AOTRITON_NS::v3::flash::BwdPreprocessVarlenMetadata;
using BwdKernelDkDvMetadata        = AOTRITON_NS::v3::flash::BwdKernelDkDvMetadata;
using BwdKernelDqMetadata          = AOTRITON_NS::v3::flash::BwdKernelDqMetadata;

using CausalType = AOTRITON_NS::v3::flash::CausalType;
using WindowValue = AOTRITON_NS::v3::flash::WindowValue;

hipError_t
bwd_preprocess(T4 out, T4 dout, T2 delta, AOTRITON_NS::Stream stream_wrap) {
  hipError_t err;
  auto stream = stream_wrap.native();
  auto gpu = getGpuFromStream(stream);
  // Note: do not unify this constexpr.
  //       Different kernels may have different rules.
  constexpr int kMinHeadDimCompiled = 16;
  int hdim_o = out.size(3);
  const auto& compiled_head_dims = BwdPreprocessMetadata::get_BLOCK_DMODEL_choices();
  int hdim_rounded = round_value(hdim_o, compiled_head_dims);
  // FIXME: Remove when compiler bug fixed
  if (Gpu2VendorArch(gpu) == CAT32(GpuVendor::kAMD, 0x950)) {
    if (hdim_rounded == 48)
      hdim_rounded = 64;
    if (hdim_rounded == 80)
      hdim_rounded = 96;
  }
  if (hdim_rounded < 0) {
#if AOTRITON_VERBOSE
    std::cerr << "Head dimension " << hdim_o << " unsupported. ";
    if (compiled_head_dims.empty()) {
      std::cerr << "No head dimension (BLOCK_DMODEL) compiled into the binary." << std::endl;
    } else {
      std::cerr << "Maximal dimesion compiled into the binary is "
                << compiled_head_dims.back()
                << std::endl;
    }
#endif
    return hipErrorInvalidValue;
  }
  EagerLazyTensor<2> eager_lazy_delta(delta);
  auto cu_seqlens_placeholder = T1::get_null_tensor(DType::kInt32);
  // Requires C++ 20
  BwdPreprocessParams params = {
    .Out = &out,
    .DO = &dout,
    .D = &eager_lazy_delta,
    .cu_seqlens_q = &cu_seqlens_placeholder,
    .num_seqlens = 0,   // Note: V2 API does not support padded varlen
    .max_seqlen_q = static_cast<int32_t>(out.size(2)),
    .hdim_vo = hdim_o,
    .BLOCK_DMODEL = int16_t(hdim_rounded),
    .PADDED_HEAD = hdim_rounded != hdim_o,
  };
  BwdPreprocessContext context;
  context.params = &params;
  err = context.lookup_optimal(gpu);
  if (err != hipSuccess) {
    return err;
  }
  err = context.launch(stream);
  return err;
}

hipError_t
bwd_preprocess_varlen(T4 out,
                      T4 dout,
                      T2 delta,
                      T1 cu_seqlens_q,
                      int32_t max_seqlen_q,
                      AOTRITON_NS::Stream stream_wrap) {
  hipError_t err;
  auto stream = stream_wrap.native();
  auto gpu = getGpuFromStream(stream);
  // Note: do not unify this constexpr.
  //       Different kernels may have different rules.
  constexpr int kMinHeadDimCompiled = 16;
  int hdim_o = out.size(3);
  const auto& compiled_head_dims = BwdPreprocessVarlenMetadata::get_BLOCK_DMODEL_choices();
  int hdim_rounded = round_value(hdim_o, compiled_head_dims);
  // FIXME: Remove when compiler bug fixed
  if (Gpu2VendorArch(gpu) == CAT32(GpuVendor::kAMD, 0x950)) {
    if (hdim_rounded == 48)
      hdim_rounded = 64;
    if (hdim_rounded == 80)
      hdim_rounded = 96;
  }
  if (hdim_rounded < 0) {
#if AOTRITON_VERBOSE
    std::cerr << "Head dimension " << hdim_o << " unsupported. ";
    if (compiled_head_dims.empty()) {
      std::cerr << "No head dimension (BLOCK_DMODEL) compiled into the binary." << std::endl;
    } else {
      std::cerr << "Maximal dimesion compiled into the binary is "
                << compiled_head_dims.back()
                << std::endl;
    }
#endif
    return hipErrorInvalidValue;
  }
  EagerLazyTensor eager_lazy_delta(delta);
  // Requires C++ 20
  BwdPreprocessVarlenParams params = {
    .Out = &out,
    .DO = &dout,
    .D = &eager_lazy_delta,
    .cu_seqlens_q = &cu_seqlens_q,
    .num_seqlens = static_cast<int32_t>(cu_seqlens_q.size(0) - 1),  // Note: V2 API does not support strided varlen
    .max_seqlen_q = max_seqlen_q,
    .hdim_vo = hdim_o,
    .BLOCK_DMODEL = int16_t(hdim_rounded),
    .PADDED_HEAD = hdim_rounded != hdim_o,
  };
  BwdPreprocessVarlenContext context;
  context.params = &params;
  err = context.lookup_optimal(gpu);
  if (err != hipSuccess) {
    return err;
  }
  err = context.launch(stream);
  return err;
}

hipError_t
bwd_kernel_dk_dv(T4 q,
                 T4 k,
                 T4 v,
                 T1 cu_seqlens_q,
                 T1 cu_seqlens_k,
                 int32_t num_seqlens,
                 int32_t max_seqlen_q,
                 int32_t max_seqlen_k,
                 T4 b,
                 float sm_scale,
                 T4 out,
                 T4 dout,
                 T4 dk,
                 T4 dv,
                 T2 softmax_lse,
                 T2 delta,
                 float dropout_p,
                 T0 philox_seed,
                 T0 philox_offset1,
                 int64_t philox_offset2,
                 bool is_causal,
                 AOTRITON_NS::Stream stream_wrap,
                 BwdExtraArguments* extargs) {
  hipError_t err;
  auto stream = stream_wrap.native();
  auto gpu = getGpuFromStream(stream);
  constexpr int kMinHeadDimCompiled = 16;
  int hdim_qk = q.size(3);
  int hdim_vo = v.size(3);
  int hdim_max = std::max(hdim_qk, hdim_vo);
  int num_head_q = q.size(1);
  int num_head_k = k.size(1);
  const auto& compiled_head_dims = BwdKernelDkDvMetadata::get_BLOCK_DMODEL_choices();
  int hdim_rounded = round_value(hdim_max, compiled_head_dims);
  // FIXME: Remove when compiler bug fixed
  if (Gpu2VendorArch(gpu) == CAT32(GpuVendor::kAMD, 0x950)) {
    if (hdim_rounded == 48)
      hdim_rounded = 64;
    if (hdim_rounded == 80)
      hdim_rounded = 96;
  }
  if (hdim_rounded < 0) {
#if AOTRITON_VERBOSE
    std::cerr << "Head dimension " << hdim_max << " unsupported. ";
    if (compiled_head_dims.empty()) {
      std::cerr << "No head dimension (BLOCK_DMODEL) compiled into the binary." << std::endl;
    } else {
      std::cerr << "Maximal dimesion compiled into the binary is "
                << compiled_head_dims.back()
                << std::endl;
    }
#endif
    return hipErrorInvalidValue;
  }
  int8_t bias_type = 0;
  if (b) {
    bias_type = 1;
  }
  EagerLazyTensor eager_lazy_delta(delta);
  auto seq_strides_placeholder = T1::get_null_tensor(DType::kInt32);
  BwdKernelDkDvParams params = {
    .Q = &q,
    .K = &k,
    .V = &v,
    .B = &b,
    .sm_scale = sm_scale,
    .Out = &out,
    .DO = &dout,
    .DK = &dk,
    .DV = &dv,
    .L = &softmax_lse,
    .D = &eager_lazy_delta,
    .num_head_q = num_head_q,
    .num_head_k = num_head_k,
    .cu_seqlens_q = &cu_seqlens_q,
    .cu_seqlens_k = &cu_seqlens_k,
    .num_seqlens = num_seqlens,
    .max_seqlen_q = max_seqlen_q,
    .max_seqlen_k = max_seqlen_k,
    .seq_strides_q = &seq_strides_placeholder,
    .seq_strides_k = &seq_strides_placeholder,
    .hdim_qk = hdim_qk,
    .hdim_vo = hdim_vo,
    .dropout_p = dropout_p,
    .philox_seed_ptr = &philox_seed,
    .philox_offset1 = &philox_offset1,
    .philox_offset2 = static_cast<uint64_t>(philox_offset2),
    .Window_left = WindowValue::TopLeftAligned,
    .Window_right = WindowValue::TopLeftAligned,
    .BLOCK_DMODEL = int16_t(hdim_rounded),
    .CAUSAL_TYPE = is_causal ? CausalType::WindowedAttention : CausalType::None,
    .ENABLE_DROPOUT = dropout_p > 0.0,
    .PADDED_HEAD = (hdim_qk != hdim_rounded || hdim_vo != hdim_rounded),
    .BIAS_TYPE = bias_type,
  };
  BwdKernelDkDvContext context;
  context.params = &params;
#if AOTRITON_BUILD_FOR_TUNING
  if (extargs) {
    context._has_preferred_kernel = extargs->dkdv.force_kernel_index;
    if (context._has_preferred_kernel == CppTuneSpecialKernelIndex::kSkipGPUCall) {
        // std::cerr << "extargs->dkdv.force_kernel_index = " << extargs->dkdv.force_kernel_index << " EKI" << std::endl;
        return hipSuccess;
    }
  }
#endif
  err = context.lookup_optimal(gpu);
#if AOTRITON_BUILD_FOR_TUNING
  if (extargs) {
    extargs->dkdv.total_number_of_kernels = context._total_number_of_kernels;
    extargs->dkdv.selected_kernel_psels = context._preferred_kernel_psels;
    extargs->dkdv.selected_kernel_copts = context._preferred_kernel_copts;
    context.peek_kernel_image = extargs->dkdv.peek_kernel_image;
    if (extargs->dkdv.peek_kernel_numbers)
      return hipSuccess;
  }
#endif
  if (err != hipSuccess) {
    return err;
  }
  err = context.launch(stream);
#if AOTRITON_BUILD_FOR_TUNING
  if (extargs && extargs->dkdv.peek_kernel_image) {
    auto essentials = context.kernel_on_device->get_image_info_iff_decompressed();
    extargs->dkdv.kernel_image = essentials.image;
    extargs->dkdv.image_size = essentials.size;
  }
#endif
  return err;
}

hipError_t
bwd_kernel_dq(T4 q,
              T4 k,
              T4 v,
              T1 cu_seqlens_q,
              T1 cu_seqlens_k,
              int32_t num_seqlens,
              int32_t max_seqlen_q,
              int32_t max_seqlen_k,
              T4 b,
              float sm_scale,
              T4 out,
              T4 dout,
              T4 dq,
              T4 db,
              T2 softmax_lse,
              T2 delta,
              float dropout_p,
              T0 philox_seed,
              T0 philox_offset1,
              int64_t philox_offset2,
              bool is_causal,
              AOTRITON_NS::Stream stream_wrap,
              BwdExtraArguments* extargs) {
  hipError_t err;
  auto stream = stream_wrap.native();
  auto gpu = getGpuFromStream(stream);
  constexpr int kMinHeadDimCompiled = 16;
  int hdim_qk = q.size(3);
  int hdim_vo = v.size(3);
  int hdim_max = std::max(hdim_qk, hdim_vo);
  int num_head_q = q.size(1);
  int num_head_k = k.size(1);
  // TODO: Add metadata to operators
  const auto& compiled_head_dims = BwdKernelDqMetadata::get_BLOCK_DMODEL_choices();
  int hdim_rounded = round_value(hdim_max, compiled_head_dims);
  // FIXME: Remove when compiler bug fixed
  if (Gpu2VendorArch(gpu) == CAT32(GpuVendor::kAMD, 0x950)) {
    if (hdim_rounded == 48)
      hdim_rounded = 64;
    if (hdim_rounded == 80)
      hdim_rounded = 96;
  }
  if (hdim_rounded < 0) {
#if AOTRITON_VERBOSE
    std::cerr << "Head dimension " << hdim_max << " unsupported. ";
    if (compiled_head_dims.empty()) {
      std::cerr << "No head dimension (BLOCK_DMODEL) compiled into the binary." << std::endl;
    } else {
      std::cerr << "Maximal dimesion compiled into the binary is "
                << compiled_head_dims.back()
                << std::endl;
    }
#endif
    return hipErrorInvalidValue;
  }
  int8_t bias_type = 0;
  if (b) {
    bias_type = 1;
  }
  EagerLazyTensor eager_lazy_delta(delta);
  auto seq_strides_placeholder = T1::get_null_tensor(DType::kInt32);
  BwdKernelDqParams params = {
    .Q = &q,
    .K = &k,
    .V = &v,
    .B = &b,
    .sm_scale = sm_scale,
    .Out = &out,
    .DO = &dout,
    .DQ = &dq,
    .DB = &db,
    .L = &softmax_lse,
    .D = &eager_lazy_delta,
    .num_head_q = num_head_q,
    .num_head_k = num_head_k,
    .cu_seqlens_q = &cu_seqlens_q,
    .cu_seqlens_k = &cu_seqlens_k,
    .num_seqlens = num_seqlens,
    .max_seqlen_q = max_seqlen_q,
    .max_seqlen_k = max_seqlen_k,
    .seq_strides_q = &seq_strides_placeholder,
    .seq_strides_k = &seq_strides_placeholder,
    .hdim_qk = hdim_qk,
    .hdim_vo = hdim_vo,
    .dropout_p = dropout_p,
    .philox_seed_ptr = &philox_seed,
    .philox_offset1 = &philox_offset1,
    .philox_offset2 = static_cast<uint64_t>(philox_offset2),
    .Window_left = WindowValue::TopLeftAligned,
    .Window_right = WindowValue::TopLeftAligned,
    .BLOCK_DMODEL = int16_t(hdim_rounded),
    .CAUSAL_TYPE = is_causal ? CausalType::WindowedAttention : CausalType::None,
    .ENABLE_DROPOUT = dropout_p > 0.0,
    .PADDED_HEAD = (hdim_qk != hdim_rounded || hdim_vo != hdim_rounded),
    .BIAS_TYPE = bias_type,
  };
  BwdKernelDqContext context;
  context.params = &params;
#if AOTRITON_BUILD_FOR_TUNING
  if (extargs) {
    context._has_preferred_kernel = extargs->dqdb.force_kernel_index;
    if (context._has_preferred_kernel == CppTuneSpecialKernelIndex::kSkipGPUCall) {
        // std::cerr << "extargs->dqdb.force_kernel_index = " << extargs->dqdb.force_kernel_index << " EKI" << std::endl;
        return hipSuccess;
    }
  }
#endif
  err = context.lookup_optimal(gpu);
#if AOTRITON_BUILD_FOR_TUNING
  if (extargs) {
    extargs->dqdb.total_number_of_kernels = context._total_number_of_kernels;
    extargs->dqdb.selected_kernel_psels = context._preferred_kernel_psels;
    extargs->dqdb.selected_kernel_copts = context._preferred_kernel_copts;
    context.peek_kernel_image = extargs->dqdb.peek_kernel_image;
    // std::cerr << "dqdb lookup_optimal = " << err << " EOL" << std::endl;
    if (extargs->dqdb.peek_kernel_numbers)
      return hipSuccess;
  }
#endif
  if (err != hipSuccess) {
    return err;
  }
  err = context.launch(stream);
#if AOTRITON_BUILD_FOR_TUNING
  if (extargs && extargs->dqdb.peek_kernel_image) {
    auto essentials = context.kernel_on_device->get_image_info_iff_decompressed();
    extargs->dqdb.kernel_image = essentials.image;
    extargs->dqdb.image_size = essentials.size;
  }
#endif
  return err;
}

hipError_t
_attn_bwd_common(T4 q,
                 T4 k,
                 T4 v,
                 T1 cu_seqlens_q,
                 T1 cu_seqlens_k,
                 int32_t num_seqlens,
                 int32_t max_seqlen_q,
                 int32_t max_seqlen_k,
                 T4 b,
                 float sm_scale,
                 T4 out,
                 T4 dout,
                 T4 dq,
                 T4 dk,
                 T4 dv,
                 T4 db,
                 T2 softmax_lse,
                 T2 delta,
                 float dropout_p,
                 T0 philox_seed,
                 T0 philox_offset1,
                 int64_t philox_offset2,
                 bool is_causal,
                 AOTRITON_NS::Stream stream,
                 BwdExtraArguments* extargs) {
  hipError_t ret;
  if (num_seqlens == 0)
    ret = bwd_preprocess(out, dout, delta, stream);
  else
    ret = bwd_preprocess_varlen(out, dout, delta, cu_seqlens_q, max_seqlen_q, stream);
  if (ret != hipSuccess)
    return ret;
  ret = bwd_kernel_dk_dv(q,
                         k,
                         v,
                         cu_seqlens_q,
                         cu_seqlens_k,
                         num_seqlens,
                         max_seqlen_q,
                         max_seqlen_k,
                         b,
                         sm_scale,
                         out,
                         dout,
                         dk,
                         dv,
                         softmax_lse,
                         delta,
                         dropout_p,
                         philox_seed,
                         philox_offset1,
                         philox_offset2,
                         is_causal,
                         stream,
                         extargs);

  if (ret != hipSuccess)
    return ret;
  ret = bwd_kernel_dq(q,
                      k,
                      v,
                      cu_seqlens_q,
                      cu_seqlens_k,
                      num_seqlens,
                      max_seqlen_q,
                      max_seqlen_k,
                      b,
                      sm_scale,
                      out,
                      dout,
                      dq,
                      db,
                      softmax_lse,
                      delta,
                      dropout_p,
                      philox_seed,
                      philox_offset1,
                      philox_offset2,
                      is_causal,
                      stream,
                      extargs);
  return ret;
}

hipError_t
attn_bwd(T4 q,
         T4 k,
         T4 v,
         T4 b,
         float sm_scale,
         T4 out,
         T4 dout,
         T4 dq,
         T4 dk,
         T4 dv,
         T4 db,
         T2 softmax_lse,
         T2 delta,
         float dropout_p,
         T0 philox_seed,
         T0 philox_offset1,
         int64_t philox_offset2,
         bool is_causal,
         AOTRITON_NS::Stream stream,
         BwdExtraArguments* extargs) {
  auto null_t1 = T1::get_null_tensor(DType::kInt32);
  return _attn_bwd_common(q,
                          k,
                          v,
                          null_t1,
                          null_t1,
                          0,
                          q.size(2),
                          k.size(2),
                          b,
                          sm_scale,
                          out,
                          dout,
                          dq,
                          dk,
                          dv,
                          db,
                          softmax_lse,
                          delta,
                          dropout_p,
                          philox_seed,
                          philox_offset1,
                          philox_offset2,
                          is_causal,
                          stream,
                          extargs);
}

hipError_t
attn_bwd_compact_varlen(T4 q,            // 1 x num_heads x total_q x hdim_qk, total_q := \sum_{i=0}^{b}
                        T4 k,            // 1 x num_heads x total_k x hdim_qk, total_k := \sum_{i=0}^{b}
                        T4 v,            // 1 x num_heads x total_v x hdim_vo, total_, := \sum_{i=0}^{b}
                        T1 cu_seqlens_q, // b+1, i64
                        T1 cu_seqlens_k, // b+1, i64
                        int32_t max_seqlen_q,
                        int32_t max_seqlen_k,
                        T4 b, // reserved
                        float sm_scale,
                        T4 out,  // batch_size x num_heads x seqlen_q x hdim_vo
                        T4 dout, // batch_size x num_heads x seqlen_q x hdim_vo
                        T4 dq,   // batch_size x num_heads x seqlen_q x hdim_qk
                        T4 dk,   // batch_size x num_heads x seqlen_k x hdim_qk
                        T4 dv,   // batch_size x num_heads x seqlen_k x hdim_vo
                        T4 db,   // batch_size x num_heads x seqlen_q x seqlen_k
                        T2 softmax_lse,
                        T2 delta, // buffer, empty_like(softmax_lse)
                        float dropout_p,
                        T0 philox_seed,
                        T0 philox_offset1,
                        int64_t philox_offset2,
                        bool is_causal,
                        AOTRITON_NS::Stream stream,
                        BwdExtraArguments* extargs) {
  return _attn_bwd_common(q,
                          k,
                          v,
                          cu_seqlens_q,
                          cu_seqlens_k,
                          cu_seqlens_q.size(0) - 1,
                          max_seqlen_q,
                          max_seqlen_k,
                          b,
                          sm_scale,
                          out,
                          dout,
                          dq,
                          dk,
                          dv,
                          db,
                          softmax_lse,
                          delta,
                          dropout_p,
                          philox_seed,
                          philox_offset1,
                          philox_offset2,
                          is_causal,
                          stream,
                          extargs);
}

}
