// Copyright © 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/config.h>
#include <aotriton/_internal/util.h>
#include <aotriton/flash.h>
#include <aotriton/util.h>
#include <flash/shim.bwd_kernel_fuse.h>
#include <flash/iface.op_attn_bwd.h>
#include <iostream>

namespace AOTRITON_NS::v3::flash {

dim3 BwdKernelFuseContext::grid_calculator() const {
  dim3 grid {
    AOTRITON_NS::cdiv<uint32_t>(params->max_seqlen_k, this->BLOCK_N) +
    AOTRITON_NS::cdiv<uint32_t>(params->max_seqlen_q, this->BLOCK_N) * (params->num_head_q / params->num_head_k),
    uint32_t(params->K->size(1)),
    params->num_seqlens == 0 ? uint32_t(params->Q->size(0)) : params->num_seqlens,
  };
  // std::cerr << "bwd_kernel_dk_dv grid conf " << grid.x << " " << grid.y << " " << grid.z << std::endl;
  return grid;
}

}

namespace AOTRITON_NS::v2::flash {

using BwdKernelFuseParams = AOTRITON_NS::v3::flash::OpAttnBwdParams;
using BwdKernelFuseContext = AOTRITON_NS::v3::flash::BwdKernelFuseContext;
using BwdKernelFuseMetadata = AOTRITON_NS::v3::flash::BwdKernelFuseMetadata;
using CausalType = AOTRITON_NS::v3::flash::CausalType;
using WindowValue = AOTRITON_NS::v3::flash::WindowValue;

hipError_t
_bwd_kernel_fuse(T4 q,
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
                 float dropout_p,
                 T0 philox_seed,
                 T0 philox_offset1,
                 int64_t philox_offset2,
                 bool is_causal,
                 AOTRITON_NS::Stream stream_wrap,
                 FusedBwdExtraArguments* extargs) {
  hipError_t err;
  auto stream = stream_wrap.native();
  auto gpu = getGpuFromStream(stream);
  int num_head_q = q.size(1);
  int num_head_k = k.size(1);
  int head_size = q.size(3);
  const auto& compiled_head_dims = BwdKernelFuseMetadata::get_BLOCK_DMODEL_choices();
  int head_size_rounded = round_value(head_size, compiled_head_dims);
  // FIXME: Remove when compiler bug fixed
  if (Gpu2VendorArch(gpu) == CAT32(GpuVendor::kAMD, 0x950)) {
    if (head_size_rounded == 48)
      head_size_rounded = 64;
    if (head_size_rounded == 80)
      head_size_rounded = 96;
  }
  if (head_size_rounded < 0) {
#if AOTRITON_VERBOSE
    std::cerr << "Head dimension " << head_size << " unsupported. ";
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
  int bias_type = 0;
  if (b) {
    bias_type = 1;
  }
  BwdKernelFuseParams params = {
    .Q = &q,
    .K = &k,
    .V = &v,
    .B = &b,
    .sm_scale = sm_scale,
    .Out = &out,
    .DO = &dout,
    .DK = &dk,
    .DV = &dv,
    .DQ = &dq,
    .DB = &db,
    .L = &softmax_lse,
    .num_head_q = num_head_q,
    .num_head_k = num_head_k,
    .cu_seqlens_q = &cu_seqlens_q,
    .cu_seqlens_k = &cu_seqlens_k,
    .num_seqlens = num_seqlens,
    .max_seqlen_q = max_seqlen_q,
    .max_seqlen_k = max_seqlen_k,
    .head_dim = head_size,
    .dropout_p = dropout_p,
    .philox_seed_ptr = &philox_seed,
    .philox_offset1 = &philox_offset1,
    .philox_offset2 = static_cast<uint64_t>(philox_offset2),
    .Window_left = WindowValue::TopLeftAligned,
    .Window_right = WindowValue::TopLeftAligned,
    .BLOCK_DMODEL = static_cast<int8_t>(head_size_rounded),
    .CAUSAL_TYPE = is_causal ? CausalType::WindowedAttention : CausalType::None,
    .ENABLE_DROPOUT = dropout_p > 0.0,
    .PADDED_HEAD = head_size_rounded != head_size,
    .BIAS_TYPE = static_cast<int8_t>(bias_type),
  };
  BwdKernelFuseContext context;
  context.params = &params;
#if AOTRITON_BUILD_FOR_TUNING
  if (extargs) {
    context._has_preferred_kernel = extargs->force_kernel_index;
    if (context._has_preferred_kernel == CppTuneSpecialKernelIndex::kSkipGPUCall) {
        // std::cerr << "extargs->force_kernel_index = " << extargs->force_kernel_index << " EKI" << std::endl;
        return hipSuccess;
    }
  }
#endif
  err = context.lookup_optimal(gpu);
#if AOTRITON_BUILD_FOR_TUNING
  if (extargs) {
    extargs->total_number_of_kernels = context._total_number_of_kernels;
    extargs->selected_kernel_psels = context._preferred_kernel_psels;
    extargs->selected_kernel_copts = context._preferred_kernel_copts;
    context.peek_kernel_image = extargs->peek_kernel_image;
#if AOTRITON_VERBOSE
    std::cerr << "extargs->peek_kernel_image " << extargs->peek_kernel_image << std::endl;
#endif
  }
#endif
  if (err != hipSuccess) {
    return err;
  }
  err = context.launch(stream);
#if AOTRITON_BUILD_FOR_TUNING
  if (extargs && extargs->peek_kernel_image) {
    auto essentials = context.kernel_on_device->get_image_info_iff_decompressed();
    extargs->kernel_image = essentials.image;
    extargs->image_size = essentials.size;
#if AOTRITON_VERBOSE
    std::cerr << "peek_kernel_image returns image at: " << essentials.image
              << " size: " << essentials.size << std::endl;
#endif
  }
#endif
  return err;
}

hipError_t
attn_bwd_fused(T4 q,
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
               float dropout_p,
               T0 philox_seed,
               T0 philox_offset1,
               int64_t philox_offset2,
               bool is_causal,
               AOTRITON_NS::Stream stream,
               FusedBwdExtraArguments* extargs) {
  auto null_t1 = T1::get_null_tensor(DType::kInt32);
  hipError_t ret;
  ret =  _bwd_kernel_fuse(q,
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
                          dropout_p,
                          philox_seed,
                          philox_offset1,
                          philox_offset2,
                          is_causal,
                          stream,
                          extargs);
    return ret;
}

}
