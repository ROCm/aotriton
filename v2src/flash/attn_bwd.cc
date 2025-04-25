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
#include <iostream>

namespace AOTRITON_NS::v2::flash {

hipError_t
bwd_preprocess(T4 out, T4 dout, T2 delta, AOTRITON_NS::Stream stream_wrap) {
  hipError_t err;
  auto stream = stream_wrap.native();
  auto gpu = getGpuFromStream(stream);
  auto grid_calculator = [](const BwdPreprocessParams& params) -> dim3 {
    dim3 grid {
      AOTRITON_NS::cdiv<uint32_t>(params.Out->size(2), params.BLOCK_M),
      uint32_t(params.Out->size(1)),
      uint32_t(params.Out->size(0)),
    };
    // std::cerr << "Grid conf " << grid.x << " " << grid.y << " " << grid.z << std::endl;
    return grid;
  };
  // Note: do not unify this constexpr.
  //       Different kernels may have different rules.
  constexpr int kMinHeadDimCompiled = 16;
  int head_size = out.size(3);
  const auto& compiled_head_dims = BwdPreprocessMetadata::get_D_HEAD_choices();
  int head_size_rounded = round_value(head_size, compiled_head_dims);
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
  // Requires C++ 20
  BwdPreprocessParams params = {
    .Out = &out,
    .DO = &dout,
    .Delta = &delta,
    .seqlen_q = static_cast<int32_t>(out.size(2)),
    .head_dim = head_size,
    .D_HEAD = head_size_rounded,
    .PADDED_HEAD = head_size_rounded != head_size,
  };
  BwdPreprocessContext context;
  context.grid_calculator = grid_calculator;
  err = context.lookup_optimal(params, gpu);
  if (err != hipSuccess) {
    return err;
  }
  err = context.launch(params, stream);
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
  auto grid_calculator = [](const BwdPreprocessVarlenParams& params) -> dim3 {
    dim3 grid {
      AOTRITON_NS::cdiv<uint32_t>(params.Out->size(2), params.BLOCK_M),
      uint32_t(params.Out->size(1)),
      uint32_t(params.cu_seqlens_q->size(0) - 1),
    };
    // std::cerr << "Grid conf " << grid.x << " " << grid.y << " " << grid.z << std::endl;
    return grid;
  };
  // Note: do not unify this constexpr.
  //       Different kernels may have different rules.
  constexpr int kMinHeadDimCompiled = 16;
  int head_size = out.size(3);
  const auto& compiled_head_dims = BwdPreprocessVarlenMetadata::get_D_HEAD_choices();
  int head_size_rounded = round_value(head_size, compiled_head_dims);
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
  // Requires C++ 20
  BwdPreprocessVarlenParams params = {
    .Out = &out,
    .DO = &dout,
    .Delta = &delta,
    .cu_seqlens_q = &cu_seqlens_q,
    .max_seqlen_q = max_seqlen_q,
    .head_dim = head_size,
    .D_HEAD = head_size_rounded,
    .PADDED_HEAD = head_size_rounded != head_size,
  };
  BwdPreprocessVarlenContext context;
  context.grid_calculator = grid_calculator;
  err = context.lookup_optimal(params, gpu);
  if (err != hipSuccess) {
    return err;
  }
  err = context.launch(params, stream);
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
  auto grid_calculator = [max_seqlen_k](const BwdKernelDkDvParams& params) -> dim3 {
    dim3 grid {
      AOTRITON_NS::cdiv<uint32_t>(max_seqlen_k, params.BLOCK_N),
      uint32_t(params.K->size(1)),
      params.num_seqlens == 0 ? uint32_t(params.Q->size(0)) : params.num_seqlens,
    };
    // std::cerr << "bwd_kernel_dk_dv grid conf " << grid.x << " " << grid.y << " " << grid.z << std::endl;
    return grid;
  };
  constexpr int kMinHeadDimCompiled = 16;
  int head_size = q.size(3);
  int num_head_q = q.size(1);
  int num_head_k = k.size(1);
  const auto& compiled_head_dims = BwdKernelDkDvMetadata::get_BLOCK_DMODEL_choices();
  int head_size_rounded = round_value(head_size, compiled_head_dims);
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
    .D = &delta,
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
    .BLOCK_DMODEL = head_size_rounded,
    .CAUSAL = is_causal,
    .ENABLE_DROPOUT = dropout_p > 0.0,
    .PADDED_HEAD = head_size_rounded != head_size,
    .BIAS_TYPE = bias_type,
  };
#if AOTRITON_BUILD_FOR_TUNING
  if (extargs) {
    params._has_preferred_kernel = extargs->dkdv.force_kernel_index;
    if (params._has_preferred_kernel == CppTuneSpecialKernelIndex::kSkipGPUCall) {
        // std::cerr << "extargs->dkdv.force_kernel_index = " << extargs->dkdv.force_kernel_index << " EKI" << std::endl;
        return hipSuccess;
    }
  }
#endif
  BwdKernelDkDvContext context;
  context.grid_calculator = grid_calculator;
  err = context.lookup_optimal(params, gpu);
#if AOTRITON_BUILD_FOR_TUNING
  if (extargs) {
    extargs->dkdv.total_number_of_kernels = params._total_number_of_kernels;
    extargs->dkdv.selected_kernel_psels = params._preferred_kernel_psels;
    extargs->dkdv.selected_kernel_copts = params._preferred_kernel_copts;
    context.peek_kernel_image = extargs->dkdv.peek_kernel_image;
  }
#endif
  if (err != hipSuccess) {
    return err;
  }
  err = context.launch(params, stream);
#if AOTRITON_BUILD_FOR_TUNING
  if (extargs && extargs->dkdv.peek_kernel_image) {
    auto essentials = params.selected_kernel->get_image_info_iff_decompressed();
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
  auto grid_calculator = [num_seqlens, max_seqlen_q](const BwdKernelDqParams& params) -> dim3 {
    dim3 grid {
      AOTRITON_NS::cdiv<uint32_t>(max_seqlen_q, params.BLOCK_M),
      uint32_t(params.Q->size(1)),
      params.num_seqlens == 0 ? uint32_t(params.Q->size(0)) : params.num_seqlens,
    };
    // std::cerr << "bwd_kernel_dq grid conf " << grid.x << " " << grid.y << " " << grid.z << std::endl;
    return grid;
  };
  constexpr int kMinHeadDimCompiled = 16;
  int head_size = q.size(3);
  int num_head_q = q.size(1);
  int num_head_k = k.size(1);
  const auto& compiled_head_dims = BwdKernelDqMetadata::get_BLOCK_DMODEL_choices();
  int head_size_rounded = round_value(head_size, compiled_head_dims);
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
  BwdKernelDqParams params = {
    .Q = &q,
    .K = &k,
    .V = &v,
    .B = &b,
    .sm_scale = sm_scale,
    .Out = &out,
    .dO = &dout,
    .dQ = &dq,
    .dB = &db,
    .L = &softmax_lse,
    .D = &delta,
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
    .BLOCK_DMODEL = head_size_rounded,
    .CAUSAL = is_causal,
    .ENABLE_DROPOUT = dropout_p > 0.0,
    .PADDED_HEAD = head_size_rounded != head_size,
    .BIAS_TYPE = bias_type,
  };
#if AOTRITON_BUILD_FOR_TUNING
  if (extargs) {
    params._has_preferred_kernel = extargs->dqdb.force_kernel_index;
    if (params._has_preferred_kernel == CppTuneSpecialKernelIndex::kSkipGPUCall) {
        // std::cerr << "extargs->dqdb.force_kernel_index = " << extargs->dqdb.force_kernel_index << " EKI" << std::endl;
        return hipSuccess;
    }
  }
#endif
  BwdKernelDqContext context;
  context.grid_calculator = grid_calculator;
  err = context.lookup_optimal(params, gpu);
#if AOTRITON_BUILD_FOR_TUNING
  if (extargs) {
    extargs->dqdb.total_number_of_kernels = params._total_number_of_kernels;
    extargs->dqdb.selected_kernel_psels = params._preferred_kernel_psels;
    extargs->dqdb.selected_kernel_copts = params._preferred_kernel_copts;
    context.peek_kernel_image = extargs->dqdb.peek_kernel_image;
    // std::cerr << "dqdb lookup_optimal = " << err << " EOL" << std::endl;
  }
#endif
  if (err != hipSuccess) {
    return err;
  }
  err = context.launch(params, stream);
#if AOTRITON_BUILD_FOR_TUNING
  if (extargs && extargs->dqdb.peek_kernel_image) {
    auto essentials = params.selected_kernel->get_image_info_iff_decompressed();
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
attn_bwd_compact_varlen(T4 q,            // 1 x num_heads x total_q x head_size, total_q := \sum_{i=0}^{b}
                        T4 k,            // 1 x num_heads x total_k x head_size, total_k := \sum_{i=0}^{b}
                        T4 v,            // 1 x num_heads x total_v x head_size, total_, := \sum_{i=0}^{b}
                        T1 cu_seqlens_q, // b+1, i64
                        T1 cu_seqlens_k, // b+1, i64
                        int32_t max_seqlen_q,
                        int32_t max_seqlen_k,
                        T4 b, // reserved
                        float sm_scale,
                        T4 out,  // batch_size x num_heads x seqlen_q x head_size
                        T4 dout, // batch_size x num_heads x seqlen_q x head_size
                        T4 dq,   // batch_size x num_heads x seqlen_q x head_size
                        T4 dk,   // batch_size x num_heads x seqlen_k x head_size
                        T4 dv,   // batch_size x num_heads x seqlen_k x head_size
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
