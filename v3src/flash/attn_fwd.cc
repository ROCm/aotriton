// Copyright © 2023-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/config.h>
#include <aotriton/_internal/util.h>
#include <aotriton/flash.h>
#include <aotriton/util.h>
#include <flash/shim.attn_fwd.h>
#include <flash/iface.op_attn_fwd.h>
#include <iostream>

#ifdef NDEBUG
#define AOTRITON_VERBOSE 0
#else
#define AOTRITON_VERBOSE 1
#endif

namespace AOTRITON_NS::v3::flash {

dim3 AttnFwdContext::grid_calculator() const {
#if AOTRITON_VERBOSE
    std::cerr << "Selected Kernel "
              << " BLOCK_M = " << this->BLOCK_M << " BLOCK_N = " << this->BLOCK_N
              << " PRE_LOAD_V = " << this->PRE_LOAD_V << std::endl;
#endif
    bool unsupported_by_persistent = params->Num_seqlens != 0;
    auto nblocks = AOTRITON_NS::cdiv<uint32_t>(params->Max_seqlen_q, this->BLOCK_M);
    // Use default grid if not persistent, or input is unsupported_by_persistent,
    // in which case persistent is turned off IN TRITON KERNEL
    // and this kernel will expect regular grid configs.
    //
    // Note: This fallback behavior is determined by GPU kernel at runtime.
    if (this->PERSISTENT_TYPE == 0 || unsupported_by_persistent) {
      dim3 grid {
        nblocks,
        uint32_t(params->Q->size(1)),
        uint32_t(params->Batch),
      };
#if AOTRITON_VERBOSE
      std::cerr << "Grid conf " << grid.x << " " << grid.y << " " << grid.z << std::endl;
#endif
      return grid;
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
  int head_dim = in.Q.size(3);
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
  int16_t head_dim_rounded = round_value(head_dim, compiled_head_dims);
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
    .Num_seqlens = num_seqlens,
    .cu_seqlens_q = &in.cu_seqlens_q,
    .cu_seqlens_k = &in.cu_seqlens_k,
    .Max_seqlen_q = max_seqlen_q,
    .Max_seqlen_k = max_seqlen_k,
    .BLOCK_DMODEL = head_dim_rounded,
    .Head_dim = head_dim,
    .PADDED_HEAD = head_dim_rounded != head_dim,
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
    .Window_right = in.window_left,
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
  err = context.lookup_optimal(gpu);
  if (err != hipSuccess) {
    return err;
  }
  return context.launch(gpu, stream);
}

} // AOTRITON_NS::v3::flash


namespace AOTRITON_NS::v2::flash {

using AttnFwdParams = AOTRITON_NS::v3::flash::OpAttnFwdParams;
using AttnFwdContext = AOTRITON_NS::v3::flash::AttnFwdContext;
using AttnFwdMetadata = AOTRITON_NS::v3::flash::AttnFwdMetadata;
using CausalType = AOTRITON_NS::v3::flash::CausalType;
using WindowValue = AOTRITON_NS::v3::flash::WindowValue;

hipError_t
_attn_fwd_common(T4 q,
                 T4 k,
                 T4 v,
                 T4 b,
                 T2 a,
                 T1 cu_seqlens_q,
                 T1 cu_seqlens_k,
                 int32_t num_seqlens,
                 int32_t max_seqlen_q,
                 int32_t max_seqlen_k,
                 float sm_scale,
                 T2 softmax_lse,
                 T4 out,
                 float dropout_p,
                 T0 philox_seed,
                 T0 philox_offset1,
                 int64_t philox_offset2,
                 T0 philox_seed_output,
                 T0 philox_offset_output,
                 T4 encoded_softmax,
                 bool is_causal,
                 T0 persistent_atomic_counter,
                 AOTRITON_NS::Stream stream_wrap,
                 FwdExtraArguments* extargs) {
  hipError_t err;
  auto stream = stream_wrap.native();
  auto gpu = getGpuFromStream(stream);
  int head_size = q.size(3);
  int num_head_q = q.size(1);
  int num_head_k = k.size(1);
  const auto& compiled_head_dims = AttnFwdMetadata::get_BLOCK_DMODEL_choices();
  int16_t head_size_rounded = round_value(head_size, compiled_head_dims);
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
  int8_t bias_type = 0;
  // TODO: Replace magic numbers used in BIAS_TYPE, PERSISTENT_TYPE
  if (b) {
    bias_type = 1;
  }
  // Requires C++ 20
  AttnFwdParams params = {
    .Q = &q,
    .K = &k,
    .V = &v,
    .B = &b,
    .A = &a,
    .Sm_scale = sm_scale,
    .L = &softmax_lse,
    .Out = &out,
    .Num_head_q = num_head_q,
    .Num_head_k = num_head_k,
    .Num_seqlens = num_seqlens,
    .cu_seqlens_q = &cu_seqlens_q,
    .cu_seqlens_k = &cu_seqlens_k,
    .Max_seqlen_q = max_seqlen_q,
    .Max_seqlen_k = max_seqlen_k,
    .BLOCK_DMODEL = head_size_rounded,
    .Head_dim = static_cast<int32_t>(head_size),
    .PADDED_HEAD = (head_size_rounded != head_size),
    .ENABLE_DROPOUT = dropout_p > 0.0,
    .dropout_p = dropout_p,
    .philox_seed_ptr = &philox_seed,
    .philox_offset1 = &philox_offset1,
    .philox_offset2 = static_cast<uint64_t>(philox_offset2),
    .philox_seed_output = &philox_seed_output,
    .philox_offset_output = &philox_offset_output,
    .RETURN_ENCODED_SOFTMAX = false,
    .encoded_softmax = &encoded_softmax,
    .CAUSAL_TYPE = is_causal ? CausalType::WindowedAttention : CausalType::None,
    .Window_left = WindowValue::TopLeftAligned,
    .Window_right = WindowValue::TopLeftAligned,
    .BIAS_TYPE = bias_type,
    .USE_ALIBI = false,
    .INT8 = false,
    .INT8_KV = false,
    .USE_P_SCALE = false,
    .persistent_atomic_counter = &persistent_atomic_counter,
    .Num_CU = is_causal ? getMultiProcessorCount(stream) : 80,
    .Batch = int32_t(num_seqlens == 0 ? q.size(0) : num_seqlens),
  };
  AttnFwdContext context;
  context.params = &params;
#if AOTRITON_BUILD_FOR_TUNING
  if (extargs) {
    context._has_preferred_kernel = extargs->force_kernel_index;
    if (context._has_preferred_kernel == CppTuneSpecialKernelIndex::kSkipGPUCall)
        return hipSuccess;
  }
#endif
  // .grid_calculator = grid_calculator
  err = context.lookup_optimal(gpu);
  // Note: PERSISTENT_TYPE is compiled as a perf tuning option, even if it
  //       drastically changes the kernel behaviors
  if (context.PERSISTENT_TYPE == 2 && !persistent_atomic_counter) {
    return hipErrorInvalidValue;  // must have persistent_atomic_counter set
  }
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
  if (err != hipSuccess) {
    return err;
  }
  if (encoded_softmax) {
    return debug_simulate_encoded_softmax(encoded_softmax,
                                          dropout_p,
                                          philox_seed,
                                          philox_offset1,
                                          philox_offset2,
                                          stream_wrap);
  }
  return err;
}

hipError_t
attn_fwd(T4 q,
         T4 k,
         T4 v,
         T4 b,
         float sm_scale,
         T2 softmax_lse,
         T4 out,
         float dropout_p,
         T0 philox_seed,
         T0 philox_offset1,
         int64_t philox_offset2,
         T0 philox_seed_output,
         T0 philox_offset_output,
         T4 encoded_softmax,
         bool is_causal,
         T0 atomic_for_causal,
         AOTRITON_NS::Stream stream_wrap,
         FwdExtraArguments* extargs) {
  auto null_t1 = T1::get_null_tensor(DType::kInt32);
  auto alibi_null_t2 = T2::get_null_tensor(q.dtype());
  return _attn_fwd_common(q,
                          k,
                          v,
                          b,
                          alibi_null_t2,
                          null_t1,
                          null_t1,
                          0,
                          q.size(2),
                          k.size(2),
                          sm_scale,
                          softmax_lse,
                          out,
                          dropout_p,
                          philox_seed,
                          philox_offset1,
                          philox_offset2,
                          philox_seed_output,
                          philox_offset_output,
                          encoded_softmax,
                          is_causal,
                          atomic_for_causal,
                          stream_wrap,
                          extargs);
}

hipError_t
attn_fwd_compact_varlen(T4 q,            // 1 x num_heads x total_q x head_size, total_q := \sum_{i=0}^{b} s_i
                        T4 k,            // 1 x num_heads x total_k x head_size, total_k := \sum_{i=0}^{b} s_i
                        T4 v,            // 1 x num_heads x total_v x head_size, total_, := \sum_{i=0}^{b} s_i
                        T4 b,            // reserved, note this b is "bias", not "batch"
                        T1 cu_seqlens_q, // b+1, i64
                        T1 cu_seqlens_k, // b+1, i64
                        int32_t max_seqlen_q,
                        int32_t max_seqlen_k,
                        float sm_scale,
                        T2 softmax_lse,
                        T4 out, // 1 x num_heads x total_q x head_size
                        float dropout_p,
                        T0 philox_seed,
                        T0 philox_offset1,
                        int64_t philox_offset2,
                        T0 philox_seed_output,
                        T0 philox_offset_output,
                        T4 encoded_softmax,
                        bool is_causal,
                        T0 atomic_for_causal,
                        AOTRITON_NS::Stream stream_wrap,
                        FwdExtraArguments* extargs) {
  auto alibi_null_t2 = T2::get_null_tensor(q.dtype());
  return _attn_fwd_common(q,
                          k,
                          v,
                          b,
                          alibi_null_t2,
                          cu_seqlens_q,
                          cu_seqlens_k,
                          cu_seqlens_q.size(0) - 1,
                          max_seqlen_q,
                          max_seqlen_k,
                          sm_scale,
                          softmax_lse,
                          out,
                          dropout_p,
                          philox_seed,
                          philox_offset1,
                          philox_offset2,
                          philox_seed_output,
                          philox_offset_output,
                          encoded_softmax,
                          is_causal,
                          atomic_for_causal,
                          stream_wrap,
                          extargs);
}

} // AOTRITON_NS::v2::flash
