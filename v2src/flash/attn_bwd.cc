// Copyright © 2023-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/flash.h>
#include <aotriton/util.h>
#include <flash/shim.bwd_kernel_dk_dv.h>
#include <flash/shim.bwd_kernel_dq.h>
#include <flash/shim.bwd_preprocess.h>
#include <iostream>

namespace aotriton::v2::flash {

hipError_t
bwd_preprocess(T4 out, T4 dout, T2 delta, aotriton::Stream stream_wrap) {
  hipError_t err;
  auto stream = stream_wrap.native();
  auto arch = getArchFromStream(stream);
  auto grid_calculator = [](const BwdPreprocessParams& params) -> dim3 {
#ifndef NDEBUG
    std::cerr << "Selected Kernel "
              << " BLOCK_M = " << params.BLOCK_M << " BLOCK_N = " << params.BLOCK_N
              << " pre_load_v = " << params.pre_load_v << std::endl;
#endif
    dim3 grid {
      aotriton::cdiv<uint32_t>(params.Out->size(2), params.BLOCK_M),
      uint32_t(params.Out->size(1)),
      uint32_t(params.Out->size(0)),
    };
    // std::cerr << "Grid conf " << grid.x << " " << grid.y << " " << grid.z << std::endl;
    return grid;
  };
  int head_size = dout.size(3);
  // Requires C++ 20
  BwdPreprocessParams params = {
    .Out = &out,
    .DO = &dout,
    .Delta = &delta,
    .seqlen_q = out.size(2),
    .D_HEAD = head_size,
  };
  BwdPreprocessContext context;
  context.grid_calculator = grid_calculator;
  err = context.lookup_optimal(params, arch);
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
                 float sm_scale,
                 T4 out,
                 T4 dout,
                 T4 dk,
                 T4 dv,
                 T2 softmax_lse,
                 T2 delta,
                 float dropout_p,
                 uint64_t philox_seed,
                 uint64_t philox_offset,
                 bool is_causal,
                 aotriton::Stream stream_wrap) {
  hipError_t err;
  auto stream = stream_wrap.native();
  auto arch = getArchFromStream(stream);
  uint32_t seqlen_q = q.size(2);
  uint32_t seqlen_k = k.size(2);
  auto grid_calculator = [seqlen_k](const BwdKernelDkDvParams& params) -> dim3 {
    dim3 grid {
      aotriton::cdiv<uint32_t>(seqlen_k, params.BLOCK_N),
      uint32_t(params.Q->size(1)),
      uint32_t(params.Q->size(0)),
    };
    return grid;
  };
  int head_size = q.size(3);
  BwdKernelDkDvParams params = {
    .Q = &q,
    .K = &k,
    .V = &v,
    .Out = &out,
    .DO = &dout,
    .DK = &dk,
    .DV = &dv,
    .sm_scale = sm_scale,
    .L = &softmax_lse,
    .D = &delta,
    .seqlen_q = seqlen_q,
    .seqlen_k = seqlen_k,
    .dropout_p = dropout_p,
    .philox_seed = philox_seed,
    .philox_offset_base = static_cast<uint32_t>(philox_offset),
    .BLOCK_DMODEL = head_size,
    .CAUSAL = is_causal,
    .ENABLE_DROPOUT = dropout_p > 0.0,
  };
  BwdKernelDkDvContext context;
  context.grid_calculator = grid_calculator;
  err = context.lookup_optimal(params, arch);
  if (err != hipSuccess) {
    return err;
  }
  err = context.launch(params, stream);
  return err;
}

hipError_t
bwd_kernel_dq(T4 q,
              T4 k,
              T4 v,
              float sm_scale,
              T4 out,
              T4 dout,
              T4 dq,
              T2 softmax_lse,
              T2 delta,
              float dropout_p,
              uint64_t philox_seed,
              uint64_t philox_offset,
              bool is_causal,
              aotriton::Stream stream_wrap) {
  hipError_t err;
  auto stream = stream_wrap.native();
  auto arch = getArchFromStream(stream);
  uint32_t seqlen_q = q.size(2);
  uint32_t seqlen_k = k.size(2);
  auto grid_calculator = [seqlen_q](const BwdKernelDqParams& params) -> dim3 {
    dim3 grid {
      aotriton::cdiv<uint32_t>(seqlen_q, params.BLOCK_M),
      uint32_t(params.Q->size(1)),
      uint32_t(params.Q->size(0)),
    };
    return grid;
  };
  int head_size = q.size(3);
  BwdKernelDqParams params = {
    .Q = &q,
    .K = &k,
    .V = &v,
    .Out = &out,
    .dO = &dout,
    .dQ = &dq,
    .sm_scale = sm_scale,
    .L = &softmax_lse,
    .D = &delta,
    .seqlen_q = seqlen_q,
    .seqlen_k = seqlen_k,
    .dropout_p = dropout_p,
    .philox_seed = philox_seed,
    .philox_offset_base = static_cast<uint32_t>(philox_offset),
    .BLOCK_DMODEL = head_size,
    .CAUSAL = is_causal,
    .ENABLE_DROPOUT = dropout_p > 0.0,
  };
  BwdKernelDqContext context;
  context.grid_calculator = grid_calculator;
  err = context.lookup_optimal(params, arch);
  if (err != hipSuccess) {
    return err;
  }
  err = context.launch(params, stream);
  return err;
}

hipError_t
attn_bwd(T4 q,
         T4 k,
         T4 v,
         float sm_scale,
         T4 out,
         T4 dout,
         T4 dq,
         T4 dk,
         T4 dv,
         T2 softmax_lse,
         T2 delta,
         float dropout_p,
         uint64_t philox_seed,
         uint64_t philox_offset,
         bool is_causal,
         aotriton::Stream stream) {
  hipError_t ret;
  ret = bwd_preprocess(out, dout, delta, stream);
  if (ret != hipSuccess)
    return ret;
  ret = bwd_kernel_dk_dv(q,
                         k,
                         v,
                         sm_scale,
                         out,
                         dout,
                         dk,
                         dv,
                         softmax_lse,
                         delta,
                         dropout_p,
                         philox_seed,
                         philox_offset,
                         is_causal,
                         stream);

  if (ret != hipSuccess)
    return ret;
  ret = bwd_kernel_dq(q,
                      k,
                      v,
                      sm_scale,
                      out,
                      dout,
                      dq,
                      softmax_lse,
                      delta,
                      dropout_p,
                      philox_seed,
                      philox_offset,
                      is_causal,
                      stream);
  return ret;
}

}
