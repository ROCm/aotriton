// Copyright © 2023-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/flash.h>
#include <aotriton/util.h>
#include <aotriton/_internal/util.h>
#include <flash/shim.attn_fwd.h>
#include <iostream>

#ifdef NDEBUG
#define AOTRITON_VERBOSE 0
#else
#define AOTRITON_VERBOSE 1
#endif

namespace aotriton::v2::flash {

hipError_t
attn_fwd(T4 q,
         T4 k,
         T4 v,
         float sm_scale,
         T2 softmax_lse,
         T4 out,
         float dropout_p,
         uint64_t philox_seed,
         uint64_t philox_offset,
         T4 encoded_softmax,
         bool is_causal,
         aotriton::Stream stream_wrap) {
  hipError_t err;
  auto stream = stream_wrap.native();
  auto arch = getArchFromStream(stream);
  auto grid_calculator = [](const AttnFwdParams& params) -> dim3 {
#if AOTRITON_VERBOSE
    std::cerr << "Selected Kernel "
              << " BLOCK_M = " << params.BLOCK_M << " BLOCK_N = " << params.BLOCK_N
              << " pre_load_v = " << params.pre_load_v << std::endl;
#endif
    dim3 grid {
      aotriton::cdiv<uint32_t>(params.seqlen_q, params.BLOCK_M),
      uint32_t(params.Q->size(1)),
      uint32_t(params.Q->size(0)),
    };
#if AOTRITON_VERBOSE
    std::cerr << "Grid conf " << grid.x << " " << grid.y << " " << grid.z << std::endl;
#endif
    return grid;
  };
  int seqlen_q = q.size(2);
  int seqlen_k = k.size(2);
  int head_size = q.size(3);
  int head_dim_rounded = std::max<int>(16, aotriton::bit_ceil(head_size));
  // Requires C++ 20
  AttnFwdParams params = {
    .Q = &q,
    .K = &k,
    .V = &v,
    .Out = &out,
    .encoded_softmax = &encoded_softmax,
    .sm_scale = sm_scale,
    .M = &softmax_lse,
    .seqlen_q = seqlen_q,
    .seqlen_k = seqlen_k,
    .head_dim = static_cast<uint64_t>(head_size),
    .dropout_p = dropout_p,
    .philox_seed = philox_seed,
    .philox_offset_base = static_cast<uint32_t>(philox_offset),
    .CAUSAL = is_causal,
    .BLOCK_DMODEL = head_dim_rounded,
    .ENABLE_DROPOUT = dropout_p > 0.0,
    .RETURN_ENCODED_SOFTMAX = bool(encoded_softmax),
    .PADDED_HEAD = head_dim_rounded != head_size,
  };
  AttnFwdContext context;
  context.grid_calculator = grid_calculator;
  // .grid_calculator = grid_calculator
  err = context.lookup_optimal(params, arch);
  if (err != hipSuccess) {
    return err;
  }
  err = context.launch(params, stream);
  return err;
}

}
