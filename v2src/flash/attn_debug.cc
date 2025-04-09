// Copyright © 2024-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/config.h>
#include <aotriton/_internal/util.h>
#include <aotriton/flash.h>
#include <aotriton/util.h>
#include <flash/shim.debug_fill_dropout_rng.h>
#include <flash/shim.debug_fill_dropout_rng_tensor.h>
#include <flash/shim.debug_simulate_encoded_softmax.h>

namespace AOTRITON_NS::v2::flash {

hipError_t
debug_fill_dropout_rng_tensor(T4 r, T0 philox_seed, T0 philox_offset, AOTRITON_NS::Stream stream_wrap) {
  hipError_t err;
  auto stream = stream_wrap.native();
  auto gpu = getGpuFromStream(stream);
  auto grid_calculator = [](const DebugFillDropoutRngTensorParams& params) -> dim3 {
    dim3 grid {
      AOTRITON_NS::cdiv<uint32_t>(params.R->size(2), params.BLOCK_M),
      uint32_t(params.R->size(1)),
      uint32_t(params.R->size(0)),
    };
    // std::cerr << "Grid conf " << grid.x << " " << grid.y << " " << grid.z << std::endl;
    return grid;
  };
  int seqlen_q = r.size(2);
  int seqlen_k = r.size(3);
  DebugFillDropoutRngTensorParams params = {
    .R = &r,
    .seqlen_q = seqlen_q,
    .seqlen_k = seqlen_k,
    .philox_seed_ptr = &philox_seed,
    .philox_offset_base_ptr = &philox_offset,
  };
  DebugFillDropoutRngTensorContext context;
  context.grid_calculator = grid_calculator;
  err = context.lookup_optimal(params, gpu);
  if (err != hipSuccess) {
    return err;
  }
  err = context.launch(params, stream);
  return err;
}

hipError_t
debug_fill_dropout_rng(T4 r, uint64_t philox_seed, uint64_t philox_offset, AOTRITON_NS::Stream stream_wrap) {
  hipError_t err;
  auto stream = stream_wrap.native();
  auto gpu = getGpuFromStream(stream);
  auto grid_calculator = [](const DebugFillDropoutRngParams& params) -> dim3 {
    dim3 grid {
      AOTRITON_NS::cdiv<uint32_t>(params.R->size(2), params.BLOCK_M),
      uint32_t(params.R->size(1)),
      uint32_t(params.R->size(0)),
    };
    // std::cerr << "Grid conf " << grid.x << " " << grid.y << " " << grid.z << std::endl;
    return grid;
  };
  int seqlen_q = r.size(2);
  int seqlen_k = r.size(3);
  DebugFillDropoutRngParams params = {
    .R = &r,
    .seqlen_q = seqlen_q,
    .seqlen_k = seqlen_k,
    .philox_seed = philox_seed,
    .philox_offset = philox_offset,
  };
  DebugFillDropoutRngContext context;
  context.grid_calculator = grid_calculator;
  err = context.lookup_optimal(params, gpu);
  if (err != hipSuccess) {
    return err;
  }
  err = context.launch(params, stream);
  return err;
}

hipError_t
debug_simulate_encoded_softmax(T4 r,
                               float dropout_p,
                               T0 philox_seed,
                               T0 philox_offset1,
                               uint64_t philox_offset2,
                               AOTRITON_NS::Stream stream_wrap) {
  hipError_t err;
  auto stream = stream_wrap.native();
  auto gpu = getGpuFromStream(stream);
  auto grid_calculator = [](const DebugSimulateEncodedSoftmaxParams& params) -> dim3 {
    dim3 grid {
      AOTRITON_NS::cdiv<uint32_t>(params.R->size(2), params.BLOCK_M),
      uint32_t(params.R->size(1)),
      uint32_t(params.R->size(0)),
    };
    return grid;
  };
  int num_heads = r.size(1);
  int seqlen_q = r.size(2);
  int seqlen_k = r.size(3);
  DebugSimulateEncodedSoftmaxParams params = {
    .R = &r,
    .dropout_p = dropout_p,
    .Num_head_q = num_heads,
    .Max_seqlen_q = seqlen_q,
    .Max_seqlen_k = seqlen_k,
    .philox_seed_ptr = &philox_seed,
    .philox_offset1 = &philox_offset1,
    .philox_offset2 = philox_offset2,
  };
  DebugSimulateEncodedSoftmaxContext context;
  context.grid_calculator = grid_calculator;
  err = context.lookup_optimal(params, gpu);
  if (err != hipSuccess) {
    return err;
  }
  err = context.launch(params, stream);
  return err;
}

}
