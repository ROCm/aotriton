// Copyright © 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/_internal/util.h>
#include <aotriton/flash.h>
#include <aotriton/util.h>
#include <flash/shim.debug_fill_dropout_rng.h>

namespace aotriton::v2::flash {

hipError_t
debug_fill_dropout_rng(T4 r, T0 philox_seed, T0 philox_offset, aotriton::Stream stream_wrap) {
  hipError_t err;
  auto stream = stream_wrap.native();
  auto arch = getArchFromStream(stream);
  auto grid_calculator = [](const DebugFillDropoutRngParams& params) -> dim3 {
    dim3 grid {
      aotriton::cdiv<uint32_t>(params.R->size(2), params.BLOCK_M),
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
    .philox_seed_ptr = &philox_seed,
    .philox_offset_base_ptr = &philox_offset,
  };
  DebugFillDropoutRngContext context;
  context.grid_calculator = grid_calculator;
  err = context.lookup_optimal(params, arch);
  if (err != hipSuccess) {
    return err;
  }
  err = context.launch(params, stream);
  return err;
}

}
