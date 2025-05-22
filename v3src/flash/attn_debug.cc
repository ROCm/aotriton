// Copyright © 2024-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/config.h>
#include <aotriton/_internal/util.h>
#include <aotriton/flash.h>
#include <aotriton/util.h>
#include <flash/shim.debug_simulate_encoded_softmax.h>
#include <flash/iface.op_attn_fwd.h>

namespace AOTRITON_NS::v3::flash {

dim3 DebugSimulateEncodedSoftmaxContext::grid_calculator() const {
  dim3 grid {
    AOTRITON_NS::cdiv<uint32_t>(params->encoded_softmax->size(2), BLOCK_M),
    uint32_t(params->encoded_softmax->size(1)),
    uint32_t(params->encoded_softmax->size(0)),
  };
  return grid;
}

}

namespace AOTRITON_NS::v2::flash {

using DebugSimulateEncodedSoftmaxParams = AOTRITON_NS::v3::flash::OpAttnFwdParams;
using DebugSimulateEncodedSoftmaxContext = AOTRITON_NS::v3::flash::DebugSimulateEncodedSoftmaxContext;

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
  int num_heads = r.size(1);
  int seqlen_q = r.size(2);
  int seqlen_k = r.size(3);
  DebugSimulateEncodedSoftmaxParams params = {
    .Num_head_q = num_heads,
    .Max_seqlen_q = seqlen_q,
    .Max_seqlen_k = seqlen_k,
    .dropout_p = dropout_p,
    .philox_seed_ptr = &philox_seed,
    .philox_offset1 = &philox_offset1,
    .philox_offset2 = philox_offset2,
    .encoded_softmax = &r,
  };
  DebugSimulateEncodedSoftmaxContext context;
  context.params = &params;
  err = context.lookup_optimal(gpu);
  if (err != hipSuccess) {
    return err;
  }
  err = context.launch(stream);
  return err;
}

}
