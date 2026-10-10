// Copyright © 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/config.h>
#include <aotriton/_internal/util.h>
#include <aotriton/flash.h>
#include <aotriton/util.h>
#include <flash/shim.bwd_kernel_fuse.h>
#include <flash/iface.op_attn_bwd.h>

#include "varlen.h"

namespace AOTRITON_NS::v3::flash {

// varlen.h keeps its symbols out of the user-facing namespace; this TU opts in.
using namespace internal;

dim3 BwdKernelFuseContext::grid_calculator() const {
  dim3 grid {
    AOTRITON_NS::cdiv<uint32_t>(params->max_seqlen_k, this->BLOCK_N) +
    AOTRITON_NS::cdiv<uint32_t>(params->max_seqlen_q, this->BLOCK_N) * (params->num_head_q / params->num_head_k),
    uint32_t(params->K->size(1)),
    // N. The fused kernel reads tl.num_programs(2) like the split ones, and
    // attn_bwd() has already validated that this is determined and positive.
    varlen_bwd_seq_count(params),
  };
  // std::cerr << "bwd_kernel_dk_dv grid conf " << grid.x << " " << grid.y << " " << grid.z << std::endl;
  return grid;
}

}
