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
    params->num_seqlens == 0 ? uint32_t(params->Q->size(0)) : std::abs(params->num_seqlens),
  };
  // std::cerr << "bwd_kernel_dk_dv grid conf " << grid.x << " " << grid.y << " " << grid.z << std::endl;
  return grid;
}

}
