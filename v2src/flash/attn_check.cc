// Copyright © 2023-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/flash.h>
#include <aotriton/runtime.h>
#include <aotriton/util.h>
#include <flash/shim.attn_fwd.h>
#include <flash/shim.bwd_kernel_dk_dv.h>
#include <flash/shim.bwd_kernel_dq.h>
#include <flash/shim.bwd_preprocess.h>
#include <iostream>

namespace aotriton::v2::flash {

hipError_t
check_gpu(aotriton::Stream stream_wrap) {
  auto stream = stream_wrap.native();
  auto arch = getArchFromStream(stream);
  if (AttnFwdContext::get_arch_number(arch) < 0 || BwdPreprocessContext::get_arch_number(arch) < 0 ||
      BwdKernelDkDvContext::get_arch_number(arch) < 0 || BwdKernelDqContext::get_arch_number(arch) < 0) {
    return hipErrorNoBinaryForGpu;
  }
  return hipSuccess;
}

}
