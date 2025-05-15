// Copyright © 2023-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/config.h>
#include <aotriton/flash.h>
#include <aotriton/runtime.h>
#include <aotriton/util.h>
#include <flash/shim.attn_fwd.h>
#include <flash/shim.bwd_kernel_dk_dv.h>
#include <flash/shim.bwd_kernel_fuse.h>
#include <flash/shim.bwd_kernel_dq.h>
#include <flash/shim.bwd_preprocess.h>
#include <flash/shim.bwd_preprocess_varlen.h>
#include <iostream>

namespace AOTRITON_NS::v2::flash {

using AttnFwdContext              = AOTRITON_NS::v3::flash::AttnFwdContext;
using BwdPreprocessContext        = AOTRITON_NS::v3::flash::BwdPreprocessContext;
using BwdPreprocessVarlenContext  = AOTRITON_NS::v3::flash::BwdPreprocessVarlenContext;
using BwdKernelDkDvContext        = AOTRITON_NS::v3::flash::BwdKernelDkDvContext;
using BwdKernelDqContext          = AOTRITON_NS::v3::flash::BwdKernelDqContext;
using BwdKernelFuseContext        = AOTRITON_NS::v3::flash::BwdKernelFuseContext;

#define CHECK_FOR_KERNEL(Context)                                 \
  do {                                                            \
    auto [arch, mod] = Context::get_archmod_number(gpu);          \
    if (arch < 0)                                                 \
      return hipErrorNoBinaryForGpu;                              \
  } while(0)

hipError_t
check_gpu(AOTRITON_NS::Stream stream_wrap) {
  auto stream = stream_wrap.native();
  auto gpu = getGpuFromStream(stream);
  CHECK_FOR_KERNEL(AttnFwdContext);
  CHECK_FOR_KERNEL(BwdPreprocessContext);
  CHECK_FOR_KERNEL(BwdKernelDkDvContext);
  CHECK_FOR_KERNEL(BwdKernelDqContext);
  CHECK_FOR_KERNEL(BwdKernelFuseContext);
  return hipSuccess;
}

#undef CHECK_FOR_KERNEL

}
