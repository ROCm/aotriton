// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/_internal/aiter_hip_common.h>
#include <aotriton/_internal/util.h>
#include <aotriton/runtime.h>

namespace AOTRITON_NS {

AiterAsmKernel::AiterAsmKernel(const char* name, const char* hsaco)
  : mangled_kernel_function_name_(name), hsaco_(hsaco)
{
}

AiterAsmKernel::~AiterAsmKernel() {
}

void
AiterAsmKernelArgs::launch_kernel(const AiterAsmKernelArgs& kargs) {
  void* config[] = { HIP_LAUNCH_PARAM_BUFFER_POINTER,
                     kargs.args_ptr,
                     HIP_LAUNCH_PARAM_BUFFER_SIZE,
                     kargs.arg_size_ptr,
                     HIP_LAUNCH_PARAM_END };
  AOTRITON_HIP_CHECK_RETURN(hipGetDevice(&device_id));
  auto lazy = [&]() -> OnDeviceKernel::OnDiskKernelInfo {
    return { package_path, hsaco_, mangled_kernel_function_name_ };
  };
  auto [kernel_func, essentials] = get_kernel(device_id, lazy);

  AOTRITON_HIP_CHECK_RETURN(hipModuleLaunchKernel(kernel_func,
                                                  kargs.gdx,
                                                  kargs.gdy,
                                                  kargs.gdz,
                                                  kargs.bdx,
                                                  kargs.bdy,
                                                  kargs.bdz,
                                                  0,
                                                  kargs.stream,
                                                  nullptr,
                                                  (void**)&config));
}

}
