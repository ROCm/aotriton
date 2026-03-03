// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/_internal/aiter_hip_common.h>
#include <aotriton/_internal/util.h>
#include <aotriton/runtime.h>
#include <aotriton/util.h>

namespace AOTRITON_NS::v3::aiter {

namespace {

const std::unordered_map<uint32_t, std::string>
AITER_KERNEL_ARCH_TO_STORAGE = {
  {CAT32(GpuVendor::kAMD, 0x950), "amd-gfx950"},
  {CAT32(GpuVendor::kAMD, 0x942), "amd-gfx942"},
};

const std::unordered_map<std::string, std::string>
AITER_KERNEL_MODULE_TO_STORAGE = {
  {"fmha_v3_bwd", "flash"},
  {"fmha_v3_fwd", "flash"},
};

}

AiterAsmKernel::AiterAsmKernel(const char* name, const char* hsaco)
  : mangled_kernel_function_name_(name), hsaco_(hsaco)
{
}

AiterAsmKernel::~AiterAsmKernel() {
}

void
AiterAsmKernel::launch_kernel(const AiterAsmKernelArgs& kargs) {
  void* config[] = { HIP_LAUNCH_PARAM_BUFFER_POINTER,
                     kargs.args_ptr,
                     HIP_LAUNCH_PARAM_BUFFER_SIZE,
                     kargs.arg_size_ptr,
                     HIP_LAUNCH_PARAM_END };
  int device_id;
  AOTRITON_HIP_CHECK_RETURN(hipGetDevice(&device_id));
  std::string persistant_storage;
  auto lazy = [&]() -> OnDeviceKernel::OnDiskKernelInfo {
    return { get_package_path(kargs.stream, persistant_storage), hsaco_, mangled_kernel_function_name_ };
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

std::string_view
AiterAsmKernel::get_package_path(hipStream_t stream, std::string& persistant_storage) const {
  if (path_cache_.empty()) {
    path_cache_ = hsaco_;
  }
  auto gpu = getGpuFromStream(stream);
  auto arch = Gpu2VendorArch(gpu);
  try {
    auto aks2_arch = AITER_KERNEL_ARCH_TO_STORAGE.at(arch);
    // Example hsaco value
    //   fmha_v3_bwd/bwd_hd64_dq_convert_fp16.co
    auto it = path_cache_.begin();
    std::string aiter_module = *it;
    auto aks2_family = AITER_KERNEL_MODULE_TO_STORAGE.at(*it);
    // AKS2 path Example
    //   amd-gfx942/flash/affine_kernels/fmha_v3_bwd.aks2
    persistant_storage = aks2_arch + "/" + aks2_family + "/affine_kernels/" + aiter_module;
  } catch (std::out_of_range&) {
    // TODO: return error?
  }

  return persistant_storage;
}

std::tuple<uint64_t, std::string_view>
get_gpu_arch(hipStream_t stream) {
  auto gpu = AOTRITON_NS::getGpuFromStream(s.stream_id_);
  auto get_gpu_arch = [gpu]() -> std::string_view {
    uint32_t vendor_arch = Gpu2VendorArch(gpu);
    if (vendor_arch == CAT32(GpuVendor::kAMD, 0x950))
      return "gfx950";
    if (vendor_arch == CAT32(GpuVendor::kAMD, 0x942))
      return "gfx942";
    return "";
  };
  return {gpu, get_gpu_arch()}
}

}
