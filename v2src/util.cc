// Copyright © 2023-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/util.h>
#include <string>
#include <unordered_map>
#include <string_view>

namespace AOTRITON_NS {

std::string_view gcnArchNameSansColon(const char* gcnArchName) {
    std::string_view arch(gcnArchName);
    const auto colon = arch.find(':');
    if (colon != arch.npos) {
      arch = std::string_view(gcnArchName, colon);
    }
    return arch;
}

struct LazyArch {
  LazyArch(hipDevice_t dev)
    : dev_(dev) {
  }
  operator GpuArch() {
    hipDeviceProp_t prop;
    hipError_t err = hipGetDeviceProperties(&prop, dev_);
    if (err != hipSuccess)
      return GPU_ARCH_UNKNOWN;
    auto arch = gcnArchNameSansColon(prop.gcnArchName);
    auto iter = string_to_arch.find(std::string(arch));
    if (iter == string_to_arch.end())
      return GPU_ARCH_UNKNOWN;
    return iter->second;
  }

private:
  hipDevice_t dev_;
  static std::unordered_map<std::string, GpuArch> string_to_arch;
};

std::unordered_map<std::string, GpuArch> LazyArch::string_to_arch = {
  {"gfx90a", GPU_ARCH_AMD_GFX90A},
  {"gfx942", GPU_ARCH_AMD_GFX942},
  {"gfx1100", GPU_ARCH_AMD_GFX1100},
  {"gfx1101", GPU_ARCH_AMD_GFX1101},
  {"gfx950", GPU_ARCH_AMD_GFX950},
  {"gfx1201", GPU_ARCH_AMD_GFX1201},
};

GpuArch
getArchFromStream(hipStream_t stream) {
  static std::unordered_map<hipDevice_t, GpuArch> device_to_arch;
  hipDevice_t dev;
  hipError_t err = hipStreamGetDevice(stream, &dev);
  if (err != hipSuccess)
    return GPU_ARCH_UNKNOWN;
  LazyArch lazy(dev);
  device_to_arch.try_emplace(dev, lazy);
  return device_to_arch[dev];
}

bool isArchExperimentallySupported(const hipDeviceProp_t* dprops) {
  auto arch = gcnArchNameSansColon(dprops->gcnArchName);
  return (arch == "gfx950" || arch == "gfx1201");
}

int getMultiProcessorCount(hipStream_t stream) {
  static std::unordered_map<hipDevice_t, int> device_to_CUs;
  hipDevice_t dev;
  hipError_t err = hipStreamGetDevice(stream, &dev);
  if (err != hipSuccess)
    return 40;  // A guessed number

  auto iter = device_to_CUs.find(dev);
  if (iter == device_to_CUs.end()) {
    hipDeviceProp_t prop;
    err = hipGetDeviceProperties(&prop, dev);
    if (err != hipSuccess)
      return 40;  // A guessed number
    device_to_CUs[dev] = prop.multiProcessorCount;
    return prop.multiProcessorCount;
  }
  return iter->second;
}

template class TensorView<1>;
template class TensorView<2>;
template class TensorView<3>;
template class TensorView<4>;

} // namespace AOTRITON_NS
