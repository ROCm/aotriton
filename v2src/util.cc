// Copyright © 2023-2025 Advanced Micro Devices, Inc.
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

using GpuClassifier = std::function<Gpu(const hipDeviceProp_t& prop)>;

template<Gpu GPU>
struct DummyClassifier {
  Gpu operator()(const hipDeviceProp_t& prop) { return GPU; }
};

struct LazyGpu {
  LazyGpu(hipDevice_t dev)
    : dev_(dev) {
  }
  operator Gpu() {
    hipDeviceProp_t prop;
    hipError_t err = hipGetDeviceProperties(&prop, dev_);
    if (err != hipSuccess)
      return GPU_ARCH_UNKNOWN;
    auto arch = gcnArchNameSansColon(prop.gcnArchName);
    auto iter = string_to_classifier.find(std::string(arch));
    if (iter == string_to_classifier.end())
      return GPU_ARCH_UNKNOWN;
    return iter->second(prop);
  }

private:
  hipDevice_t dev_;
  static std::unordered_map<std::string, GpuClassifier> string_to_classifier;
};

std::unordered_map<std::string, GpuClassifier> LazyGpu::string_to_classifier = {
  { "gfx90a", DummyClassifier<GPU_AMD_ARCH_GFX90A_MOD0 >() },
  { "gfx942", DummyClassifier<GPU_AMD_ARCH_GFX942_MOD0 >() },
  {"gfx1100", DummyClassifier<GPU_AMD_ARCH_GFX1100_MOD0>() },
  {"gfx1101", DummyClassifier<GPU_AMD_ARCH_GFX1101_MOD0>() },
  {"gfx1151", DummyClassifier<GPU_AMD_ARCH_GFX1151_MOD0>() },
  { "gfx950", DummyClassifier<GPU_AMD_ARCH_GFX950_MOD0 >() },
  {"gfx1201", DummyClassifier<GPU_AMD_ARCH_GFX1201_MOD0>() },
};

Gpu
getGpuFromStream(hipStream_t stream) {
  static std::unordered_map<hipDevice_t, Gpu> device_to_arch;
  hipDevice_t dev;
  hipError_t err = hipStreamGetDevice(stream, &dev);
  if (err != hipSuccess)
    return GPU_ARCH_UNKNOWN;
  LazyGpu lazy(dev);
  device_to_arch.try_emplace(dev, lazy);
  return device_to_arch[dev];
}

bool isArchExperimentallySupported(hipStream_t stream) {
  auto gpu = getGpuFromStream(stream);
  uint32_t vendor_arch = Gpu2VendorArch(gpu);
  return (vendor_arch == CAT32(GpuVendor::kAMD,  0x950) ||
          vendor_arch == CAT32(GpuVendor::kAMD, 0x1151) ||
          vendor_arch == CAT32(GpuVendor::kAMD, 0x1201));
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
