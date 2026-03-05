// Copyright © 2023-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/_internal/packed_kernel.h>
#include <aotriton/_internal/util.h>
#include <aotriton/_internal/on_device_kernel.h>
#include <aotriton/runtime.h>
#include <iostream>
#include <string>
#include <mutex>

#ifdef NDEBUG
#define AOTRITON_KERNEL_VERBOSE 0
#else
#define AOTRITON_KERNEL_VERBOSE 1
#endif

#if AOTRITON_KERNEL_VERBOSE
#include <stdio.h>
#endif

namespace AOTRITON_NS {

OnDeviceKernel::~OnDeviceKernel() {
  clear_device_kernel();
  clear_decompressed_image();
}

std::tuple<hipFunction_t, const OnDeviceKernel::Essentials&>
OnDeviceKernel::get_kernel(int device_id,
                           std::function<OnDiskKernelInfo()> lazy) {
  hipFunction_t func = nullptr;
  // Use reader lock to peek the state
  {
    std::shared_lock lock(funcache_mutex_);
    func = cfind_function(device_id);
  }

  if (!func) {
    // Use writer lock to initialize the module for device
    std::unique_lock lock(funcache_mutex_);
    // Check again, in case another waiter has initialized the device
    func = cfind_function(device_id);
    if (!func) {
      hipError_t err;
      std::tie(func, err) = load_for_device(device_id,
                                            lazy());
    }
  }
  return {func, essentials_};
}


hipFunction_t
OnDeviceKernel::cfind_function(int device_id) const {
  auto iter = funcache_.find(device_id);
  if (iter == funcache_.end())
    return nullptr;
  return iter->second.func;
}


std::tuple<hipFunction_t, hipError_t>
OnDeviceKernel::load_for_device(int device_id,
                                const OnDeviceKernel::OnDiskKernelInfo& info) {
  hipJitOption opt[] = { hipJitOptionErrorLogBufferSizeBytes,
                         hipJitOptionErrorLogBuffer,
                         hipJitOptionInfoLogBufferSizeBytes,
                         hipJitOptionInfoLogBuffer,
                         hipJitOptionLogVerbose };
  const unsigned int errbufsize = 8192;
  const unsigned int logbufsize = 8192;
  std::vector<char> err(errbufsize, 0);
  std::vector<char> log(logbufsize, 0);
  void* optval[] = { (void*)(uintptr_t)err.size(),
                     err.data(),
                     (void*)(uintptr_t)log.size(),
                     log.data(),
                     (void*)(uintptr_t)1 };

#if AOTRITON_KERNEL_VERBOSE
#if defined(_WIN32)
  std::wcerr << L"Trying to decompress kernel " << info.package_path;
  std::cerr << " " << info.stem_name << std::endl;
#else
  std::cerr << "Trying to decompress kernel " << info.package_path << " " << info.stem_name << std::endl;
#endif
#endif
  decompress_kernel(info.package_path, info.stem_name);
#if AOTRITON_KERNEL_VERBOSE
  std::cerr << "Decompress kernel to " << essentials_.image << std::endl;
#endif
  if (!essentials_.image) {
    return std::make_tuple(nullptr, hipErrorInvalidImage);
  }
  hipModule_t mod;
  hipFunction_t func;
  AOTRITON_HIP_CHECK_RETURN(hipModuleLoadDataEx(&mod, essentials_.image, 5, opt, optval));
  AOTRITON_HIP_CHECK_RETURN(hipModuleGetFunction(&func, mod, info.function_name.data()));
  funcache_.emplace(std::piecewise_construct,
                    std::forward_as_tuple(device_id),
                    std::forward_as_tuple(device_id, mod, func));
  return std::make_tuple(func, hipSuccess);
}


// kernel_loaded_ is essential. When build for tuning, it is not possible to
// tell if a kernel is loaded, or the kernel image failed to compile and thus
// does not exists from beginning by testing essentials_.image == nullptr
void
OnDeviceKernel::decompress_kernel(pstring_view package_path,
                                      std::string_view stem_name) {
  if (kernel_loaded_) {
    return ;
  }

  std::unique_lock lock(packedkernel_mutex_);
  // Check again, another thread may have updated this when this thread is
  // waiting for the lock
  if (kernel_loaded_) {
    return ;
  }
  if (!packed_kernel_) {
    packed_kernel_ = PackedKernel::open(package_path);
  }
  if (packed_kernel_) { // open may fail
#if AOTRITON_KERNEL_VERBOSE
    std::cerr << "PackedKernel::open returns " << packed_kernel_.get()
              << " status: " << packed_kernel_->status() << std::endl;
#endif
    essentials_ = packed_kernel_->filter(stem_name);
  }
  // FIXME: There should be a memory barrier here for non-X86 CPUs.
  kernel_loaded_ = true;
}


void
OnDeviceKernel::clear_device_kernel() {
  std::unique_lock lock(funcache_mutex_);
  funcache_.clear();
  kernel_loaded_ = false;
}

void
OnDeviceKernel::clear_decompressed_image() {
  std::unique_lock lock(packedkernel_mutex_);
  essentials_.image = nullptr;
  packed_kernel_.reset();
}


OnDeviceKernel::DeviceFunction::DeviceFunction(int device_id_, hipModule_t mod_, hipFunction_t func_)
  : device_id(device_id_)
  , mod(mod_)
  , func(func_) {
}

OnDeviceKernel::DeviceFunction::~DeviceFunction() {
  if (mod != nullptr) {
    (void)hipModuleUnload(mod);
  }
}

} // namespace AOTRITON_NS
