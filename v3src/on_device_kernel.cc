// Copyright © 2023-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/_internal/packed_kernel.h>
#include <aotriton/_internal/util.h>
#include <aotriton/_internal/on_device_kernel.h>
#include <aotriton/_internal/log.h>
#include <aotriton/runtime.h>
#include <filesystem>
#include <string>
#include <mutex>

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

  AOTRITON_LOG(LOG_DEBUG,
               "Trying to decompress kernel {} entry={} stem={}",
               info.flatzip_path, info.aks2_entry, info.stem_name);
  decompress_kernel(info);
  AOTRITON_LOG(LOG_DEBUG, "Decompress kernel to {}", static_cast<const void*>(essentials_.image));
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
OnDeviceKernel::decompress_kernel(const OnDeviceKernel::OnDiskKernelInfo& info) {
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
    packed_kernel_ = PackedKernel::open(info.flatzip_path, info.aks2_entry);
  }
  if (packed_kernel_) { // open may fail
    AOTRITON_LOG(LOG_DEBUG, "PackedKernel::open returns {} status: {}",
                 static_cast<const void*>(packed_kernel_.get()),
                 static_cast<int>(packed_kernel_->status()));
    essentials_ = packed_kernel_->filter(info.stem_name);
  }
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

#if AOTRITON_BUILD_FOR_TUNING
OnDeviceKernel::Essentials
OnDeviceKernel::get_image_info_iff_decompressed() const {
  return essentials_;
}
#endif

} // namespace AOTRITON_NS
