// Copyright © 2023-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/_internal/triton_kernel.h>
#include <aotriton/_internal/packed_kernel.h>
#include <aotriton/runtime.h>
#include <incbin.h>
#include <iostream>
#include <mutex>

#ifdef NDEBUG
#define AOTRITON_KERNEL_VERBOSE 0
#else
#define AOTRITON_KERNEL_VERBOSE 1
#endif

#define AOTRITON_HIP_CHECK_RETURN(expr)                                                                      \
  do {                                                                                                       \
    auto r = (expr);                                                                                         \
    if (r != hipSuccess)                                                                                     \
      throw std::runtime_error("FAILURE at Line " INCBIN_STRINGIZE(__LINE__));                               \
  } while (0)

namespace AOTRITON_NS {

TritonKernel::DeviceFunction::DeviceFunction(int device_id_, hipModule_t mod_, hipFunction_t func_)
  : device_id(device_id_)
  , mod(mod_)
  , func(func_) {
}

TritonKernel::DeviceFunction::~DeviceFunction() {
  if (mod != nullptr) {
    (void)hipModuleUnload(mod);
  }
}


TritonKernel::TritonKernel(const char* package_path, const char* stem_name)
  : package_path_(package_path)
  , stem_name_(stem_name)
{
}

hipError_t
TritonKernel::invoke(const char* kernel_name, dim3 grid, std::vector<void*>& args, hipStream_t stream) {
#if AOTRITON_KERNEL_VERBOSE
  std::cerr << "Invoking TritonKernel " << this << " with kernel_name = " << kernel_name << std::endl;
#endif
  int device_id;
  AOTRITON_HIP_CHECK_RETURN(hipGetDevice(&device_id));
  // Use reader lock to peek the state
  hipFunction_t func = nullptr;
  {
    std::shared_lock lock(mutex_);
    func = cfind_function(device_id);
  }

  if (!func) {
    // Use writer lock to initialize the module for device
    std::unique_lock lock(mutex_);
    // Check again, in case another waiter has initialized the device
    func = cfind_function(device_id);
    if (!func) {
      hipError_t err;
      std::tie(func, err) = load_for_device(device_id, kernel_name);
    }
  }
  return hipModuleLaunchKernel(
    func, grid.x, grid.y, grid.z, block_.x, block_.y, block_.z, shared_memory_size_, stream, args.data(), 0);
}

hipFunction_t
TritonKernel::cfind_function(int device_id) const {
  auto iter = funcache_.find(device_id);
  if (iter == funcache_.end())
    return nullptr;
  return iter->second.func;
}

std::tuple<hipFunction_t, hipError_t>
TritonKernel::load_for_device(int device_id, const char* kernel_name) {
  hipJitOption opt[] = { hipJitOptionErrorLogBufferSizeBytes,
                         hipJitOptionErrorLogBuffer,
                         hipJitOptionInfoLogBufferSizeBytes,
                         hipJitOptionInfoLogBuffer,
                         hipJitOptionLogVerbose };
  const unsigned int errbufsize = 8192;
  const unsigned int logbufsize = 8192;
  std::vector<char> err(errbufsize, 0);
  std::vector<char> log(errbufsize, 0);
  void* optval[] = {
    (void*)(uintptr_t)err.size(), err.data(), (void*)(uintptr_t)log.size(), log.data(), (void*)(uintptr_t)1
  };

  std::tie(kernel_image_, shared_memory_size_, block_) = decompress_kernel();
#if AOTRITON_KERNEL_VERBOSE
  // std::cerr << "Decompress kernel from " << kernel_image_ << " with size " << image_size_ << " to " << image
  //           << " with size " << decompressed_kernel_image_.size() << std::endl;
#endif
  if (!kernel_image_) {
    return std::make_tuple(nullptr, hipErrorInvalidImage);
  }
  hipModule_t mod;
  hipFunction_t func;
  AOTRITON_HIP_CHECK_RETURN(hipModuleLoadDataEx(&mod, kernel_image_, 5, opt, optval));
  AOTRITON_HIP_CHECK_RETURN(hipModuleGetFunction(&func, mod, kernel_name));
  funcache_.emplace(std::piecewise_construct,
                    std::forward_as_tuple(device_id),
                    std::forward_as_tuple(device_id, mod, func));
  return std::make_tuple(func, hipSuccess);
}

void
TritonKernel::clear_decompressed_image() {
  std::unique_lock lock(mutex_);
  kernel_image_ = nullptr;
  packed_kernel_.reset();
}

TritonKernel::Essentials
TritonKernel::decompress_kernel() {
  {
    std::shared_lock lock(mutex_);
    if (kernel_image_) {
      return std::make_tuple(kernel_image_, shared_memory_size_, block_);
    }
  }
  std::unique_lock lock(mutex_);
  if (!packed_kernel_) {
    packed_kernel_ = PackedKernel::open(package_path_);
  }
  return packed_kernel_->filter(stem_name_);
}

}
