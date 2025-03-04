// Copyright © 2023-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/_internal/packed_kernel.h>
#include <aotriton/_internal/triton_kernel.h>
#include <aotriton/runtime.h>
#include <iostream>
#include <mutex>

#ifdef NDEBUG
#define AOTRITON_KERNEL_VERBOSE 0
#else
#define AOTRITON_KERNEL_VERBOSE 1
#endif

#define STRINGIFICATION(s) STRINGIFICATION_I(s)
#define STRINGIFICATION_I(s) #s

#define AOTRITON_HIP_CHECK_RETURN(expr)                                                                      \
  do {                                                                                                       \
    auto r = (expr);                                                                                         \
    if (r != hipSuccess)                                                                                     \
      throw std::runtime_error("FAILURE at Line " STRINGIFICATION(__LINE__));                                \
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
  , stem_name_(stem_name) {
}

hipError_t
TritonKernel::invoke(const char* kernel_name,
                     dim3 grid,
                     std::vector<void*>& args,
#if AOTRITON_BUILD_FOR_TUNING
                     bool peek_kernel_image,
#endif
                     hipStream_t stream) {
#if AOTRITON_KERNEL_VERBOSE
  std::cerr << "Invoking TritonKernel " << this << " with kernel_name = \"" << kernel_name << '"'
            << std::endl;
#endif
  int device_id;
  AOTRITON_HIP_CHECK_RETURN(hipGetDevice(&device_id));
  // Use reader lock to peek the state
  hipFunction_t func = nullptr;
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
      std::tie(func, err) = load_for_device(device_id, kernel_name);
    }
  }
#if AOTRITON_BUILD_FOR_TUNING
  if (peek_kernel_image)
    return hipSuccess;
#endif
  return hipModuleLaunchKernel(func,
                               grid.x,
                               grid.y,
                               grid.z,
                               essentials_.block.x,
                               essentials_.block.y,
                               essentials_.block.z,
                               essentials_.shared_memory_size,
                               stream,
                               args.data(),
                               0);
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
  void* optval[] = { (void*)(uintptr_t)err.size(),
                     err.data(),
                     (void*)(uintptr_t)log.size(),
                     log.data(),
                     (void*)(uintptr_t)1 };

#if AOTRITON_KERNEL_VERBOSE
  std::cerr << "Trying to decompress kernel " << package_path_ << " " << stem_name_ << std::endl;
#endif
  decompress_kernel();
#if AOTRITON_KERNEL_VERBOSE
  std::cerr << "Decompress kernel to " << essentials_.image << std::endl;
#endif
  if (!essentials_.image) {
    return std::make_tuple(nullptr, hipErrorInvalidImage);
  }
  hipModule_t mod;
  hipFunction_t func;
  AOTRITON_HIP_CHECK_RETURN(hipModuleLoadDataEx(&mod, essentials_.image, 5, opt, optval));
  AOTRITON_HIP_CHECK_RETURN(hipModuleGetFunction(&func, mod, kernel_name));
  funcache_.emplace(std::piecewise_construct,
                    std::forward_as_tuple(device_id),
                    std::forward_as_tuple(device_id, mod, func));
  return std::make_tuple(func, hipSuccess);
}

void
TritonKernel::clear_decompressed_image() {
  std::unique_lock lock(packedkernel_mutex_);
  essentials_.image = nullptr;
  packed_kernel_.reset();
}

#if AOTRITON_BUILD_FOR_TUNING
TritonKernel::Essentials
TritonKernel::get_image_info_iff_decompressed() const {
  return essentials_;
}
#endif

// kernel_loaded_ is essential. When build for tuning, it is not possible to
// tell if a kernel is loaded, or the kernel image failed to compile and thus
// does not exists from beginning by testing essentials_.image == nullptr
void
TritonKernel::decompress_kernel() {
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
    packed_kernel_ = PackedKernel::open(package_path_);
  }
  if (packed_kernel_) { // open may fail
#if AOTRITON_KERNEL_VERBOSE
    std::cerr << "PackedKernel::open returns " << packed_kernel_.get()
              << " status: " << packed_kernel_->status() << std::endl;
#endif
    essentials_ = packed_kernel_->filter(stem_name_);
  }
  // FIXME: There should be a memory barrier here for non-X86 CPUs.
  kernel_loaded_ = true;
}

}
