// Copyright © 2023-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/_internal/packed_kernel.h>
#include <aotriton/_internal/triton_kernel.h>
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

#define STRINGIFICATION(s) STRINGIFICATION_I(s)
#define STRINGIFICATION_I(s) #s

#define AOTRITON_HIP_CHECK_RETURN(expr)                                                                      \
  do {                                                                                                       \
    auto r = (expr);                                                                                         \
    if (r != hipSuccess)                                                                                     \
      throw std::runtime_error("FAILURE at Line " STRINGIFICATION(__LINE__));                                \
  } while (0)

namespace AOTRITON_NS {

constexpr std::string_view INTER_KERNEL_FUNC { "-Sig-F__" };
constexpr std::string_view INTER_FUNC_PSEL { "__P__" };
constexpr std::string_view INTER_PSEL_COPT { "__CO__" };
constexpr std::string_view INTER_COPT_ARCH { "--Arch_" };

std::string construct_stem_name(std::string_view kernel_name,
                                std::string_view func_name,
                                std::string_view psel_name,
                                std::string_view copt_name,
                                std::string_view arch_name) {
  std::string ret;
  ret.reserve(kernel_name.size() +
              INTER_KERNEL_FUNC.size() +
              func_name.size() +
              INTER_FUNC_PSEL.size() +
              psel_name.size() +
              INTER_PSEL_COPT.size() +
              copt_name.size() +
              INTER_COPT_ARCH.size() +
              arch_name.size());
  ret += kernel_name;
  ret += INTER_KERNEL_FUNC;
  ret += func_name;
  ret += INTER_FUNC_PSEL;
  ret += psel_name;
  ret += INTER_PSEL_COPT;
  ret += copt_name;
  ret += INTER_COPT_ARCH;
  ret += arch_name;
  return ret;
}

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

void TritonKernel::delayed_init(uint32_t blake2b_lo,
                                uint32_t blake2b_hi,
                                const char* psel,
                                const char* copt) {
  blake2b_ = (uint64_t) blake2b_hi << 32 | blake2b_lo;
  ksig_psel_ = psel;
  ksig_copt_ = copt;
}

hipError_t
TritonKernel::invoke(std::string_view kernel_name,
                     pstring_view package_path,
                     std::string_view func_name,
                     std::string_view arch_name,
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
      std::string stem_name = construct_stem_name(kernel_name, func_name, ksig_psel_, ksig_copt_, arch_name);
      std::tie(func, err) = load_for_device(device_id,
                                            kernel_name,
                                            stem_name,
                                            package_path);
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

hipError_t
TritonKernel::direct_invoke(std::string_view mangled_kernel_function_name,
                            pstring_view package_path,
                            std::string_view func_name,
                            std::string_view arch_name,
                            dim3 grid,
                            dim3 block,
                            void* struct_of_args,
                            size_t sizeof_struct,
                            hipStream_t stream)
{
  // TODO: Deduplication
#if AOTRITON_KERNEL_VERBOSE
  std::cerr << "Invoking Kernel " << this << " with kernel_name = \"" << mangled_kernel_function_name << '"'
            << " struct_of_args = " << struct_of_args
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
      std::tie(func, err) = load_for_device(device_id,
                                            mangled_kernel_function_name,
                                            ksig_copt_,  // Affine use ksig_psel_ as arch, ksig_copt_ as file name
                                            package_path);
    }
  }
  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                    struct_of_args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE,
                    &sizeof_struct,
                    HIP_LAUNCH_PARAM_END};
#if AOTRITON_KERNEL_VERBOSE
  auto hexdump = [](void *ptr, int buflen) {
    unsigned char *buf = (unsigned char*)ptr;
    int i, j;
    fprintf(stderr, "hexdump: %08p\n", buf);
    for (i=0; i<buflen; i+=16) {
      fprintf(stderr, "%06x: ", i);
      for (j=0; j<16; j++)
        if (i+j < buflen)
          fprintf(stderr, "%02x ", buf[i+j]);
        else
          fprintf(stderr, "   ");
      fprintf(stderr, " ");
      for (j=0; j<16; j++)
        if (i+j < buflen)
          fprintf(stderr, "%c", isprint(buf[i+j]) ? buf[i+j] : '.');
      fprintf(stderr, "\n");
    }
  };
  hexdump(struct_of_args, sizeof_struct);
#endif
  return hipModuleLaunchKernel(func,
                               grid.x,
                               grid.y,
                               grid.z,
                               block.x,
                               block.y,
                               block.z,
                               0,
                               stream,
                               nullptr,
                               reinterpret_cast<void**>(&config));
}

hipFunction_t
TritonKernel::cfind_function(int device_id) const {
  auto iter = funcache_.find(device_id);
  if (iter == funcache_.end())
    return nullptr;
  return iter->second.func;
}

std::tuple<hipFunction_t, hipError_t>
TritonKernel::load_for_device(int device_id,
                              std::string_view kernel_function_name,
                              std::string_view stem_name,
                              pstring_view package_path) {
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
#if defined(_WIN32)
  std::wcerr << L"Trying to decompress kernel " << package_path;
  std::cerr << " " << stem_name << std::endl;
#else
  std::cerr << "Trying to decompress kernel " << package_path << " " << stem_name << std::endl;
#endif
#endif
  decompress_kernel(package_path, stem_name);
#if AOTRITON_KERNEL_VERBOSE
  std::cerr << "Decompress kernel to " << essentials_.image << std::endl;
#endif
  if (!essentials_.image) {
    return std::make_tuple(nullptr, hipErrorInvalidImage);
  }
  hipModule_t mod;
  hipFunction_t func;
  AOTRITON_HIP_CHECK_RETURN(hipModuleLoadDataEx(&mod, essentials_.image, 5, opt, optval));
  AOTRITON_HIP_CHECK_RETURN(hipModuleGetFunction(&func, mod, kernel_function_name.data()));
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
TritonKernel::decompress_kernel(pstring_view package_path,
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

}
