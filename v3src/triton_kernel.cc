// Copyright © 2023-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/_internal/triton_kernel.h>
#include <aotriton/_internal/util.h>
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
  std::string stem_name;
  auto lazy = [&]() -> OnDeviceKernel::OnDiskKernelInfo {
    stem_name = construct_stem_name(kernel_name, func_name, ksig_psel_, ksig_copt_, arch_name);
    return { package_path, stem_name, kernel_name };
  };
  hipFunction_t func = get_kernel(device_id, lazy);
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
  auto lazy = [&]() -> OnDeviceKernel::OnDiskKernelInfo {
    return { package_path,
             ksig_copt_,  // Affine use ksig_psel_ as arch, ksig_copt_ as file name
             mangled_kernel_function_name };
  };
  hipFunction_t func = get_kernel(device_id, lazy);
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

#if AOTRITON_BUILD_FOR_TUNING
TritonKernel::Essentials
TritonKernel::get_image_info_iff_decompressed() const {
  return essentials_;
}
#endif

}
