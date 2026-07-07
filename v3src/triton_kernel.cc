// Copyright © 2023-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/_internal/triton_kernel.h>
#include <aotriton/_internal/util.h>
#include <aotriton/_internal/log.h>
#include <aotriton/runtime.h>
#include <cctype>
#include <string>
#include <mutex>

namespace AOTRITON_NS {

constexpr std::string_view SEC_F    { ";;#F;" };
constexpr std::string_view SEC_P    { ";;#P;" };
constexpr std::string_view SEC_CO   { ";;#CO;" };
constexpr std::string_view SEC_ARCH { ";;arch=" };

std::string construct_stem_name(std::string_view /* kernel_name */,
                                std::string_view func_name,
                                std::string_view psel_name,
                                std::string_view copt_name,
                                std::string_view arch_name) {
  std::string ret;
  ret.reserve(SEC_F.size() + func_name.size() +
              SEC_P.size() + psel_name.size() +
              SEC_CO.size() + copt_name.size() +
              SEC_ARCH.size() + arch_name.size());
  ret += SEC_F;    ret += func_name;
  ret += SEC_P;    ret += psel_name;
  ret += SEC_CO;   ret += copt_name;
  ret += SEC_ARCH; ret += arch_name;
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
                     pstring_view flatzip_path,
                     std::string_view aks2_entry,
                     std::string_view func_name,
                     std::string_view arch_name,
                     dim3 grid,
                     std::vector<void*>& args,
#if AOTRITON_BUILD_FOR_TUNING
                     bool peek_kernel_image,
#endif
                     hipStream_t stream) {
  AOTRITON_LOG(LOG_DEBUG, "Invoking TritonKernel %p with kernel_name = \"%.*s\"",
               static_cast<const void*>(this), int(kernel_name.size()), kernel_name.data());
  int device_id;
  AOTRITON_HIP_CHECK_RETURN(hipGetDevice(&device_id));
  std::string stem_name;
  auto lazy = [&]() -> OnDeviceKernel::OnDiskKernelInfo {
    stem_name = construct_stem_name(kernel_name, func_name, ksig_psel_, ksig_copt_, arch_name);
    return { flatzip_path, aks2_entry, stem_name, kernel_name };
  };
  auto [func, essentials] = get_kernel(device_id, lazy);
#if AOTRITON_BUILD_FOR_TUNING
  if (peek_kernel_image)
    return hipSuccess;
#endif
  return hipModuleLaunchKernel(func,
                               grid.x,
                               grid.y,
                               grid.z,
                               essentials.block.x,
                               essentials.block.y,
                               essentials.block.z,
                               essentials.shared_memory_size,
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
  AOTRITON_LOG(LOG_DEBUG, "Invoking Kernel %p with kernel_name = \"%.*s\" struct_of_args = %p",
               static_cast<const void*>(this),
               int(mangled_kernel_function_name.size()), mangled_kernel_function_name.data(),
               struct_of_args);
  int device_id;
  AOTRITON_HIP_CHECK_RETURN(hipGetDevice(&device_id));
  auto lazy = [&]() -> OnDeviceKernel::OnDiskKernelInfo {
    // UNMAINTAINED: legacy package_path layout. The flatzip migration left
    // aks2_entry empty here because direct_invoke has no live callers.
    return { package_path,
             {},  // aks2_entry — legacy path predates flatzip
             ksig_copt_,  // Affine use ksig_psel_ as arch, ksig_copt_ as file name
             mangled_kernel_function_name };
  };
  auto [func, essentials] = get_kernel(device_id, lazy);
  void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                    struct_of_args,
                    HIP_LAUNCH_PARAM_BUFFER_SIZE,
                    &sizeof_struct,
                    HIP_LAUNCH_PARAM_END};
  if (log_enabled(LOG_EXTRA_DEBUG)) {
    auto hexdump = [](const void* ptr, size_t buflen) {
      const auto* buf = static_cast<const unsigned char*>(ptr);
      AOTRITON_LOG(LOG_EXTRA_DEBUG, "hexdump: %p", ptr);
      for (size_t i = 0; i < buflen; i += 16) {
        char line[80];
        int n = 0;
        for (size_t j = 0; j < 16; j++) {
          if (i+j < buflen)
            n += std::snprintf(line + n, sizeof(line) - n, "%02x ", static_cast<unsigned>(buf[i+j]));
          else
            n += std::snprintf(line + n, sizeof(line) - n, "   ");
        }
        line[n++] = ' ';
        for (size_t j = 0; j < 16 && i+j < buflen; j++) {
          unsigned char c = buf[i+j];
          line[n++] = static_cast<char>(std::isprint(c) ? c : '.');
        }
        line[n] = '\0';
        AOTRITON_LOG(LOG_EXTRA_DEBUG, "%06zx: %s", i, line);
      }
    };
    hexdump(struct_of_args, sizeof_struct);
  }
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

}
