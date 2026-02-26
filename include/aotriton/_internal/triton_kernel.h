// Copyright © 2023-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AOTRITON_V2_API_TRITON_KERNEL_H
#define AOTRITON_V2_API_TRITON_KERNEL_H

#include "../runtime.h"
#include "on_device_kernel.h"
#include <memory>
#include <vector>

namespace AOTRITON_NS {

struct TritonKernelCompactMeta {
  uint32_t blake2b_hi;
  uint32_t blake2b_lo;
  int16_t psel_offset;
  int16_t copt_offset;
};

// To reduce the size of compiled shared library, we decompose the kernel image
// name into a few components and only store one copy for components shared by
// multiple kernels.
//
// CAVEAT: Due to the multi-source storage scheme, it is not feasible to create
//         a unified struct to keep all information together
//
//                              KERNEL NAME COMPONENTS
//
// With
//   amd-gfx942/flash/attn_fwd/FONLY__^fp16@16,80,True,False,0,0___MI300X.aks2
//   attn_fwd-Sig-F__^fp16@16_80_True_False_0_0__P__0_2_64_64_True__CO__wave0_warp4_stg1-Gpu-MI300X.hsaco
// As an example
//   package_path: amd-gfx942/flash/attn_fwd/FONLY__^fp16@16,80,True,False,0,0___MI300X
//   kernel_name:  attn_fwd
//              (JOIN WITH -Sig-)
//              (PREFIX F__)
//   func_name:    ^fp16@16_80_True_False_0_0
//              (JOIN WITH __)
//              (PREFIX P__)
//   psel_name:    0_2_64_64_True
//              (JOIN WITH __)
//              (PREFIX CO__)
//   copt_name:    wave0_warp4_stg1
//              (JOIN WITH -Gpu-)
//   arch_name:    MI300X
// Note:
//   arch_name should be "GPU name" since we are using MI300X etc. right now.
//   But it will be gfx942 etc after dispatcher v3 refactor
//
//                          KERNEL NAME STORAGE SCHEME
//  kernel_name: stored in shim.<kernel_name>.cc
//  package_path, func_name, arch_name: stored in autotune_table_entry
//  psel_name, copt_name: consolated and stored in packed_string object.
//                        Their offsets are stored in TritonKernelCompactMeta
//                        objects.
//

class TritonKernel : public OnDeviceKernel {
public:
  TritonKernel() {
  }

  void delayed_init(uint32_t blake2b_hi,
                    uint32_t blake2b_lo,
                    const char* psel,
                    const char* copt);

  hipError_t invoke(std::string_view kernel_name,
                    pstring_view package_path,
                    std::string_view func_name,
                    std::string_view arch_name,
                    dim3 grid,
                    std::vector<void*>& args,
#if AOTRITON_BUILD_FOR_TUNING
                    bool peek_kernel_image,
#endif
                    hipStream_t stream);
  hipError_t direct_invoke(std::string_view mangled_kernel_function_name,
                           pstring_view package_path,
                           std::string_view func_name,
                           std::string_view arch_name,
                           dim3 grid,
                           dim3 block,
                           void* struct_of_args,
                           size_t sizeof_struct,
                           hipStream_t stream);


#if AOTRITON_BUILD_FOR_TUNING
  // Will not work unless invoke is called at least once, i.e., If-and-only-iF decompressed
  Essentials get_image_info_iff_decompressed() const;
#endif
private:

  uint64_t blake2b_; // TODO: sanity check of assemblied stem name
  std::string_view ksig_psel_; // psel_name component
  std::string_view ksig_copt_; // copt_name component
};

struct TritonAuxiliaryArguments {
  hipDeviceptr_t global_scratch = 0;
  hipDeviceptr_t profile_scratch = 0;
};

}

#endif
