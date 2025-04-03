// Copyright © 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AOTRITON_V2_API_KERNEL_CLUSTER_H
#define AOTRITON_V2_API_KERNEL_CLUSTER_H

#include <aotriton/config.h>
#include <aotriton/_internal/triton_kernel.h>

namespace AOTRITON_NS {

template<int N>
class TritonKernelCluster {
public:
  TritonKernelCluster(TritonKernelCompactMeta* meta_list,
                      const char* packed_string) {
    for (int i = 0; i < N; i++) {
      const auto& meta = meta_list[i];
      cluster_[i].delayed_init(meta.blake2b_hi,
                               meta.blake2b_lo,
                               packed_string + meta.psel_offset,
                               packed_string + meta.copt_offset);
    }
  }
  TritonKernel* get(int index) { return &cluster_[index]; }
private:
  TritonKernel cluster_[N];
};

}

#endif
