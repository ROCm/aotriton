// Copyright © 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AOTRITON_V2_API_PACKED_KERNEL_H
#define AOTRITON_V2_API_PACKED_KERNEL_H

#include <aotriton/config.h>
#include <aotriton/_internal/triton_kernel.h>
#include <memory>
#include <shared_mutex>
#include <stdint.h>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace AOTRITON_NS {

using PackedKernelPtr = std::shared_ptr<PackedKernel>;
struct AKS2_Metadata;

class PackedKernel {
public:
  static PackedKernelPtr open(const char* package_path);
  PackedKernel(int fd);
  ~PackedKernel();
  hipError_t status() const {
    return final_status_;
  }

  TritonKernel::Essentials filter(const char* stem_name) const;

private:
  static std::shared_mutex registry_mutex_;
  static std::unordered_map<std::string_view, PackedKernelPtr> registry_;
  std::vector<char> decompressed_content_;
  hipError_t final_status_;

  const char* kernel_start_;
  std::unordered_map<std::string_view, AKS2_Metadata> directory_;
};

};

#endif
