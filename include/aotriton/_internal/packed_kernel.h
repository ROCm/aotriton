// Copyright © 2024-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AOTRITON_V2_API_PACKED_KERNEL_H
#define AOTRITON_V2_API_PACKED_KERNEL_H

#include <aotriton/_internal/triton_kernel.h>
#include <aotriton/config.h>
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
  static PackedKernelPtr open(pstring_view package_path);
  PackedKernel(int fd);
  ~PackedKernel();
  hipError_t status() const {
    return final_status_;
  }

  TritonKernel::Essentials filter(std::string_view stem_name) const;

private:
  static std::shared_mutex registry_mutex_;
  static std::unordered_map<pstring_view, PackedKernelPtr> registry_;
  // Note: do NOT drop the decompressed directory, its content is used by
  //       the unordered_map directory_
  std::vector<uint8_t> decompressed_content_;
  hipError_t final_status_;

  const uint8_t* kernel_start_;
  // Note: again, AKS2_Metadata points to directory at decompressed_content_
  std::unordered_map<std::string_view, const AKS2_Metadata*> directory_;
};

};

#endif
