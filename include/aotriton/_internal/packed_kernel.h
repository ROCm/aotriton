// Copyright © 2024-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AOTRITON_V2_API_PACKED_KERNEL_H
#define AOTRITON_V2_API_PACKED_KERNEL_H

#include <aotriton/_internal/triton_kernel.h>
#include <aotriton/config.h>
#include <cstddef>
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
  static PackedKernelPtr open(pstring_view flatzip_path, std::string_view aks2_entry);
  PackedKernel(int fd, size_t offset = 0, size_t size = SIZE_MAX);
  ~PackedKernel();
  hipError_t status() const {
    return final_status_;
  }

  TritonKernel::Essentials filter(std::string_view stem_name) const;

private:
  struct PStringHash {
    using is_transparent = void;
    size_t operator()(pstring_view s) const noexcept {
      return std::hash<pstring_view>{}(s);
    }
    size_t operator()(const pstring_type& s) const noexcept {
      return std::hash<pstring_type>{}(s);
    }
  };
  struct StringHash {
    using is_transparent = void;
    size_t operator()(std::string_view s) const noexcept {
      return std::hash<std::string_view>{}(s);
    }
    size_t operator()(const std::string& s) const noexcept {
      return std::hash<std::string>{}(s);
    }
  };
  struct CachedEntry {
    uint64_t        offset;
    uint64_t        size;
    PackedKernelPtr ptr;
  };
  // InnerMap is fully populated from ZIP central directory on first open;
  // inner lookup result is ground truth (not-found means entry absent from ZIP).
  using InnerMap = std::unordered_map<std::string, CachedEntry, StringHash, std::equal_to<>>;
  static std::shared_mutex registry_mutex_;
  static std::unordered_map<pstring_type, InnerMap, PStringHash, std::equal_to<>> registry_;
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
