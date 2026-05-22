// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AOTRITON_V3_FLAT_ZIP_H
#define AOTRITON_V3_FLAT_ZIP_H

#include <aotriton/_internal/on_device_kernel.h>
#include <cstdint>
#include <functional>
#include <optional>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>

namespace AOTRITON_NS {

// FlatZip: reads the central directory of a STORED (uncompressed) ZIP file
// and caches entry→(offset, size) in a static registry keyed by zip path.
//
// Thread-safe: reader-writer lock, double-checked locking pattern (see on_device_kernel.cc).
class FlatZip {
public:
  struct EntryLocation {
    uint64_t offset;
    uint64_t size;
  };

  // Warm the static registry for zip_path: parse ZIP central directory and
  // cache entry→EntryLocation. No-op if already cached.
  // fd must be an open file descriptor for zip_path (caller owns it).
  static hipError_t warm(pstring_view zip_path, int fd);

  // Returns std::nullopt if the zip or entry is not found in the cache.
  static std::optional<EntryLocation> lookup(pstring_view zip_path,
                                             std::string_view entry_name);

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

  using DirectoryMap = std::unordered_map<std::string, EntryLocation>;
  using Registry     = std::unordered_map<pstring_type, DirectoryMap, PStringHash, std::equal_to<>>;

  static std::shared_mutex registry_mutex_;
  static Registry           registry_;

  static DirectoryMap parse_central_directory(int fd);
};

} // namespace AOTRITON_NS

#endif // AOTRITON_V3_FLAT_ZIP_H
