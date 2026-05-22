// Copyright © 2024-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/_internal/packed_kernel.h>
#include <aotriton/_internal/flat_zip.h>
#include <aotriton/runtime.h>
#include <mutex>
#include <cstring>
#include <cassert>
#include <dlfcn.h>
#include <errno.h>
#include <filesystem>
#include <iostream>
#include <lzma.h>

#if defined(_WIN32)
#include "packed_kernel_win32.h"
#else
#include "packed_kernel_unix.h"
#endif

#ifdef NDEBUG
#define AOTRITON_KERNEL_VERBOSE 0
#else
#define AOTRITON_KERNEL_VERBOSE 1
#endif

namespace fs = std::filesystem;
static const std::string_view KERNEL_STORAGE_V2_BASE = "aotriton.images";
static const std::string AKS2_MAGIC = "AKS2";
constexpr int AOTRITON_LZMA_BUFSIZ = 64 * 1024;

namespace {

const fs::path&
locate_aotriton_images() {
  static fs::path aotriton_images = []() {
    Dl_info info;
    dladdr((void*)locate_aotriton_images, &info);
#if AOTRITON_KERNEL_VERBOSE
    std::cerr << "dladdr locates libaotriton at: " << info.dli_fname << std::endl;
#endif
    return fs::path(info.dli_fname).parent_path() / KERNEL_STORAGE_V2_BASE;
  }();
  return aotriton_images;
}

}

namespace AOTRITON_NS {

std::shared_mutex PackedKernel::registry_mutex_;
std::unordered_map<pstring_type, PackedKernel::InnerMap,
                   PackedKernel::PStringHash, std::equal_to<>> PackedKernel::registry_;

PackedKernelPtr
PackedKernel::open(pstring_view flatzip_path, std::string_view aks2_entry) {
  // Fast path: this AKS2 entry already cached with a loaded PackedKernel.
  {
    std::shared_lock lock(registry_mutex_);
    auto outer = registry_.find(flatzip_path);
    if (outer != registry_.end()) {
      auto inner = outer->second.find(aks2_entry);
      if (inner != outer->second.end() && inner->second.ptr)
        return inner->second.ptr;
      // Fall through: entry absent or not yet constructed.
      // PackedKernel::registry_ only caches previously opened entries, not the
      // full ZIP directory — FlatZip::registry_ holds that. Do not short-circuit.
    }
  }

  const auto& storage_base = locate_aotriton_images();
  int dirfd = -1, zipfd = -1;

  // Lazy-open the ZIP file on demand; captures by reference, no-op if already open.
  auto open_zip = [&]() {
    if (zipfd != -1)
      return;
#if !defined(_WIN32)
    // Linux/POSIX: use dirfd+openat to avoid PATH_MAX overflow for long install prefixes.
    if (dirfd == -1)
      dirfd = ::open(storage_base.c_str(), O_RDONLY);
    std::string rel_path(flatzip_path);
    zipfd = ::openat(dirfd, rel_path.c_str(), O_RDONLY);
    if (dirfd != -1) { fd_close(dirfd); dirfd = -1; }
#else
    // Windows: construct full path via fs::path and open with fd_open.
    fs::path full_path = storage_base / std::wstring(flatzip_path);
    auto u8str = full_path.u8string();
    std::string utf8_path(reinterpret_cast<const char*>(u8str.data()), u8str.size());
    zipfd = fd_open(utf8_path.c_str());
#endif
  };

  std::unique_lock lock(registry_mutex_);

  // Warm the FlatZip central directory cache if this ZIP hasn't been seen yet.
  if (!registry_.contains(flatzip_path)) {
    open_zip();
    if (zipfd < 0) {
#if AOTRITON_KERNEL_VERBOSE
      std::cerr << "PackedKernel::open: failed to open zip " << std::string(flatzip_path) << std::endl;
#endif
      return nullptr;
    }
    FlatZip::warm(flatzip_path, zipfd);
    // Populate registry_ from the warm cache.
    InnerMap& inner_map = registry_[pstring_type(flatzip_path)];
    // iterate all entries returned by lookup — we seed them lazily on demand instead.
    // (No bulk iteration API on FlatZip; entries are inserted as they are opened.)
    (void)inner_map;
  }

  // Look up this entry in the FlatZip cache to get (offset, size).
  auto loc = FlatZip::lookup(flatzip_path, aks2_entry);
  if (!loc) {
    if (zipfd != -1) fd_close(zipfd);
    return nullptr;
  }

  InnerMap& inner_map = registry_[pstring_type(flatzip_path)];
  auto& entry = inner_map[std::string(aks2_entry)];
  entry.offset = loc->offset;
  entry.size   = loc->size;

  if (entry.ptr) {
    if (zipfd != -1) fd_close(zipfd);
    return entry.ptr;
  }

  open_zip();
  if (zipfd < 0) {
    if (dirfd != -1) fd_close(dirfd);
    return nullptr;
  }
  entry.ptr = std::make_shared<PackedKernel>(zipfd, entry.offset, entry.size);
  fd_close(zipfd);
  if (entry.ptr->status() != hipSuccess) {
#if AOTRITON_KERNEL_VERBOSE
    std::cerr << "PackedKernel: AKS2 decompression failed for entry "
              << std::string(aks2_entry) << std::endl;
#endif
    entry.ptr.reset();
    return nullptr;
  }
  return entry.ptr;
}

struct AKS2_Header {
  char magic[4];
  uint32_t uncompressed_size;
  uint32_t number_of_kernels;
  uint32_t directory_size;
};

struct AKS2_Metadata {
  uint32_t shared_memory;
  uint32_t number_of_threads;
  uint32_t offset;
  uint32_t image_size;
  uint32_t filename_length;
};
// AKS2 Format
// -- Uncompressed
// 4B: AKS2  (AOTriton Kernel Storage version 2)
// 4B: Total uncompressed content size
// 4B: Number of Kernels (N)
// 4B: directory size
// -- Compressed
// N * varlen: (Directory)
//     4B shared memory size
//     4B number of threads in a GPU thread block
//     4B offset (from end of the header file)
//     4B image size
//     4B file name length (M), including trailing '\0'
//     MB file name
// N * varlen: Kernel Images (TODO: alignment requirements?)
PackedKernel::PackedKernel(int fd, size_t offset, size_t size) {
  if (offset != 0)
    ::lseek(fd, static_cast<off_t>(offset), SEEK_SET);
  AKS2_Header header;
  auto header_read = fd_read(fd, &header, sizeof(AKS2_Header));
  if (header_read == sizeof(AKS2_MAGIC) && std::string_view(header.magic, 4) != AKS2_MAGIC) {
    final_status_ = hipErrorInvalidSource; // Broken at XZ level
    return;
  }
  decompressed_content_.resize(header.uncompressed_size);
  directory_.clear();

  lzma_stream strm = LZMA_STREAM_INIT;
  lzma_ret ret = lzma_stream_decoder(&strm, UINT64_MAX, 0);
  if (ret != LZMA_OK) {
#if AOTRITON_KERNEL_VERBOSE
    std::cerr << " lzma_stream_decoder error: " << ret << std::endl;
#endif
    final_status_ = hipErrorInvalidSource; // Broken at XZ level
    return;
  }
  uint8_t inbuf[AOTRITON_LZMA_BUFSIZ];
  strm.next_in = nullptr;
  strm.avail_in = 0;
  strm.next_out = (uint8_t*)decompressed_content_.data();
  strm.avail_out = decompressed_content_.size();
  lzma_action action = LZMA_RUN;
  // Track remaining bytes when size is bounded (reading AKS2 from inside a ZIP).
  size_t remaining = (size == SIZE_MAX) ? SIZE_MAX : (size - sizeof(AKS2_Header));
  while (true) {
    if (strm.avail_in == 0) {
      strm.next_in = inbuf;
      size_t to_read = std::min<size_t>(AOTRITON_LZMA_BUFSIZ, remaining);
      auto rbytes = (to_read > 0) ? fd_read(fd, inbuf, to_read) : 0;
      if (rbytes <= 0) {
        action = LZMA_FINISH;
        break;
      }
      if (remaining != SIZE_MAX)
        remaining -= static_cast<size_t>(rbytes);
      strm.avail_in = rbytes;
    }
    lzma_ret ret = lzma_code(&strm, action);
    if (ret != LZMA_OK && ret != LZMA_STREAM_END) {
      decompressed_content_.clear();
      directory_.clear();
      final_status_ = hipErrorIllegalState; // Content not fully decompressed
      return;
    }
  }
#if AOTRITON_KERNEL_VERBOSE
  std::cerr << "PackedKernel decompressed to " << (void*)decompressed_content_.data() << std::endl;
#endif
  const uint8_t* parse_ptr = decompressed_content_.data();
  for (uint32_t i = 0; i < header.number_of_kernels; i++) {
    auto metadata = reinterpret_cast<const AKS2_Metadata*>(parse_ptr);
    parse_ptr += sizeof(*metadata);
    std::string_view filename(reinterpret_cast<const char*>(parse_ptr));
    directory_.emplace(filename, metadata);
#if AOTRITON_KERNEL_VERBOSE
    std::cerr << "Add kernel " << i << ": " << filename << " offset: " << metadata->offset << std::endl;
#endif
    parse_ptr += metadata->filename_length;
  }
  kernel_start_ = parse_ptr;
#if AOTRITON_KERNEL_VERBOSE
  std::cerr << "PackedKernel.kernel_start_ = " << (void*)kernel_start_ << std::endl;
#endif
  if (kernel_start_ - decompressed_content_.data() != header.directory_size) {
    decompressed_content_.clear();
    directory_.clear();
    // Directory size not matching
    final_status_ = hipErrorIllegalAddress;
    return;
  }
#if AOTRITON_KERNEL_VERBOSE
  std::cerr << "PackedKernel.kernel_start_ sanity check passed" << std::endl;
#endif
  final_status_ = hipSuccess;
}

PackedKernel::~PackedKernel() {
}

TritonKernel::Essentials
PackedKernel::filter(std::string_view stem_name) const {
  if (status() != hipSuccess) {
    return { nullptr, 0, 0, dim3 { 0, 0, 0 } };
  }
  auto iter = directory_.find(stem_name);
  if (iter == directory_.end())
    return { nullptr, 0, 0, dim3 { 0, 1, 1 } };
  auto meta = iter->second;
  if (meta->image_size == 0) {
    // TODO: Sanity check for shared_memory
    assert(meta->shared_memory == 0);
    assert(meta->number_of_threads == 0);
    return { nullptr, 0, 0, 0 };
  }
  return { kernel_start_ + meta->offset,
           meta->image_size,
           static_cast<int>(meta->shared_memory),
           dim3 { meta->number_of_threads, 1, 1 } };
}

}