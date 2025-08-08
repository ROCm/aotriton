// Copyright © 2024-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/_internal/packed_kernel.h>
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
std::unordered_map<pstring_view, PackedKernelPtr> PackedKernel::registry_;

PackedKernelPtr
PackedKernel::open(pstring_view package_path) {
  {
    // Fast path
    std::shared_lock lock(registry_mutex_);
    if (registry_.contains(package_path))
      return registry_[package_path];
  }

  // Slow path, registry doesn't contain this kernel
  std::unique_lock lock(registry_mutex_);
  // Prevent TOCTTOU b/w two locks
  if (registry_.contains(package_path))
    return registry_[package_path];
  const auto& storage_base = locate_aotriton_images();
#if !defined(_WIN32)
#if AOTRITON_KERNEL_VERBOSE
  std::cerr << "open dir " << storage_base << std::endl;
#endif
  int dirfd = ::open(storage_base.c_str(), O_RDONLY);
  std::string rel_path(package_path);
  rel_path += ".aks2";
#if AOTRITON_KERNEL_VERBOSE
  std::cerr << "openat " << rel_path << std::endl;
#endif
  int aks2fd = ::openat(dirfd, rel_path.c_str(), O_RDONLY);
#else
  // Build the full path using filesystem::path
  fs::path full_path = storage_base / (std::wstring(package_path) + L".aks2");
  std::string utf8_path;
  auto u8str = full_path.u8string();
  utf8_path = std::string(reinterpret_cast<const char*>(u8str.data()), u8str.size());

  int aks2fd = fd_open(utf8_path.c_str());
#endif
  if (aks2fd < 0) {
#if AOTRITON_KERNEL_VERBOSE
#if defined(_WIN32)
    std::cerr << "open(\"" << utf8_path << "\")" << " failed." << std::endl;
#else
    std::cerr << "openat(\"" << storage_base << "\", \"" << rel_path << "\")"
              << " failed. errno: " << errno << std::endl;
#endif
#endif
    return nullptr;
  }
  auto ret = std::make_shared<PackedKernel>(aks2fd);
  fd_close(aks2fd);
#if !defined(_WIN32)
  fd_close(dirfd);
#endif
  if (ret->status() == hipSuccess) {
    registry_.emplace(package_path, ret);
    return ret;
  }
#if AOTRITON_KERNEL_VERBOSE
#if defined(_WIN32)
  std::wcerr << L"PackedKernel::open(" << package_path << L") failed."
            << L" Final status: " << ret->status() << std::endl;
#else
  std::cerr << "PackedKernel::open(" << package_path << ") failed."
            << " Final status " << ret->status() << std::endl;
#endif
#endif
  return nullptr;
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
PackedKernel::PackedKernel(int fd) {
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
  while (true) {
    if (strm.avail_in == 0) {
      strm.next_in = inbuf;
      auto rbytes = fd_read(fd, inbuf, AOTRITON_LZMA_BUFSIZ);
      if (rbytes <= 0) {
        action = LZMA_FINISH;
        break;
      }
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