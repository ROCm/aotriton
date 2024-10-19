// Copyright © 2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/_internal/packed_kernel.h>
#include <aotriton/runtime.h>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <filesystem>
#include <linux/limits.h> // PATH_MAX
#include <lzma.h>
#include <unistd.h>

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
    return fs::path(info.dli_fname).parent_path() / KERNEL_STORAGE_V2_BASE;
  }();
  return aotriton_images;
}

}

namespace AOTRITON_NS {

std::shared_mutex PackedKernel::registry_mutex_;
std::unordered_map<std::string_view, PackedKernelPtr> PackedKernel::registry_;

PackedKernelPtr
PackedKernel::open(const char* package_path) {
  std::string_view path_view(package_path);
  {
    // Fast path
    std::shared_lock lock(registry_mutex_);
    if (registry_.contains(package_path))
      return registry_[path_view];
  }

  // Slow path, registry doesn't contain this kernel
  std::unique_lock lock(registry_mutex_);
  // Prevent TOCTTOU b/w two locks
  if (registry_.contains(package_path))
    return registry_[path_view];
  const auto& storage_base = locate_aotriton_images();
  int dirfd = ::open(storage_base.c_str(), O_RDONLY);
  int aks2fd = ::openat(dirfd, package_path, O_RDONLY);
  auto ret = std::make_shared<PackedKernel>(aks2fd);
  close(aks2fd);
  close(dirfd);
  registry_.emplace(path_view, ret);
  return ret;
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
// N * varlen: Kernel Images
PackedKernel::PackedKernel(int fd) {
  AKS2_Header header;
  ::read(fd, &header, sizeof(AKS2_Header));
  if (header.magic != AKS2_MAGIC) {
    final_status_ = hipErrorInvalidSource; // Broken at XZ level
    return;
  }
  decompressed_content_.resize(header.uncompressed_size);

  lzma_stream strm = LZMA_STREAM_INIT;
  lzma_ret ret = lzma_stream_decoder(&strm, UINT64_MAX, LZMA_CONCATENATED);
  if (ret != LZMA_OK) {
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
      auto rbytes = read(fd, inbuf, AOTRITON_LZMA_BUFSIZ);
      if (rbytes <= 0) {
        action = LZMA_FINISH;
        break;
      }
      strm.avail_in = rbytes;
    }
    lzma_ret ret = lzma_code(&strm, action);
    if (ret != LZMA_OK && ret != LZMA_STREAM_END) {
      decompressed_content_.clear();
      final_status_ = hipErrorIllegalState; // Content not fully decompressed
      return;
    }
  }
  const uint8_t* parse_ptr = decompressed_content_.data();
  for (uint32_t i = 0; i < header.number_of_kernels; i++) {
    auto metadata = reinterpret_cast<const AKS2_Metadata*>(parse_ptr);
    parse_ptr += sizeof(*metadata);
    std::string_view filename(reinterpret_cast<const char*>(parse_ptr));
    directory_.emplace(filename, metadata);
    parse_ptr += metadata->filename_length;
  }
  kernel_start_ = parse_ptr;
  if (kernel_start_ - decompressed_content_.data() != header.directory_size) {
    decompressed_content_.clear();
    // Directory size not matching
    final_status_ = hipErrorIllegalAddress;
  }
}

PackedKernel::~PackedKernel() {
}

TritonKernel::Essentials
PackedKernel::filter(const char* stem_name) const {
  std::string_view filename(stem_name);
  auto iter = directory_.find(filename);
  if (iter == directory_.end())
    return std::make_tuple(nullptr, 0, dim3{0, 1, 1});
  auto meta = iter->second;
  return std::make_tuple(kernel_start_ + meta->offset,
                         meta->shared_memory,
                         dim3{meta->number_of_threads, 1, 1});
}

}
