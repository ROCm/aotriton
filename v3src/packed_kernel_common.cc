// Copyright © 2024-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/_internal/packed_kernel.h>
#include <aotriton/_internal/lszip.h>
#include <aotriton/_internal/fd.h>
#include <aotriton/_internal/log.h>
#include <aotriton/runtime.h>
#include <algorithm>
#include <mutex>
#include <cstring>
#include <cassert>
#include <new>
#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#endif
#include <errno.h>
#include <filesystem>
#include <lzma.h>

namespace fs = std::filesystem;
static const std::string_view KERNEL_STORAGE_V2_BASE = "aotriton.images";
static const std::string AKS2_MAGIC = "AKS2";
constexpr int AOTRITON_LZMA_BUFSIZ = 64 * 1024;
// Upper bound on the decompressed size of a single AKS2 entry. Shipped entries
// are a few MiB; this only stops a corrupt header from requesting a multi-GiB
// allocation before any of its content has been validated.
constexpr uint64_t AOTRITON_AKS2_MAX_UNCOMPRESSED = 1ull << 30; // 1 GiB
// Memory budget for the LZMA decoder. The largest dictionary xz selects at -9
// is 64 MiB, so this leaves generous headroom while still bounding the
// allocation a crafted stream header can request.
constexpr uint64_t AOTRITON_LZMA_MEMLIMIT = 256ull << 20; // 256 MiB

namespace {

#if defined(_WIN32)
fs::path
module_path_from_address(const void* address) {
  constexpr DWORD kModulePathChars = 2 * MAX_PATH;
  HMODULE module = nullptr;
  if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                          GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                          reinterpret_cast<LPCWSTR>(address),
                          &module)) {
    return {};
  }

  // Match the previous Windows fixed-buffer shape; it has covered current
  // install paths, while the W API preserves UTF-16 contents.
  std::wstring path(kModulePathChars, L'\0');
  DWORD size = GetModuleFileNameW(module, path.data(), kModulePathChars);
  if (size == 0 || size == kModulePathChars) {
    return {};
  }
  path.resize(size);
  return fs::path(path);
}
#endif

const fs::path&
locate_aotriton_images() {
  static fs::path aotriton_images = []() {
#if defined(_WIN32)
    fs::path module_path = module_path_from_address(
      reinterpret_cast<const void*>(locate_aotriton_images));
    if (module_path.empty()) {
      return fs::path{};
    }
    AOTRITON_LOG(AOTRITON_NS::LOG_DEBUG, "Win32 locates libaotriton at: %s",
                 module_path.string().c_str());
    return module_path.parent_path() / KERNEL_STORAGE_V2_BASE;
#else
    Dl_info info;
    dladdr((void*)locate_aotriton_images, &info);
    AOTRITON_LOG(AOTRITON_NS::LOG_DEBUG, "dladdr locates libaotriton at: %s", info.dli_fname);
    return fs::path(info.dli_fname).parent_path() / KERNEL_STORAGE_V2_BASE;
#endif
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
  // Fast path: registry already has the full directory for this ZIP.
  // InnerMap is fully populated on first open, so inner lookup is ground truth.
  {
    std::shared_lock lock(registry_mutex_);
    auto outer = registry_.find(flatzip_path);
    if (outer != registry_.end()) {
      auto inner = outer->second.find(aks2_entry);
      if (inner == outer->second.end())
        return nullptr;          // entry absent from ZIP
      if (inner->second.ptr)
        return inner->second.ptr;
      // Entry exists but PackedKernel not yet constructed — fall through to slow path.
    }
  }

  const auto& storage_base = locate_aotriton_images();
  fd_t dirfd = invalid_fd();
  fd_t zipfd = invalid_fd();

  auto open_zip = [&]() {
    if (fd_is_valid(zipfd))
      return;
#if !defined(_WIN32)
    if (!fd_is_valid(dirfd))
      dirfd = ::open(storage_base.c_str(), O_RDONLY);
    if (!fd_is_valid(dirfd))
      return;
    std::string rel_path(flatzip_path);
    zipfd = ::openat(dirfd, rel_path.c_str(), O_RDONLY);
    if (fd_is_valid(dirfd)) { fd_close(dirfd); dirfd = invalid_fd(); }
#else
    if (storage_base.empty())
      return;
    fs::path full_path = storage_base / std::wstring(flatzip_path);
    zipfd = fd_open(full_path);
#endif
  };

  std::unique_lock lock(registry_mutex_);

  // Populate InnerMap for this ZIP if not yet done. Hold a single reference
  // into the map across populate + lookup to avoid repeated hashes and
  // pstring_type copies.
  auto outer_it = registry_.find(flatzip_path);
  if (outer_it == registry_.end()) {
    open_zip();
    if (!fd_is_valid(zipfd)) {
      AOTRITON_LOG(LOG_DEBUG, "PackedKernel::open: failed to open zip %s",
                   pstring_to_utf8(flatzip_path).data());
      return nullptr;
    }
    InnerMap staging_map;
    bool ok = lszip(zipfd, [&staging_map](std::string_view name, uint64_t off, uint64_t sz) {
      staging_map.try_emplace(std::string(name), CachedEntry{ off, sz, nullptr });
    });
    if (!ok) {
      // Partial directory must not be cached as authoritative.
      AOTRITON_LOG(LOG_DEBUG, "PackedKernel::open: lszip failed to fully parse %s",
                   pstring_to_utf8(flatzip_path).data());
      if (fd_is_valid(zipfd)) fd_close(zipfd);
      return nullptr;
    }
    outer_it = registry_.emplace(pstring_type(flatzip_path), std::move(staging_map)).first;
  }

  InnerMap& inner_map = outer_it->second;
  auto it = inner_map.find(aks2_entry);
  if (it == inner_map.end()) {
    // Entry not present in ZIP central directory.
    if (fd_is_valid(zipfd)) fd_close(zipfd);
    return nullptr;
  }

  if (it->second.ptr) {
    // Another thread constructed it while we waited for the lock.
    if (fd_is_valid(zipfd)) fd_close(zipfd);
    return it->second.ptr;
  }

  open_zip();
  if (!fd_is_valid(zipfd)) {
    if (fd_is_valid(dirfd)) fd_close(dirfd);
    return nullptr;
  }
  it->second.ptr = std::make_shared<PackedKernel>(zipfd, it->second.offset, it->second.size);
  fd_close(zipfd);
  if (it->second.ptr->status() != hipSuccess) {
    AOTRITON_LOG(LOG_DEBUG, "PackedKernel: AKS2 decompression failed for entry %.*s",
                 int(aks2_entry.size()), aks2_entry.data());
    it->second.ptr.reset();
    return nullptr;
  }
  return it->second.ptr;
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
PackedKernel::PackedKernel(fd_t fd, size_t offset, size_t size) {
  if (size < sizeof(AKS2_Header)) {
    final_status_ = hipErrorInvalidSource;
    return;
  }
  if (offset != 0)
    fd_seek(fd, static_cast<off_t>(offset), SEEK_SET);
  AKS2_Header header;
  auto header_read = fd_read(fd, &header, sizeof(AKS2_Header));
  if (header_read != static_cast<ssize_t>(sizeof(AKS2_Header))
      || std::string_view(header.magic, 4) != AKS2_MAGIC) {
    final_status_ = hipErrorInvalidSource; // Broken at XZ level
    return;
  }
  // Establish the header invariants the parser below relies on. The payload is
  // the directory followed by the kernel images, so uncompressed_size must
  // cover directory_size, and every directory entry costs at least
  // sizeof(AKS2_Metadata) plus the trailing '\0' of its name, which bounds
  // number_of_kernels. See python/aks2.py for the writer side.
  if (header.uncompressed_size == 0
      || header.uncompressed_size > AOTRITON_AKS2_MAX_UNCOMPRESSED
      || header.directory_size > header.uncompressed_size
      || header.number_of_kernels > header.directory_size / (sizeof(AKS2_Metadata) + 1)) {
    AOTRITON_LOG(LOG_DEBUG, "AKS2 header failed validation: uncompressed_size=%u"
                 " directory_size=%u number_of_kernels=%u",
                 unsigned(header.uncompressed_size), unsigned(header.directory_size),
                 unsigned(header.number_of_kernels));
    final_status_ = hipErrorInvalidSource;
    return;
  }
  try {
    decompressed_content_.resize(header.uncompressed_size);
  } catch (const std::bad_alloc&) {
    // Otherwise this escapes the constructor through make_shared() in open()
    // and terminates the process.
    final_status_ = hipErrorOutOfMemory;
    return;
  }
  directory_.clear();

  lzma_stream strm = LZMA_STREAM_INIT;
  lzma_ret ret = lzma_stream_decoder(&strm, AOTRITON_LZMA_MEMLIMIT, 0);
  if (ret != LZMA_OK) {
    AOTRITON_LOG(LOG_DEBUG, "lzma_stream_decoder error: %d", static_cast<int>(ret));
    final_status_ = hipErrorInvalidSource; // Broken at XZ level
    return;
  }
  // lzma_stream_decoder allocates decoder state that has to be released on
  // every exit path below; without this each AKS2 load leaks it.
  struct LzmaGuard {
    lzma_stream* stream;
    ~LzmaGuard() { lzma_end(stream); }
  } lzma_guard{ &strm };
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
  AOTRITON_LOG(LOG_DEBUG, "PackedKernel decompressed to %p",
               static_cast<const void*>(decompressed_content_.data()));
  auto reject = [this](hipError_t status) {
    decompressed_content_.clear();
    directory_.clear();
    final_status_ = status;
  };
  // A short stream would leave the tail of the buffer zero-filled and parsed as
  // if it were real directory content.
  if (strm.total_out != header.uncompressed_size) {
    AOTRITON_LOG(LOG_DEBUG, "AKS2 payload is %llu bytes, header declares %u",
                 static_cast<unsigned long long>(strm.total_out),
                 unsigned(header.uncompressed_size));
    reject(hipErrorIllegalState);
    return;
  }
  const uint8_t* const content_begin = decompressed_content_.data();
  // The directory occupies the first directory_size bytes; the images follow.
  const uint8_t* const dir_end = content_begin + header.directory_size;
  const size_t image_region_size = header.uncompressed_size - header.directory_size;
  const uint8_t* parse_ptr = content_begin;
  for (uint32_t i = 0; i < header.number_of_kernels; i++) {
    if (static_cast<size_t>(dir_end - parse_ptr) < sizeof(AKS2_Metadata)) {
      reject(hipErrorInvalidSource); // Entry header runs past the directory
      return;
    }
    auto metadata = reinterpret_cast<const AKS2_Metadata*>(parse_ptr);
    parse_ptr += sizeof(*metadata);
    // filename_length counts the trailing '\0', so the name must be non-empty,
    // fit inside the directory, and actually be NUL-terminated before it is
    // handed to std::string_view.
    if (metadata->filename_length == 0
        || static_cast<size_t>(dir_end - parse_ptr) < metadata->filename_length
        || parse_ptr[metadata->filename_length - 1] != '\0') {
      reject(hipErrorInvalidSource); // Name runs past the directory or is unterminated
      return;
    }
    // Bound the image here, so filter() can never hand an out-of-range pointer
    // to hipModuleLoadDataEx(). Written to avoid overflowing the addition.
    if (metadata->offset > image_region_size
        || metadata->image_size > image_region_size - metadata->offset) {
      reject(hipErrorInvalidSource); // Image runs past the payload
      return;
    }
    std::string_view filename(reinterpret_cast<const char*>(parse_ptr),
                              metadata->filename_length - 1);
    directory_.emplace(filename, metadata);
    AOTRITON_LOG(LOG_DEBUG, "Add kernel %u: %.*s offset: %u",
                 unsigned(i), int(filename.size()), filename.data(), unsigned(metadata->offset));
    parse_ptr += metadata->filename_length;
  }
  kernel_start_ = parse_ptr;
  AOTRITON_LOG(LOG_DEBUG, "PackedKernel.kernel_start_ = %p", static_cast<const void*>(kernel_start_));
  if (parse_ptr != dir_end) {
    // Directory size not matching: the entries did not consume it exactly.
    reject(hipErrorIllegalAddress);
    return;
  }
  AOTRITON_LOG(LOG_DEBUG, "PackedKernel.kernel_start_ sanity check passed");
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
  // offset/image_size were bounded against the payload when the directory was
  // parsed, so this stays inside decompressed_content_.
  return { kernel_start_ + meta->offset,
           meta->image_size,
           static_cast<int>(meta->shared_memory),
           dim3 { meta->number_of_threads, 1, 1 } };
}

}
