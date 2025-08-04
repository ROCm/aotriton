// Copyright © 2024-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/_internal/packed_kernel.h>
#include <aotriton/runtime.h>
#include <mutex>
#include <cstring>
#include <cassert>
#include <dlfcn.h>
#include <filesystem>
#include <iostream>
#include <lzma.h>
#include <vector>
#include <string>

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

#include <windows.h>

#if !defined(ssize_t)
    #include <BaseTsd.h> // For SSIZE_T
    typedef SSIZE_T ssize_t;
#endif

// Invalid handle value that can be returned as int
constexpr int INVALID_FD = -1;

// Helper function to convert UTF-8 string to wide string
static std::wstring utf8_to_wide(const std::string& utf8_str) {
    if (utf8_str.empty()) {
        return std::wstring();
    }
    
    // Calculate required buffer size
    int wpath_len = MultiByteToWideChar(CP_UTF8, 0, utf8_str.c_str(), -1, nullptr, 0);
    if (wpath_len == 0) {
        return std::wstring();
    }
    
    // Allocate and convert
    std::wstring wide_str(wpath_len - 1, L'\0'); // -1 because MultiByteToWideChar includes null terminator
    if (MultiByteToWideChar(CP_UTF8, 0, utf8_str.c_str(), -1, &wide_str[0], wpath_len) == 0) {
        return std::wstring();
    }
    
    return wide_str;
}

// Helper function to add long path prefix for Windows
static std::wstring add_long_path_prefix(const std::wstring& path) {
    // Regular paths - add \\?\ prefix
    return L"\\\\?\\" + path;
}

// Convert HANDLE to int for our interface
// We use intptr_t to ensure the HANDLE fits
static int handle_to_fd(HANDLE handle) {
    if (handle == INVALID_HANDLE_VALUE) {
        return INVALID_FD;
    }
    // Cast HANDLE to intptr_t then to int
    // On 64-bit Windows, HANDLE is 64-bit, but we need to maintain int interface
    // This is safe as Windows guarantees handles fit in 32 bits for compatibility
    return static_cast<int>(reinterpret_cast<intptr_t>(handle));
}

// Convert int back to HANDLE
static HANDLE fd_to_handle(int fd) {
    if (fd == INVALID_FD) {
        return INVALID_HANDLE_VALUE;
    }
    return reinterpret_cast<HANDLE>(static_cast<intptr_t>(fd));
}

static int fd_open(const char* pathname) {
    // Convert UTF-8 pathname to wide string
    std::wstring wide_path = utf8_to_wide(std::string(pathname));
    if (wide_path.empty()) {
        return INVALID_FD;
    }

    wide_path = add_long_path_prefix(wide_path);
    
    DWORD desired_access = GENERIC_READ;
    DWORD share_mode = FILE_SHARE_READ;
    DWORD creation_disposition = OPEN_EXISTING;
    DWORD file_flags = FILE_ATTRIBUTE_NORMAL;

    HANDLE file_handle = CreateFileW(
        wide_path.c_str(),
        desired_access,
        share_mode,
        nullptr,  // Default security
        creation_disposition,
        file_flags,
        nullptr   // No template file
    );
    
    if (file_handle == INVALID_HANDLE_VALUE) {
        DWORD error_code = GetLastError();
        
#if AOTRITON_KERNEL_VERBOSE
        char error_message[256] = {0};
        FormatMessageA(
            FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
            nullptr,
            error_code,
            MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
            error_message,
            sizeof(error_message),
            nullptr
        );
        // Remove trailing newline
        size_t len = strlen(error_message);
        if (len > 0 && (error_message[len-1] == '\n' || error_message[len-1] == '\r')) {
            error_message[len-1] = '\0';
        }
        std::cerr << "CreateFileW failed for: " << pathname 
                  << ", Win32 error " << error_code 
                  << ": " << error_message << std::endl;
#endif
        
        return INVALID_FD;
    }
    
    // Return the handle as an int
    return handle_to_fd(file_handle);
}

static int fd_close(int fd) {
    HANDLE file_handle = fd_to_handle(fd);
    if (file_handle == INVALID_HANDLE_VALUE) {
        return -1;
    }
    
    // Close using Win32 API
    if (!CloseHandle(file_handle)) {
        DWORD error_code = GetLastError();
        
#if AOTRITON_KERNEL_VERBOSE
        std::cerr << "CloseHandle failed with error: " << error_code << std::endl;
#endif
        
        return -1;
    }
    
    return 0;
}

static ssize_t fd_read(int fd, void *buf, size_t count) {
    if (count == 0) {
        return 0;
    }
    
    HANDLE file_handle = fd_to_handle(fd);
    if (file_handle == INVALID_HANDLE_VALUE) {
        return -1;
    }
    
    // ReadFile uses DWORD for byte count, so we need to handle large reads
    DWORD bytes_to_read;
    if (count > MAXDWORD) {
        bytes_to_read = MAXDWORD;
    } else {
        bytes_to_read = static_cast<DWORD>(count);
    }
    
    DWORD bytes_read = 0;
    
    // Read using Win32 API
    if (!ReadFile(file_handle, buf, bytes_to_read, &bytes_read, nullptr)) {
        DWORD error_code = GetLastError();
        
        // ERROR_HANDLE_EOF is not really an error
        if (error_code == ERROR_HANDLE_EOF) {
            return 0;
        }
        
#if AOTRITON_KERNEL_VERBOSE
        std::cerr << "ReadFile failed with error: " << error_code << std::endl;
#endif
        
        return -1;
    }
    
    return static_cast<ssize_t>(bytes_read);
}

namespace AOTRITON_NS {

std::shared_mutex PackedKernel::registry_mutex_;
std::unordered_map<PathStringView, PackedKernelPtr> PackedKernel::registry_;

PackedKernelPtr
PackedKernel::open(PathStringView package_path) {
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
  
  // Build the full path using filesystem::path
  fs::path full_path = storage_base / (std::wstring(package_path) + L".aks2");

  // Convert path to UTF-8 string
  // In C++20, u8string() returns std::u8string, so we need to handle that
  std::string utf8_path;
#if __cplusplus >= 202002L
  // C++20 and later: u8string() returns std::u8string
  auto u8str = full_path.u8string();
  utf8_path = std::string(reinterpret_cast<const char*>(u8str.data()), u8str.size());
#else
  // Pre-C++20: u8string() returns std::string
  utf8_path = full_path.u8string();
#endif
  
  int aks2fd = fd_open(utf8_path.c_str());
  if (aks2fd < 0) {
#if AOTRITON_KERNEL_VERBOSE
    std::cerr << "open(\"" << utf8_path << "\")" << " failed." << std::endl;
#endif
    return nullptr;
  }
  
  auto ret = std::make_shared<PackedKernel>(aks2fd);
  fd_close(aks2fd);
  if (ret->status() == hipSuccess) {
    registry_.emplace(package_path, ret);
    return ret;
  }
#if AOTRITON_KERNEL_VERBOSE
  std::wcerr << L"PackedKernel::open(" << package_path << L") failed."
            << L" Final status: " << hipGetErrorString(ret->status()) << std::endl;
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
           int(meta->shared_memory),
           dim3 { meta->number_of_threads, 1, 1 } };
}

}