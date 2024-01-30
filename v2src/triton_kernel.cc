#include <aotriton/_internal/triton_kernel.h>
#include <aotriton/runtime.h>
#include <incbin.h>
#include <iostream>
#if AOTRITON_USE_ZSTD
#include <zstd.h>
#endif

#ifdef NDEBUG
#define AOTRITON_KERNEL_VERBOSE 0
#else
#define AOTRITON_KERNEL_VERBOSE 1
#endif


#define AOTRITON_HIP_CHECK_RETURN(expr)                                 \
    do {                                                                \
        auto r = (expr);                                                \
        if (r != hipSuccess)                                            \
            throw std::runtime_error("FAILURE at Line " INCBIN_STRINGIZE(__LINE__) );   \
    } while(0)

namespace aotriton {

TritonKernel::TritonKernel(const void* image,
                           size_t image_size,
                           dim3 block,
                           int shared_memory_size)
  : kernel_image_(image), image_size_(image_size), block_(block), shared_memory_size_(shared_memory_size)
{
}

hipError_t TritonKernel::invoke(const char* kernel_name,
                                dim3 grid,
                                std::vector<void*>& args,
                                hipStream_t stream)
{
#if AOTRITON_KERNEL_VERBOSE
  std::cerr << "Invoking TritonKernel " << this << " with kernel_name = " << kernel_name << std::endl;
#endif
  if (fun_ == nullptr) {
    hipJitOption opt[] = { hipJitOptionErrorLogBufferSizeBytes,
                           hipJitOptionErrorLogBuffer,
                           hipJitOptionInfoLogBufferSizeBytes,
                           hipJitOptionInfoLogBuffer, hipJitOptionLogVerbose};
    const unsigned int errbufsize = 8192;
    const unsigned int logbufsize = 8192;
    std::vector<char> err(errbufsize, 0);
    std::vector<char> log(errbufsize, 0);
    void *optval[] = {(void *)(uintptr_t)err.size(), err.data(),
      (void *)(uintptr_t)log.size(), log.data(), (void *)(uintptr_t)1};

#if AOTRITON_USE_ZSTD
    auto image = decompress_kernel();
#if AOTRITON_KERNEL_VERBOSE
    std::cerr << "Decompress kernel from " << kernel_image_ << " with size " << image_size_ << " to " << image << " with size " << decompressed_kernel_image_.size() << std::endl;
#endif
    if (!image)
        return hipErrorInvalidImage;
#else
    auto image = kernel_image_;
#endif
    AOTRITON_HIP_CHECK_RETURN(hipModuleLoadDataEx(&mod_, image, 5, opt, optval));
    AOTRITON_HIP_CHECK_RETURN(hipModuleGetFunction(&fun_, mod_, kernel_name));
  }
  return hipModuleLaunchKernel(fun_,
                               grid.x, grid.y, grid.z,
                               block_.x, block_.y, block_.z,
                               shared_memory_size_, stream, args.data(), 0);
}

#if AOTRITON_USE_ZSTD
void TritonKernel::clear_decompressed_image()
{
    decompressed_kernel_image_.clear();
}

void* TritonKernel::decompress_kernel()
{
    if (!decompressed_kernel_image_.empty()) {
#if AOTRITON_KERNEL_VERBOSE
        std::cerr << "decompressed_kernel_image_ already initialized as " << (void*)decompressed_kernel_image_.data() << " with size: " << decompressed_kernel_image_.size() << std::endl;
#endif
        return decompressed_kernel_image_.data();
    }
    unsigned long long const decompressed_size = ZSTD_getFrameContentSize(kernel_image_, image_size_);
    if (decompressed_size == ZSTD_CONTENTSIZE_ERROR) {
#if AOTRITON_KERNEL_VERBOSE
        std::cerr << "Image not compressed by zstd" << std::endl;
#endif
        return nullptr;
    }
    if (decompressed_size == ZSTD_CONTENTSIZE_UNKNOWN) {
#if AOTRITON_KERNEL_VERBOSE
        std::cerr << "Unknown original size" << std::endl;
#endif
        return nullptr;
    }
    if (ZSTD_isError(decompressed_size))
        return nullptr;
#if AOTRITON_KERNEL_VERBOSE
    std::cerr << "decompressed_size read as " << decompressed_size << std::endl;
#endif
    decompressed_kernel_image_.resize(decompressed_size);
#if AOTRITON_KERNEL_VERBOSE
    std::cerr << "decompressed_kernel_image_ resized to " << (void*)decompressed_kernel_image_.data() << " with size: " << decompressed_kernel_image_.size() << std::endl;
#endif
    auto err = ZSTD_decompress(decompressed_kernel_image_.data(), decompressed_kernel_image_.size(),
                               kernel_image_, image_size_);
    if (ZSTD_isError(err))
        return nullptr;
    return decompressed_kernel_image_.data();
}

#endif

}
