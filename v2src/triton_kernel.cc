#include <aotriton/_internal/triton_kernel.h>
#include <aotriton/runtime.h>
#include <incbin.h>

#define AOTRITON_HIP_CHECK_RETURN(expr)                                 \
    do {                                                                \
        auto r = (expr);                                                \
        if (r != hipSuccess)                                            \
            throw std::runtime_error("FAILURE at Line " INCBIN_STRINGIZE(__LINE__) );   \
    } while(0)

namespace aotriton {

TritonKernel::TritonKernel(const void* image,
                           dim3 block,
                           int shared_memory_size)
  : kernel_image_(image), block_(block), shared_memory_size_(shared_memory_size)
{
}

hipError_t TritonKernel::invoke(const char* kernel_name,
                                dim3 grid,
                                std::vector<void*>& args,
                                hipStream_t stream)
{
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

    AOTRITON_HIP_CHECK_RETURN(hipModuleLoadDataEx(&mod_, kernel_image_, 5, opt, optval));
    AOTRITON_HIP_CHECK_RETURN(hipModuleGetFunction(&fun_, mod_, kernel_name));
  }
  return hipModuleLaunchKernel(fun_,
                               grid.x, grid.y, grid.z,
                               block_.x, block_.y, block_.z,
                               shared_memory_size_, stream, args.data(), 0);
}

}
