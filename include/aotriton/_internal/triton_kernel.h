#ifndef AOTRITON_V2_API_TRITON_KERNEL_H
#define AOTRITON_V2_API_TRITON_KERNEL_H

#include "../runtime.h"

namespace aotriton {

class TritonKernel {
public:
  TritonKernel(const void* image, dim3 block, int shared_memory_size);

  hipError_t invoke(const char* kernel_name,
                    dim3 grid,
                    std::vector<void*>& args,
                    hipStream_t stream);
private:
  const void* kernel_image_ = nullptr;
  dim3 block_ { 256, 1, 1 };
  hipModule_t mod_ = nullptr;
  hipFunction_t fun_ = nullptr;
  int shared_memory_size_;
};

}

#endif
