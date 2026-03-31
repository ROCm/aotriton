// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AOTRITON_V3_API_CPP_TUNE_H
#define AOTRITON_V3_API_CPP_TUNE_H

#include <aotriton/config.h>
#include <cstddef>
#include <stdint.h>
#include <memory>
#include <vector>
#include <stdexcept>

namespace AOTRITON_NS::v3 {

struct AOTRITON_API KernelControl {
  // Constants
  enum KernelControlBits {
    IgnoreBit = 0,
    ManualBit = 1,
    SkipBit = 2,
    QueryBit = 3,
    ExtractImageBit = 4
  };
  static constexpr uint16_t Default = 0;
#define AOTRITON_U16_FROM_BIT_ENUM(x) static constexpr uint16_t x = (1 << x ## Bit)
  AOTRITON_U16_FROM_BIT_ENUM(Ignore);
  AOTRITON_U16_FROM_BIT_ENUM(Manual);
  AOTRITON_U16_FROM_BIT_ENUM(Skip);
  AOTRITON_U16_FROM_BIT_ENUM(Query);
  AOTRITON_U16_FROM_BIT_ENUM(ExtractImage);
#undef AOTRITON_U16_FROM_BIT_ENUM

  // Control bits (input)
  uint16_t control_bits = 0;  // Flags controlling kernel behavior:
                              // - Ignore: Completely skip lookup optimal kernel
                              //           and consequently the execution
                              // - Manual: Use hsaco_index (otherwise hsaco_index is ignored)
                              // - Skip: Lookup optimal kernel but
                              //         skip the execution
                              // - Query: Query kernel metadata (does not control skip)
                              // - ExtractImage: Extract kernel binary image.
                              //                 This flag will suppress kernel
                              //                 launching.
  uint16_t hsaco_index = 0;   // Kernel index to use (only if Manual is set in control_bits)

  // Information bits (output, written by backend)
  mutable int32_t total_hsacos = -1;      // Total number of kernels (written if Query is set)
  mutable const char* kernel_psels = nullptr;  // Kernel psels string (written if Query; for Manual kernel or autotuned kernel)
  mutable const char* kernel_copts = nullptr;  // Kernel copts string (written if Query; for Manual kernel or autotuned kernel)
  mutable const void* kernel_image = nullptr;  // Kernel binary image (written if Manual & ExtractImage are set)
  mutable size_t image_size = 0;          // Size of kernel binary (written if Manual & ExtractImage are set)
};

class AOTRITON_API KernelFineControl {
private:
  mutable std::vector<std::shared_ptr<KernelControl>> controls_;
  void ensure_initialized(size_t index) const;

public:
  explicit KernelFineControl(size_t size);

  // C++ access - returns nullptr for OOB (segfaults on dereference)
  std::shared_ptr<KernelControl> operator[](size_t index) const;

  // Python access - throws exception for OOB
  std::shared_ptr<KernelControl> at(size_t index) const;

  size_t size() const;
};

}

#endif
