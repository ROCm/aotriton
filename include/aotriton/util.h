#ifndef AOTRITON_V2_API_UTIL_H
#define AOTRITON_V2_API_UTIL_H

#include <stdint.h>
#include <functional>
#include <string_view>
#include "dtypes.h"

namespace aotriton {

inline constexpr uint64_t CAT(constexpr uint32_t high, 
                              constexpr uint32_t low)
{
    constexpr uint64_t high64 = high;
    constexpr uint64_t low64 = low;
    return (high64 << 32) | low64;
}

// Use PCI IDs to avoid allocating numbers by ourselves
enum GpuVendor : uint32_t {
    kAMD = 0x1002,
    kNVIDIA = 0x10de,
    kINTEL = 0x8086,
}

// More bits for potential non-PCI architectures
enum GpuArch : uint64_t {
    kGcnGfx90a = CAT(GpuVendor::kAMD, 0x90a),
};

template<int Rank>
class TensorView {
public:
    TensorView()
    {
    }

    // Use to enclose aten::Tensor
    template<typename Tensor>
    TensorView(const Tensor& tensor,
               std::function<DType(const Tensor&)> dtype_extractor) {
        base_ = tensor.data_ptr();
        for (int i = 0; i < Rank; i++) {
            sizes_ = tensor.size(i);
            strides_ = tensor.stride(i);
        }
        dtype_ = dtype_extractor(tensor);
    }

    opeartor bool() const {
        return base_ != nullptr;
    }

    uint64_t size(int i) const {
        return sizes_[i];
    }

    uint64_t stride(int i) const {
        return strides_[i];
    }

    const void* data_ptr() const {
        return base_;
    }

    DType dtype() const {
        return dtype_;
    }
private:
    const void* base_ = nullptr;
    uint64_t sizes_[Rank];
    uint64_t strides_[Rank];
    DType dtype_ = kUnknown;
};

std::string_view getArchFromStream(hipStream_t);

} // namespace aotriton

#endif
