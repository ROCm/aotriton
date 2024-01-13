#include <aotriton/util.h>
#include <unordered_map>
#include <string>

namespace aotriton {

struct LazyArch {
    LazyArch(hipDevice_t dev) : dev_(dev) {
    }
    operator GpuArch() {
        hipDeviceProp_t prop;
        hipError_t err = hipGetDeviceProperties(&prop, dev_);
        if (err != hipSuccess)
            return GPU_ARCH_UNKNOWN;
        auto iter = string_to_arch.find(prop.gcnArchName);
        if (iter == string_to_arch.end())
            return GPU_ARCH_UNKNOWN;
        return iter->second;
    }
private:
    hipDevice_t dev_;
    static std::unordered_map<std::string, GpuArch> string_to_arch;
};

std::unordered_map<std::string, GpuArch> LazyArch::string_to_arch = {
    {"gfx90a:sramecc+:xnack-", GPU_ARCH_AMD_GFX90A},
};

GpuArch getArchFromStream(hipStream_t stream)
{
    static std::unordered_map<hipDevice_t, GpuArch> device_to_arch;
    hipDevice_t dev;
    hipError_t err = hipStreamGetDevice(stream, &dev);
    if (err != hipSuccess)
        return GPU_ARCH_UNKNOWN;
    LazyArch lazy(dev);
    device_to_arch.try_emplace(dev, lazy);
    return device_to_arch[dev];
}

template class TensorView<1>;
template class TensorView<2>;
template class TensorView<3>;
template class TensorView<4>;

} // namespace aotriton
