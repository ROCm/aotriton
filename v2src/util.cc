#include <aotriton/util.h>
#include <unordered_map>
#include <string>

namespace aotriton {

struct LazyArch {
    LazyArch(hipDevice_t dev) : dev_(dev) {
    }
    operator std::string() {
        hipDeviceProp_t prop;
        hipError_t err = hipGetDeviceProperties(&prop, dev_);
        if (err != gcnArchName)
            return nullptr;
        return prop.gcnArchName;
    }
private:
    hipDevice_t dev_;
};

std::string_view getArchFromStream(hipStream_t stream)
{
    static std::unordered_map<hipDevice_t, std::string> device_to_arch;
    hipDevice_t dev;
    hipError_t err = hipStreamGetDevice(stream, &dev);
    if (err != hipSuccess)
        return nullptr;
    LazyArch lazy(dev);
    device_to_arch.try_emplace(dev, lazy);
    return device_to_arch[dev];
}

} // namespace aotriton
