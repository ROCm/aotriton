#ifndef AOTRITON_V2_API__INTERNAL_STATIC_SWITCH_H
#define AOTRITON_V2_API__INTERNAL_STATIC_SWITCH_H

#define BOOL_SWITCH(COND, KNAME, ...)               \
    [&] {                                           \
        if (COND) {                                 \
            constexpr bool KNAME = true;            \
            return __VA_ARGS__();                   \
        } else {                                    \
            constexpr bool KNAME = false;           \
            return __VA_ARGS__();                   \
        }                                           \
    }

#define TENSOR_DTYPE_SWITCH(tensor, TNAME, ...)                         \
    [&] {                                                               \
        if (tensor.dtype() == DType::kFloat32) {                        \
            using TNAME = float;                                        \
            return __VA_ARGS__();                                       \
        } else if (tensor.dtype() == DType::kFloat16) {                 \
            using TNAME = __fp16;                                       \
            return __VA_ARGS__();                                       \
        } else if (tensor.dtype() == DType::kBFloat16) {                \
            using TNAME = __bf16;                                       \
            return __VA_ARGS__();                                       \
        } else if (tensor.dtype() == DType::kInt8) {                    \
            using TNAME = int8_t;                                       \
            return __VA_ARGS__();                                       \
        } else if (tensor.dtype() == DType::kInt16) {                   \
            using TNAME = int16_t;                                      \
            return __VA_ARGS__();                                       \
        } else if (tensor.dtype() == DType::kInt32) {                   \
            using TNAME = int32_t;                                      \
            return __VA_ARGS__();                                       \
        } else if (tensor.dtype() == DType::kInt64) {                   \
            using TNAME = int64_t;                                      \
            return __VA_ARGS__();                                       \
        } else if (tensor.dtype() == DType::kUInt8) {                   \
            using TNAME = uint8_t;                                      \
            return __VA_ARGS__();                                       \
        } else if (tensor.dtype() == DType::kUInt16) {                  \
            using TNAME = uint16_t;                                     \
            return __VA_ARGS__();                                       \
        } else if (tensor.dtype() == DType::kUInt32) {                  \
            using TNAME = uint32_t;                                     \
            return __VA_ARGS__();                                       \
        } else if (tensor.dtype() == DType::kUInt64) {                  \
            using TNAME = uint64_t;                                     \
            return __VA_ARGS__();                                       \
        }                                                               \
    }

#define ARCH_SWITCH(arch, KNAME, ...)                                   \
    [&] {                                                               \
        if (arch == "gfx90a:sramecc+:xnack-") {                         \
            constexpr bool KNAME = GpuArch::kGcnGfx90a;                 \
            return __VA_ARGS__();                                       \
        } else {                                                        \
            err = hipErrorInsufficientDriver;                           \
        }                                                               \
    }

#endif
