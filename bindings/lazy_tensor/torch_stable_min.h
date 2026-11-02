// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AOTRITON_V3_BINDING_TORCH_STABLE_MIN_H
#define AOTRITON_V3_BINDING_TORCH_STABLE_MIN_H

// A minimal port of the PyTorch stable C++ API (torch/csrc/stable, 2.11):
// allocate a float32 tensor on a device, zero it, and read its data pointer
// and strides. This is all lazy_tensor needs, and it lets pyaotriton build
// without any PyTorch headers or libraries.
//
// Only functions that the stable headers themselves call are declared, and
// the StableIValue encodings follow torch::stable::empty and
// torch::stable::zero_ in torch/csrc/stable/ops.h. Keep them identical to
// those two functions. The symbols are resolved at load time from the
// libtorch_cpu that `import torch` has already loaded; see
// torch_stub/libtorch_cpu_stub.cc.

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

extern "C" {
struct AtenTensorOpaque;
struct StableListOpaque;
using AtenTensorHandle = AtenTensorOpaque*;
using StableListHandle = StableListOpaque*;
using AOTITorchError = int32_t;
using StableIValue = uint64_t;

// torch/csrc/inductor/aoti_torch/c/shim.h
int32_t aoti_torch_dtype_float32();
int32_t aoti_torch_device_type_cuda();
AOTITorchError aoti_torch_delete_tensor_object(AtenTensorHandle);
AOTITorchError aoti_torch_new_tensor_handle(AtenTensorHandle, AtenTensorHandle*);
AOTITorchError aoti_torch_get_strides(AtenTensorHandle, int64_t**);

// torch/csrc/stable/c/shim.h
AOTITorchError torch_get_mutable_data_ptr(AtenTensorHandle, void**);
AOTITorchError torch_new_list_reserve_size(size_t, StableListHandle*);
AOTITorchError torch_list_push_back(StableListHandle, StableIValue);
AOTITorchError torch_delete_list(StableListHandle);
AOTITorchError torch_call_dispatcher(const char* op_name,
                                     const char* overload_name,
                                     StableIValue* stack,
                                     uint64_t extension_build_version);
}

namespace pyaotriton::lazy_tensor::torch_stable_min {

// The libtorch version whose calling convention is used. Newer libtorch
// honors it, which is what TORCH_TARGET_VERSION does for the real headers.
constexpr uint64_t kBuildVersion = (uint64_t(2) << 56) | (uint64_t(11) << 48);

inline void check(AOTITorchError err, const char* call) {
  if (err != 0) {
    throw std::runtime_error(std::string("libtorch stable ABI call failed: ") + call);
  }
}
#define AOTRITON_TORCH_STABLE_CHECK(call) ::pyaotriton::lazy_tensor::torch_stable_min::check((call), #call)

template<typename T>
StableIValue pack(T v) {
  static_assert(sizeof(T) <= sizeof(StableIValue));
  StableIValue r = 0;
  __builtin_memcpy(&r, &v, sizeof(T));
  return r;
}

inline StableIValue pack_pointer(void* p) {
  return reinterpret_cast<StableIValue>(p);
}

inline StableIValue pack_nullopt() {
  return 0;
}

// An optional that holds a value is a pointer to a heap StableIValue.
// libtorch takes ownership of it.
inline StableIValue pack_some(StableIValue v) {
  return pack_pointer(new StableIValue(v));
}

// Owning reference to a tensor
class Tensor {
public:
  Tensor() = default;
  explicit Tensor(AtenTensorHandle handle) : handle_(handle) {}
  Tensor(Tensor&& other) noexcept : handle_(other.handle_) {
    other.handle_ = nullptr;
  }
  Tensor& operator=(Tensor&& other) noexcept {
    if (this != &other) {
      reset();
      handle_ = other.handle_;
      other.handle_ = nullptr;
    }
    return *this;
  }
  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;
  ~Tensor() { reset(); }

  void reset() {
    if (handle_) {
      aoti_torch_delete_tensor_object(handle_);
      handle_ = nullptr;
    }
  }

  AtenTensorHandle get() const { return handle_; }

  void* data_ptr() const {
    void* ptr = nullptr;
    AOTRITON_TORCH_STABLE_CHECK(torch_get_mutable_data_ptr(handle_, &ptr));
    return ptr;
  }

  int64_t stride(int dim) const {
    int64_t* strides = nullptr;
    AOTRITON_TORCH_STABLE_CHECK(aoti_torch_get_strides(handle_, &strides));
    return strides[dim];
  }

private:
  AtenTensorHandle handle_ = nullptr;
};

// torch::stable::empty(sizes, kFloat, nullopt, Device(kCUDA, device_index))
template<size_t kRank>
Tensor empty_float32(const std::array<int64_t, kRank>& sizes, int device_index) {
  StableListHandle size_list = nullptr;
  AOTRITON_TORCH_STABLE_CHECK(torch_new_list_reserve_size(kRank, &size_list));
  for (int64_t size : sizes) {
    AOTRITON_TORCH_STABLE_CHECK(torch_list_push_back(size_list, pack(size)));
  }
  // Device is packed as (type << 32) | index
  StableIValue device =
      (StableIValue(uint32_t(aoti_torch_device_type_cuda())) << 32) |
      StableIValue(uint32_t(device_index));
  std::array<StableIValue, 6> stack {
    pack_pointer(size_list),
    pack_some(pack(aoti_torch_dtype_float32())),
    pack_nullopt(),                                 // layout
    pack_some(device),
    pack_nullopt(),                                 // pin_memory
    pack_nullopt()                                  // memory_format
  };
  AOTRITON_TORCH_STABLE_CHECK(torch_call_dispatcher("aten::empty", "memory_format",
                                                    stack.data(), kBuildVersion));
  return Tensor(reinterpret_cast<AtenTensorHandle>(stack[0]));
}

// torch::stable::zero_(t)
inline void zero_(Tensor& tensor) {
  AtenTensorHandle self = nullptr;
  AOTRITON_TORCH_STABLE_CHECK(aoti_torch_new_tensor_handle(tensor.get(), &self));
  std::array<StableIValue, 1> stack { pack_pointer(self) };
  AOTRITON_TORCH_STABLE_CHECK(torch_call_dispatcher("aten::zero_", "",
                                                    stack.data(), kBuildVersion));
  // The op returns its input; release that reference
  Tensor returned(reinterpret_cast<AtenTensorHandle>(stack[0]));
}

} // namespace pyaotriton::lazy_tensor::torch_stable_min

#endif
