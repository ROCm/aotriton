// Copyright © 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/config.h>
#include <aotriton/runtime.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
#if AOTRITON_ENABLE_SUFFIX
namespace aotriton = AOTRITON_NS;
#endif

namespace pyaotriton {
  // Simple GPU Memory managed by HIP API
  class HipMemory {
  public:
    HipMemory() {
    };
    ~HipMemory() {
      last_state_ = hipFree(memory_);
      memory_ = nullptr;
    };
    void alloc(size_t size) {
      if (memory_) {
        last_state_ = hipFree(memory_);
      }
      last_state_ = hipMalloc(&memory_, size);
      if (last_state_ == hipSuccess) {
        memory_size_ = size;
      }
    }
    void free() {
      last_state_ = hipFree(memory_);
      memory_ = nullptr;
    }
    intptr_t get_pointer() {
      return reinterpret_cast<intptr_t>(memory_);
    }
    void load_from_host(intptr_t ptr, size_t nbytes) {
      if (memory_ && nbytes <= memory_size_) {
        last_state_ = hipMemcpy(memory_, reinterpret_cast<void*>(ptr), nbytes, hipMemcpyHostToDevice);
      }
    }
    void store_to_host(intptr_t ptr, size_t nbytes) {
      if (memory_ && nbytes <= memory_size_) {
        last_state_ = hipMemcpy(reinterpret_cast<void*>(ptr), memory_, nbytes, hipMemcpyDeviceToHost);
      }
    }
    hipError_t last_state() const { return last_state_; }
  private:
    void* memory_ = nullptr;
    size_t memory_size_ = 0;
    hipError_t last_state_;
  };
  void def_hipmemory(py::module_& m) {
    py::class_<HipMemory>(m, "HipMemory")
      .def(py::init<>())
      .def("alloc", &HipMemory::alloc)
      .def("free", &HipMemory::free)
      .def("get_pointer", &HipMemory::get_pointer)
      .def("last_state", &HipMemory::last_state)
      .def("load_from_host", &HipMemory::load_from_host)
      .def("store_to_host", &HipMemory::store_to_host)
    ;
  }
}
