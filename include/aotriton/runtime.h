// Copyright © 2023-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AOTRITON_V2_API_RUNTIME_H
#define AOTRITON_V2_API_RUNTIME_H

#include <hip/hip_runtime.h>

namespace aotriton {

// This is not a class for stream management (at least for now), but a way to
// make sure AOTriton APIs can have python bindings with pybind11
template<typename DeviceStreamType>
class StreamTemplate {
public:
  StreamTemplate()
    : stream_(nullptr) {
  }
  StreamTemplate(DeviceStreamType stream)
    : stream_(stream) {
  }
  DeviceStreamType native() const {
    return stream_;
  }

private:
  DeviceStreamType stream_;
};


//specialization
template<>
class StreamTemplate<hipStream_t> {
public:
  StreamTemplate()
    : stream_(nullptr) {
  }
  StreamTemplate(intptr_t valptr): stream_(reinterpret_cast<hipStream_t>(valptr)){}
  StreamTemplate(hipStream_t stream)
    : stream_(stream) {
  }
  hipStream_t native() const {
    return stream_;
  }

private:
  hipStream_t stream_;
};



using Stream = StreamTemplate<hipStream_t>;

}

#endif
