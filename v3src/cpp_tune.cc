// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/cpp_tune.h>

namespace AOTRITON_NS::v3 {

void KernelFineControl::ensure_initialized(size_t index) const {
  if (index < controls_.size() && !controls_[index]) {
    controls_[index] = std::make_shared<KernelControl>();
  }
}

KernelFineControl::KernelFineControl(size_t size) : controls_(size) {}

std::shared_ptr<KernelControl> KernelFineControl::operator[](size_t index) const {
  if (index >= controls_.size()) {
    return nullptr;
  }
  ensure_initialized(index);
  return controls_[index];
}

std::shared_ptr<KernelControl> KernelFineControl::at(size_t index) const {
  if (index >= controls_.size()) {
    throw std::out_of_range("KernelFineControl index out of range");
  }
  ensure_initialized(index);
  return controls_[index];
}

size_t KernelFineControl::size() const {
  return controls_.size();
}

}
