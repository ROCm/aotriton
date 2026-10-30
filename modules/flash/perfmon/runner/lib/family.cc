// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// The flash family (perfmon::Family, perfmon_abi.h), the same for every
// release: all that differs between releases is the adapter behind
// adapter_prepare() and the generated backend table.

#include "common.h"

#include <perfmon/perfmon_abi.h>

static_assert(PMON_ABI_VERSION == 3, "family.cc is written for PMON_ABI_VERSION 3");

namespace perfmon::flash {

namespace {

using namespace perfmon_flash;

// A stream for the fills of one prepare(), destroyed on every exit path.
class ScratchStream {
 public:
  ScratchStream() { hip_check(hipStreamCreate(&stream_), "hipStreamCreate"); }
  ScratchStream(const ScratchStream&) = delete;
  ScratchStream& operator=(const ScratchStream&) = delete;
  ~ScratchStream() { static_cast<void>(hipStreamDestroy(stream_)); }
  hipStream_t get() const { return stream_; }

 private:
  hipStream_t stream_ = nullptr;
};

class FlashFamily final : public Family {
 public:
  int iface_index(std::string_view name) const override {
    if (name == "op_attn_fwd") return kIfaceFwd;
    if (name == "op_attn_bwd") return kIfaceBwd;
    return -1;
  }

  std::vector<std::string> backend_names(int iface) const override {
    if (iface < 0 || iface >= kIfaceCount) {
      throw std::invalid_argument("unknown iface index " + std::to_string(iface));
    }
    const BackendNames& b = kBackendNames[iface];
    return std::vector<std::string>(b.names, b.names + b.count);
  }

  std::unique_ptr<Measurement> prepare(const perfmon::Entry& entry,
                                       std::optional<int> backend) override {
    // The core only checked the PON's syntax. A malformed flash entry is an
    // error (std::invalid_argument), not Unsupported.
    Request req;
    req.iface = entry.iface;
    req.seed = entry.seed;
    req.entry = parse_entry(std::string(entry.pon));
    if (entry.iface < 0 || entry.iface >= kIfaceCount) {
      throw std::invalid_argument("unknown iface index " + std::to_string(entry.iface));
    }
    int backend_index = kBackendUnforced;
    if (backend) {
      backend_index = *backend;
      if (backend_index < 0 || backend_index >= kBackendNames[entry.iface].count) {
        throw std::invalid_argument("backend index " + std::to_string(backend_index) +
                                    " is outside this release's table");
      }
    }
    // One scratch stream for every fill, synchronized before returning, so a
    // measurement is fully populated whichever stream launches it later.
    ScratchStream scratch;
    auto measurement = adapter_prepare(req, backend_index, scratch.get());
    hip_check(hipStreamSynchronize(scratch.get()), "hipStreamSynchronize");
    return measurement;
  }
};

}  // namespace

}  // namespace perfmon::flash

perfmon::Family& perfmon::family() {
  static perfmon::flash::FlashFamily instance;
  return instance;
}
