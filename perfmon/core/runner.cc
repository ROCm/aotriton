// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "perfmon_abi.h"

#include "config.h"
#include "core_info.h"
#include "pon.h"
#include "protocol.h"
#include "stats.h"
#include "thermal.h"
#include "timing.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace perfmon {

namespace {

void hip_check(hipError_t err, const char* what) {
  if (err != hipSuccess) {
    throw std::runtime_error(std::string(what) + " failed: " + hipGetErrorString(err));
  }
}

// The PCI device id (e.g. 0x74a1) from sysfs. HIP's pciDeviceID is the device
// number inside the BDF, not this. Returns -1 when unreadable.
long pci_product_id(const std::string& bdf) {
  std::ifstream f("/sys/bus/pci/devices/" + bdf + "/device");
  std::string text;
  if (!(f >> text)) return -1;
  try {
    return std::stol(text, nullptr, 16);
  } catch (const std::exception&) {
    return -1;
  }
}

class Runner {
 public:
  Runner(Family& family, int abi_version) : family_(family) {
    arches_ = split_arch_list(core_arches());
    try {
      startup(abi_version);
    } catch (const std::exception& e) {
      refusal_ = e.what();
    }
  }

  ~Runner() {
    if (stream_) static_cast<void>(hipStreamDestroy(stream_));
  }

  Reply platform(const std::vector<std::string>& args) {
    require_ready();
    if (!args.empty()) throw std::runtime_error("platform takes no arguments");
    std::ostringstream o;
    o << "{\"pmon_abi_version\":" << PMON_ABI_VERSION
      << ",\"gcn_arch\":" << json_string(bare_arch(prop_.gcnArchName))
      << ",\"vram_total_bytes\":" << static_cast<unsigned long long>(prop_.totalGlobalMem)
      << ",\"pci_device_id\":";
    if (pci_id_ >= 0) o << pci_id_; else o << "null";
    o << ",\"pci_bdf\":" << json_string(bdf_)
      << ",\"core_arches\":" << json_string_array(arches_)
      << ",\"harness\":" << json_string(core_commit())
      << ",\"thermal_gate\":" << (gate_ ? "true" : "false") << "}";
    return Reply::ok(o.str());
  }

  Reply enumerate_cmd(const std::vector<std::string>& args) {
    require_ready();
    if (args.size() != 2) {
      throw std::runtime_error("usage: enumerate <entry_pon> <iface>");
    }
    check_pon(args[0]);  // reject a malformed entry even here
    return Reply::ok("{\"backends\":" + json_string_array(backends(iface_index(args[1]))) + "}");
  }

  Reply measure(const std::vector<std::string>& args) {
    require_ready();
    if (args.size() != 3) {
      throw std::runtime_error("usage: measure <entry_pon> <iface> <backend>");
    }
    const std::string& pon = args[0];
    const std::string& iface = args[1];
    const std::string& backend = args[2];

    check_pon(pon);
    Entry entry{iface_index(iface), entry_seed(pon, iface), pon};

    std::optional<int> backend_index;  // nullopt: the unforced operator
    if (backend != "-") {
      const auto names = backends(entry.iface);
      auto it = std::find(names.begin(), names.end(), backend);
      if (it == names.end()) {
        throw std::runtime_error("backend '" + backend + "' is not one of " + iface +
                                 "'s backends " + json_string_array(names));
      }
      backend_index = static_cast<int>(it - names.begin());
    }

    std::unique_ptr<Measurement> measurement;
    try {
      measurement = family_.prepare(entry, backend_index);
    } catch (const Unsupported& e) {
      return Reply::unavailable(e.what());
    }
    if (!measurement) throw std::runtime_error("prepare returned no measurement");

    // One untimed call first. For a forced backend its error is how a backend
    // that cannot run this entry on this arch shows itself, since no release
    // can be asked which backends apply. The unforced operator has no such
    // excuse: failing here is a regression, so it is an error.
    try {
      measurement->launch(stream_);
    } catch (const std::exception& e) {
      if (!backend_index) {
        throw std::runtime_error(std::string("the unforced operator failed its untimed launch: ") +
                                 e.what());
      }
      return Reply::unavailable(std::string("launch failed: ") + e.what());
    }
    // A failure here is an asynchronous GPU error; the context may be gone,
    // so it is an error, not an unavailable backend.
    hip_check(hipStreamSynchronize(stream_), "hipStreamSynchronize after the untimed launch");

    // The temperature is read right before the timed loop, so it describes
    // the conditions of the measurement.
    if (gate_) wait_until_cool(kThermalLimitC, std::cout);
    std::string temp = "null";
    if (gate_) {
      temp = json_double(read_temperature_c());
    } else if (thermal_available()) {
      try {
        temp = json_double(read_temperature_c());
      } catch (const std::exception&) {
        // Informational only while the gate is off.
      }
    }

    TimingResult t = time_launches(*measurement, stream_);
    Stats s = compute_stats(t.samples_us);
    std::ostringstream o;
    o << "{\"timing_method\":" << json_string(t.method)
      << ",\"l2_flush\":true"
      << ",\"n\":" << s.n
      << ",\"median\":" << json_double(s.median)
      << ",\"mean\":" << json_double(s.mean)
      << ",\"stddev\":" << json_double(s.stddev)
      << ",\"min\":" << json_double(s.min)
      << ",\"p05\":" << json_double(s.p05)
      << ",\"p95\":" << json_double(s.p95)
      << ",\"seed\":" << entry.seed
      << ",\"thermal\":{\"gate\":" << (gate_ ? "true" : "false")
      << ",\"limit_c\":" << json_double(kThermalLimitC)
      << ",\"temp_c\":" << temp << "}}";
    return Reply::ok(o.str());
  }

 private:
  void startup(int abi_version) {
    if (abi_version != PMON_ABI_VERSION) {
      throw std::runtime_error("the adapter was built for PMON_ABI_VERSION " +
                               std::to_string(abi_version) + ", the core for " +
                               std::to_string(PMON_ABI_VERSION));
    }
    // Device 0 of whatever HIP_VISIBLE_DEVICES leaves visible: the launcher
    // masks the process to one GPU.
    hip_check(hipGetDeviceProperties(&prop_, 0), "hipGetDeviceProperties");
    const std::string why = check_arch(prop_.gcnArchName, arches_);
    if (!why.empty()) throw std::runtime_error("refusing to measure: " + why);

    char bdf[64] = {0};
    hip_check(hipDeviceGetPCIBusId(bdf, sizeof(bdf), 0), "hipDeviceGetPCIBusId");
    bdf_ = bdf;
    std::transform(bdf_.begin(), bdf_.end(), bdf_.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    pci_id_ = pci_product_id(bdf_);

    gate_ = parse_thermal_gate(std::getenv("AOTRITON_THERMAL_GATE"));
    if (gate_) {
      // Fail now rather than measure ungated while claiming to be gated.
      read_temperature_c();
    }
    // A blocking stream, as verified on hardware: work an adapter wrongly puts
    // on the null stream then fails the capture instead of racing it.
    hip_check(hipStreamCreate(&stream_), "hipStreamCreate");
  }

  void require_ready() const {
    if (!refusal_.empty()) throw std::runtime_error(refusal_);
  }

  int32_t iface_index(const std::string& name) const {
    const int i = family_.iface_index(name);
    if (i < 0) throw std::runtime_error("unknown iface '" + name + "'");
    return i;
  }

  std::vector<std::string> backends(int32_t iface) const {
    return family_.backend_names(iface);
  }

  // Syntax only: the core knows no family's keys, the adapter decodes them.
  static void check_pon(const std::string& pon) {
    const std::string bad = Pon(pon).check();
    if (!bad.empty()) throw std::runtime_error("malformed entry: " + bad);
  }

  Family& family_;
  std::vector<std::string> arches_;
  std::string refusal_;
  hipDeviceProp_t prop_{};
  std::string bdf_;
  long pci_id_ = -1;
  bool gate_ = false;
  hipStream_t stream_ = nullptr;
};

}  // namespace

int run_runner(Family& family, int abi_version) {
  Runner runner(family, abi_version);
  Handlers h;
  h.platform = [&](const std::vector<std::string>& a) { return runner.platform(a); };
  h.enumerate_cmd = [&](const std::vector<std::string>& a) { return runner.enumerate_cmd(a); };
  h.measure = [&](const std::vector<std::string>& a) { return runner.measure(a); };
  return protocol_loop(std::cin, std::cout, h);
}

}  // namespace perfmon
