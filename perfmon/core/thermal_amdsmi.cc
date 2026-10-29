// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// thermal.h backed by amd-smi (PERFMON_ENABLE_AMDSMI=ON).
//
// Only this process's GPU is opened, resolved by PCI BDF from HIP device 0
// (the launcher masks the process to one GPU). Enumerating every amd-smi
// handle instead would make this process show up on every GPU of the node,
// and HIP's view is the only mapping that survives HIP_VISIBLE_DEVICES.
//
// Built and run on gfx942 with amd-smi 26.5. Two facts learned there:
//  * amdsmi_get_temp_metric reports WHOLE degrees Celsius. Dividing by 1000,
//    as if it were millidegrees, turned 41 C into 0.041 C and silently
//    disabled the gate.
//  * gfx942 has no EDGE sensor and gfx1151 has no JUNCTION sensor, so the
//    sensor is picked per GPU, JUNCTION first.
// The cooldown loop itself has not yet been seen to trigger on hardware.

#include "thermal.h"
#include "protocol.h"

#include <hip/hip_runtime.h>
#include <amd_smi/amdsmi.h>

#include <chrono>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace perfmon {

namespace {

constexpr amdsmi_temperature_type_t kTempSensors[] = {
    AMDSMI_TEMPERATURE_TYPE_JUNCTION,
    AMDSMI_TEMPERATURE_TYPE_EDGE,
};

constexpr int kPollSeconds = 5;

struct OwnedDevice {
  amdsmi_processor_handle handle = nullptr;
  amdsmi_temperature_type_t sensor = AMDSMI_TEMPERATURE_TYPE_JUNCTION;
};

amdsmi_bdf_t bdf_of_device0() {
  hipDeviceProp_t prop;
  hipError_t err = hipGetDeviceProperties(&prop, 0);
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("thermal: hipGetDeviceProperties failed: ") +
                             hipGetErrorString(err));
  }
  amdsmi_bdf_t bdf{};
  bdf.bdf.function_number = 0;
  bdf.bdf.device_number = static_cast<uint64_t>(prop.pciDeviceID);
  bdf.bdf.bus_number = static_cast<uint64_t>(prop.pciBusID);
  bdf.bdf.domain_number = static_cast<uint64_t>(prop.pciDomainID);
  return bdf;
}

OwnedDevice open_device() {
  if (amdsmi_init(AMDSMI_INIT_AMD_GPUS) != AMDSMI_STATUS_SUCCESS) {
    throw std::runtime_error("thermal: amdsmi_init failed");
  }
  OwnedDevice dev;
  if (amdsmi_get_processor_handle_from_bdf(bdf_of_device0(), &dev.handle) !=
      AMDSMI_STATUS_SUCCESS) {
    amdsmi_shut_down();
    throw std::runtime_error("thermal: amdsmi_get_processor_handle_from_bdf failed");
  }
  for (auto sensor : kTempSensors) {
    int64_t t = 0;
    if (amdsmi_get_temp_metric(dev.handle, sensor, AMDSMI_TEMP_CURRENT, &t) ==
        AMDSMI_STATUS_SUCCESS) {
      dev.sensor = sensor;
      return dev;
    }
  }
  throw std::runtime_error("thermal: the GPU reports neither a JUNCTION nor an EDGE temperature");
}

const OwnedDevice& device() {
  // Opened once per process; a failure is retried on the next call.
  static OwnedDevice dev;
  static bool opened = false;
  if (!opened) {
    dev = open_device();
    opened = true;
  }
  return dev;
}

}  // namespace

bool thermal_available() {
  return true;
}

double read_temperature_c() {
  const OwnedDevice& dev = device();
  int64_t temp_c = 0;
  if (amdsmi_get_temp_metric(dev.handle, dev.sensor, AMDSMI_TEMP_CURRENT, &temp_c) !=
      AMDSMI_STATUS_SUCCESS) {
    throw std::runtime_error("thermal: amdsmi_get_temp_metric failed");
  }
  return static_cast<double>(temp_c);  // whole degrees Celsius; see the top
}

void wait_until_cool(double limit_c, std::ostream& out) {
  const auto start = std::chrono::steady_clock::now();
  for (double temp = read_temperature_c(); temp > limit_c; temp = read_temperature_c()) {
    const auto waited = std::chrono::duration_cast<std::chrono::seconds>(
                            std::chrono::steady_clock::now() - start).count();
    std::ostringstream detail;
    detail << "temp_c=" << temp << " limit_c=" << limit_c << " waited_s=" << waited;
    emit_overheating(out, detail.str());
    std::this_thread::sleep_for(std::chrono::seconds(kPollSeconds));
  }
}

}  // namespace perfmon
