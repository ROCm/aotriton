/* Copyright © 2026 Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: MIT
 *
 * perfmon_abi.h -- the C++ boundary between libperfmon_core (no AOTriton
 * dependency: HIP and amd-smi only) and a family adapter, which is built
 * against one subject's AOTriton and linked into that subject's bin/runner.
 *
 * Rules:
 *  - Only std types and hipStream_t cross this boundary. No AOTriton type
 *    does, so the core never needs an AOTriton header and times every subject
 *    with the same code.
 *  - Any change to Entry, Unsupported, Measurement or Family (a member, its
 *    signature, its meaning, the order of virtual functions) bumps
 *    PMON_ABI_VERSION. A subject's runner passes the version it was compiled
 *    with to run_runner(), which refuses a mismatch, so a stale adapter fails
 *    loudly instead of calling through a wrong vtable.
 *  - Toolchain contract: the core and a subject's adapter are built by the
 *    same ROCm toolchain (a subject is built with its core's ROCm), with
 *    libstdc++ and the same _GLIBCXX_USE_CXX11_ABI, because std::string,
 *    std::vector and exceptions cross the boundary. The core's symbols have
 *    default visibility (PMON_API), so vtables, typeinfo and Unsupported have
 *    one home, in libperfmon_core, and exceptions match across the .so files.
 */

#ifndef PERFMON_CORE_PERFMON_ABI_H
#define PERFMON_CORE_PERFMON_ABI_H

#include <hip/hip_runtime.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#define PMON_ABI_VERSION 3

#define PMON_API __attribute__((visibility("default")))

namespace perfmon {

// One measurement request. The core knows no family: `pon` is the entry's PON
// text, which it has only syntax-checked, and the family decodes it (and
// rejects a malformed one with an exception other than Unsupported). Valid for
// the duration of Family::prepare() only; copy what is needed. `seed` drives
// the input fill and any RNG it needs.
struct Entry {
  int iface;  // index from Family::iface_index()
  uint64_t seed;
  std::string_view pon;
};

// Thrown by Family::prepare() when this release cannot run the entry at all,
// e.g. a feature its API lacks. The runner reports the measurement as
// unavailable, for any backend; every other exception is an error.
struct PMON_API Unsupported : std::runtime_error {
  using std::runtime_error::runtime_error;
  ~Unsupported() override;  // out of line: one typeinfo, in the core
};

// Everything one measurement needs, allocated and filled by prepare(). Its
// destructor frees the buffers.
class PMON_API Measurement {
 public:
  virtual ~Measurement();
  // Enqueues one call of the operator on `stream`. Called inside a hipGraph
  // capture, so it must only enqueue stream-ordered work on `stream`, never
  // synchronize. Throws on failure.
  virtual void launch(hipStream_t stream) = 0;
};

class PMON_API Family {
 public:
  virtual ~Family();
  // The index of iface `name` (e.g. "op_attn_bwd"), or -1 if unknown.
  virtual int iface_index(std::string_view name) const = 0;
  // The canonical names of backend 0..n-1 of `iface`. Empty for a release
  // that cannot force a backend.
  virtual std::vector<std::string> backend_names(int iface) const = 0;
  // Allocates and fills every buffer one measurement needs. `backend` is an
  // index into backend_names(); nullopt is the unforced operator, as users
  // call it. Throws Unsupported or any other exception.
  virtual std::unique_ptr<Measurement> prepare(const Entry& entry,
                                               std::optional<int> backend) = 0;
};

// Serves the protocol on stdin/stdout until "exit" or EOF, and returns the
// process exit status. Every subject's bin/runner is main.cc, which only hands
// its family to this function; all measurement code is in the shared core, so
// every subject is timed by the same code.
//
// Startup checks do not end the process: a failed one turns every command into
// an error reply that says why, which the caller sees on its first command.
// They are: `abi_version` (the PMON_ABI_VERSION the caller was compiled with)
// equals the core's; the device's arch is in the core's arch list; and
// AOTRITON_THERMAL_GATE is valid and can be honoured.
PMON_API int run_runner(Family& family, int abi_version);

// The family of this runner binary. Not defined by the core: it is defined by
// the one family library linked into a subject's runner (each runner binary
// serves exactly one family), and main.cc passes it to run_runner().
Family& family();

}  // namespace perfmon

#endif  // PERFMON_CORE_PERFMON_ABI_H
