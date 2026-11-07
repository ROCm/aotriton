// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Code shared by every flash adapter (<tag>/adapter.cc).
//
// The dividing line is not "used more than once" but "does not depend on the
// part of AOTriton's API that drifts between releases", so that the drift stays
// in the one small adapter file of each release. What is stable across 0.9.2b
// to head, read from each tag's include/aotriton/{util,dtypes}.h:
//   TensorView<Rank>(intptr_t, sizes, strides, DType), TensorView<0>(intptr_t,
//   DType), TensorView<Rank>::get_null_tensor(DType), kFloat16, kBFloat16,
//   kFloat32, kInt32, kUInt64.
// What drifts, and stays in the adapters: the entry points and their params
// structs, how a backend is forced, and LazyTensor.
//
// An adapter only builds one measurement (all buffers and the params, in a
// class deriving from perfmon::Measurement). Everything the family needs
// around that (the interface names, the backend table, the scratch stream) is
// in family.cc.
//
// This source is compiled once per subject, together with that subject's
// adapter, so "shared" means shared source, never a shared binary. The only
// binary shared by every subject is libperfmon_core.

#ifndef PERFMON_FLASH_LIB_COMMON_H
#define PERFMON_FLASH_LIB_COMMON_H

#include <perfmon/fill.h>
#include <perfmon/perfmon_abi.h>

#include "entry_codec.h"  // modules/flash/perfmon/core_ext

#include <aotriton/config.h>
#include <aotriton/dtypes.h>
#include <aotriton/util.h>

#include <hip/hip_runtime.h>

#include <array>
#include <memory>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace perfmon_flash {

// Thrown when this release cannot run the entry at all (an API without the
// feature): an unavailable measurement rather than an error.
using perfmon::Unsupported;

// The interface indices, as returned by FlashFamily::iface_index().
enum Iface : int32_t {
  kIfaceFwd = 0,  // "op_attn_fwd"
  kIfaceBwd = 1,  // "op_attn_bwd"
  kIfaceCount = 2,
};

// --- Backend table ---------------------------------------------------------
// Generated at configure time (backends_table.cc) from the release's
// backends.csv or from its public backends.h. Index i is what
// attn_options::force_backend_index means. count is 0 for a release that
// cannot force a backend.
struct BackendNames {
  const char* const* names;
  int count;
};
extern const BackendNames kBackendNames[kIfaceCount];

// The name of backend `index` of `iface`, or nullptr if out of range.
const char* backend_name(int32_t iface, int index);

// --- Device memory -----------------------------------------------------------
// Owns every device allocation of one context, freed in reverse order.
class Arena {
 public:
  Arena() = default;
  Arena(const Arena&) = delete;
  Arena& operator=(const Arena&) = delete;
  ~Arena();
  // Throws std::runtime_error when hipMalloc fails.
  void* alloc(size_t bytes);

 private:
  std::vector<void*> ptrs_;
};

void hip_check(hipError_t err, const char* what);

// For Measurement::launch(): throws when an operator call returned an error.
void launch_check(hipError_t err);

// --- The request ---------------------------------------------------------------
// What FlashFamily::prepare() hands an adapter: the core's perfmon::Entry
// (iface, seed, PON text) with the PON decoded by core_ext's codec.
struct Request {
  int32_t iface = 0;
  uint64_t seed = 0;
  perfmon::flash::Entry entry;
};

// The core's fill element type of a flash dtype.
perfmon::ElementType element_type(perfmon::flash::Dtype dtype);

// --- The entry, validated -----------------------------------------------------
// Sizes and options common to every release. dims_of() rejects what no adapter
// can run (Unsupported) and what is malformed (std::invalid_argument).
struct Dims {
  uint64_t batch = 0;
  uint64_t heads_q = 0;
  uint64_t heads_kv = 0;
  uint64_t seqlen_q = 0;
  uint64_t seqlen_k = 0;
  uint64_t hdim = 0;  // hdim_qk == hdim_v is enforced
  perfmon::flash::Dtype dtype = perfmon::flash::Dtype::Float16;
  bool storage_flip = false;
  perfmon::flash::BiasShape bias_shape = perfmon::flash::BiasShape::None;
  double dropout_p = 0.0;
  uint64_t seed = 0;
  float sm_scale = 1.0f;
};
Dims dims_of(const Request& req);

// How the entry's mask maps onto an API.
enum class CausalMode {
  None,
  TopLeft,  // plain causal, aligned at the top left
  Window,   // an explicit window of `left` and `right` positions
};
struct Causal {
  CausalMode mode = CausalMode::None;
  int32_t left = 0;
  int32_t right = 0;
};
// `has_window` is whether the release has windowed attention. Without it, an
// entry with a window is Unsupported. An entry whose window leaves one side
// open, or has a causal mask plus a right side beyond the diagonal, is
// Unsupported everywhere: TODO(perfmon): decide with hardware what an open
// side means to each release's kernels before mapping it.
Causal resolve_causal(const Request& req, bool has_window);

// --- Tensors -------------------------------------------------------------------
using AOTRITON_NS::TensorView;

// One (B, H, S, D) tensor of T_io: uniformly filled, or zeroed for outputs.
// With storage_flip the buffer is physically (B, S, H, D) and the logical view
// is its transpose, as the tuning reference allocates it. `salt` keeps buffers
// of one entry different from each other and identical across releases.
TensorView<4> make_bhsd(Arena* arena, const Dims& d, uint64_t heads, uint64_t seqlen,
                        uint64_t salt, bool zero, hipStream_t stream);

// L and D of the backward pass: (B, H*S) fp32.
TensorView<2> make_lse(Arena* arena, const Dims& d, uint64_t salt, bool zero,
                       hipStream_t stream);

// The bias in the layout of d.bias_shape: broadcast dimensions are size 1 in
// memory and get stride 0, like PyTorch's expanded key-padding mask. The key
// dimension is padded to a multiple of 8 and sliced back, as the tuning
// reference does, because the row stride is what memory access depends on.
TensorView<4> make_bias(Arena* arena, const Dims& d, uint64_t salt, hipStream_t stream);

// The bias gradient: a real buffer of the bias's layout for bhqk. For a
// broadcast bias a real gradient would need a reduction over the broadcast
// dimensions, so it is the null tensor (the kernels skip the store when its
// strides are 0). TODO(perfmon): confirm on hardware that every release
// accepts the null gradient, and that it does not change the timing.
TensorView<4> make_bias_grad(Arena* arena, const Dims& d, uint64_t salt, hipStream_t stream);

// fp32 (B, Hq, Sq, D) accumulator of dq, never storage_flipped: it is not one
// of Q, K and V.
TensorView<4> make_dq_acc(Arena* arena, const Dims& d);

// Dropout RNG state. All null when dropout_p == 0 (the kernels never read
// them); otherwise real device scalars derived from the entry's seed, so every
// release draws the same mask. Null ones would fault.
struct Philox {
  TensorView<0> seed = TensorView<0>::get_null_tensor(AOTRITON_NS::kUInt64);
  TensorView<0> offset1 = TensorView<0>::get_null_tensor(AOTRITON_NS::kUInt64);
  TensorView<0> seed_out = TensorView<0>::get_null_tensor(AOTRITON_NS::kUInt64);
  TensorView<0> offset_out = TensorView<0>::get_null_tensor(AOTRITON_NS::kUInt64);
  uint64_t offset2 = 0;
};
Philox make_philox(Arena* arena, const Dims& d);

// A device int32 initialised to zero, for the causal tile counter.
void* make_counter(Arena* arena, hipStream_t stream);
// Enqueues zeroing of that counter on `stream`.
void zero_counter(void* counter, hipStream_t stream);

// --- What each adapter implements ---------------------------------------------------
// The backend argument of adapter_prepare() for the operator as users call it.
constexpr int kBackendUnforced = -1;

// Builds the measurement of `req`. `backend` is kBackendUnforced or an index
// into the release's backend table (FlashFamily has checked the range, and it
// is never >= 0 for a release without forcing). Every fill is enqueued on
// `scratch`, which the caller synchronizes. The returned object's launch()
// enqueues one operator call and throws when it fails; its destructor frees
// the buffers. Throws Unsupported, or any other exception for an error.
std::unique_ptr<perfmon::Measurement> adapter_prepare(const Request& req, int backend,
                                                      hipStream_t scratch);

}  // namespace perfmon_flash

#endif  // PERFMON_FLASH_LIB_COMMON_H
