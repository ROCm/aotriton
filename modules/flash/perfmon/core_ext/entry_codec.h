// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Decodes a flash entry PON into Entry. The wire form is what
// python/utils/pon.py's render_pon writes, e.g.
//   T_io='bf16';hdim_qk=64;hdim_v=64;causal=False;window_left=-1;window_right=-1;
//   dropout_p=0.1;bias_type=1;bias_shape='b1qk';gqa_ratio=1;varlen=False;
//   storage_flip=False;seqlen_q=384;seqlen_k=384;batch=512;n_heads_q=16
// (one line, no spaces). Exactly these keys: a missing, unknown or wrongly
// typed key is an error, so a Python side that adds a field cannot be silently
// half-understood. Flags may be True/False or 0/1.
//
// Flash only and release-neutral: no HIP, no AOTriton, so it is unit-tested
// without ROCm and compiled into every subject's adapter (libperfmon_flash).
// It is not part of the central core (perfmon/core), which knows no family;
// the syntax layer is the core's generic perfmon::Pon (pon.h).

#ifndef PERFMON_FLASH_CORE_EXT_ENTRY_CODEC_H
#define PERFMON_FLASH_CORE_EXT_ENTRY_CODEC_H

#include <cstdint>
#include <string>

namespace perfmon::flash {

enum class Dtype : int32_t {
  Float16 = 0,   // 'fp16'
  BFloat16 = 1,  // 'bf16'
  Float32 = 2,   // 'fp32'
};

// Which dimensions of the (B, H, Sq, Sk) bias are real. Broadcast dimensions
// get stride 0, which is how PyTorch passes an expanded key-padding mask.
enum class BiasShape : int32_t {
  None = 0,  // '-', no bias
  BHQK = 1,  // one bias per batch and head
  B1QK = 2,  // broadcast over heads
  B11QK = 3,  // '11qk': broadcast over batch and heads
};

// One entry: a problem plus all its sizes. n_heads_kv = n_heads_q / gqa_ratio.
// Window values of -1 mean "no window".
struct Entry {
  Dtype     dtype = Dtype::Float16;
  int32_t   hdim_qk = 0;
  int32_t   hdim_v = 0;
  bool      causal = false;
  int32_t   window_left = -1;
  int32_t   window_right = -1;
  double    dropout_p = 0.0;
  int32_t   bias_type = 0;  // 0: none, 1: a matrix bias
  BiasShape bias_shape = BiasShape::None;
  int32_t   gqa_ratio = 1;
  bool      varlen = false;
  bool      storage_flip = false;
  int32_t   seqlen_q = 0;
  int32_t   seqlen_k = 0;
  int32_t   batch = 0;
  int32_t   n_heads_q = 0;
};

// Throws std::invalid_argument naming the offending key on any malformed,
// missing, unknown or inconsistent field.
Entry parse_entry(const std::string& entry_pon);

// T_io name ('fp16', 'bf16', 'fp32') -> Dtype. Throws on others.
Dtype dtype_from_name(const std::string& name);
// bias_shape name ('-', 'bhqk', 'b1qk', '11qk') -> BiasShape. Throws on others.
BiasShape bias_shape_from_name(const std::string& name);

}  // namespace perfmon::flash

#endif  // PERFMON_FLASH_CORE_EXT_ENTRY_CODEC_H
