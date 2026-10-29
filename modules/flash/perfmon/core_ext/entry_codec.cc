// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "entry_codec.h"

#include <perfmon/pon.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace perfmon::flash {

namespace {

constexpr const char* kKeys[] = {
    "T_io",       "hdim_qk",      "hdim_v",    "causal",     "window_left", "window_right",
    "dropout_p",  "bias_type",    "bias_shape", "gqa_ratio", "varlen",      "storage_flip",
    "seqlen_q",   "seqlen_k",     "batch",     "n_heads_q",
};

[[noreturn]] void fail(const std::string& key, const std::string& why) {
  throw std::invalid_argument("entry \"" + key + "\": " + why);
}

// The raw value of `key`; "None" counts as missing, like every Pon lookup.
std::string_view need(const Pon& r, const char* key) {
  const auto v = r.find(key);
  if (!v) fail(key, "missing");
  return *v;
}

int32_t get_int32(const Pon& r, const char* key) {
  need(r, key);
  const auto v = r.get_int(key);
  if (!v) fail(key, "expected an integer");
  if (*v < std::numeric_limits<int32_t>::min() || *v > std::numeric_limits<int32_t>::max()) {
    fail(key, "out of range");
  }
  return static_cast<int32_t>(*v);
}

// True/False, or 0/1: Python's render_pon writes 0 and 1 for these fields.
bool get_flag(const Pon& r, const char* key) {
  need(r, key);
  if (const auto b = r.get_bool(key)) return *b;
  if (const auto i = r.get_int(key); i && (*i == 0 || *i == 1)) return *i == 1;
  fail(key, "expected True/False or 0/1");
}

std::string get_name(const Pon& r, const char* key) {
  // get_str also passes an unquoted value through, which the wire form lacks.
  const std::string_view raw = need(r, key);
  if (raw.size() < 2 || raw.front() != '\'' || raw.back() != '\'') {
    fail(key, "expected a single-quoted string");
  }
  return std::string(*r.get_str(key));
}

}  // namespace

Dtype dtype_from_name(const std::string& name) {
  if (name == "fp16") return Dtype::Float16;
  if (name == "bf16") return Dtype::BFloat16;
  if (name == "fp32") return Dtype::Float32;
  fail("T_io", "unknown dtype '" + name + "'");
}

BiasShape bias_shape_from_name(const std::string& name) {
  if (name == "-") return BiasShape::None;
  if (name == "bhqk") return BiasShape::BHQK;
  if (name == "b1qk") return BiasShape::B1QK;
  if (name == "11qk") return BiasShape::B11QK;
  fail("bias_shape", "unknown bias shape '" + name + "'");
}

Entry parse_entry(const std::string& entry_pon) {
  const Pon r(entry_pon);
  // The lookups are lenient (first wins, no syntax check), so check first.
  const std::string bad = r.check();
  if (!bad.empty()) throw std::invalid_argument("entry: " + bad);
  r.for_each_pair([](std::string_view key, std::string_view) {
    bool known = false;
    for (const char* k : kKeys) known = known || key == k;
    if (!known) fail(std::string(key), "unknown key");
  });

  Entry f;
  f.dtype = dtype_from_name(get_name(r, "T_io"));
  f.hdim_qk = get_int32(r, "hdim_qk");
  f.hdim_v = get_int32(r, "hdim_v");
  f.causal = get_flag(r, "causal");
  f.window_left = get_int32(r, "window_left");
  f.window_right = get_int32(r, "window_right");
  need(r, "dropout_p");
  const auto dropout = r.get_double("dropout_p");
  if (!dropout) fail("dropout_p", "expected a number");
  f.dropout_p = *dropout;
  f.bias_type = get_int32(r, "bias_type");
  f.bias_shape = bias_shape_from_name(get_name(r, "bias_shape"));
  f.gqa_ratio = get_int32(r, "gqa_ratio");
  f.varlen = get_flag(r, "varlen");
  f.storage_flip = get_flag(r, "storage_flip");
  f.seqlen_q = get_int32(r, "seqlen_q");
  f.seqlen_k = get_int32(r, "seqlen_k");
  f.batch = get_int32(r, "batch");
  f.n_heads_q = get_int32(r, "n_heads_q");

  // Consistency. Anything rejected here would otherwise surface as a GPU
  // fault or a wrong-sized buffer much later.
  const struct { const char* key; int32_t v; } positive[] = {
      {"hdim_qk", f.hdim_qk},   {"hdim_v", f.hdim_v},     {"gqa_ratio", f.gqa_ratio},
      {"seqlen_q", f.seqlen_q}, {"seqlen_k", f.seqlen_k}, {"batch", f.batch},
      {"n_heads_q", f.n_heads_q},
  };
  for (const auto& p : positive) {
    if (p.v <= 0) fail(p.key, "must be positive");
  }
  if (f.n_heads_q % f.gqa_ratio != 0) fail("n_heads_q", "not divisible by gqa_ratio");
  if (f.window_left < -1) fail("window_left", "must be -1 or a window size");
  if (f.window_right < -1) fail("window_right", "must be -1 or a window size");
  if (!(f.dropout_p >= 0.0 && f.dropout_p < 1.0)) fail("dropout_p", "must be in [0, 1)");
  if (f.bias_type != 0 && f.bias_type != 1) fail("bias_type", "must be 0 or 1");
  if ((f.bias_type == 0) != (f.bias_shape == BiasShape::None)) {
    fail("bias_shape", "must be '-' exactly when bias_type is 0");
  }
  return f;
}

}  // namespace perfmon::flash
