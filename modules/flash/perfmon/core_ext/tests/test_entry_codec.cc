// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "check.h"
#include "entry_codec.h"

#include <stdexcept>
#include <string>

using namespace perfmon::flash;

// The example from the runner's interface contract.
static const std::string kRoberta =
    "T_io='bf16';hdim_qk=64;hdim_v=64;causal=False;window_left=-1;window_right=-1;"
    "dropout_p=0.1;bias_type=1;bias_shape='b1qk';gqa_ratio=1;varlen=False;"
    "storage_flip=False;seqlen_q=384;seqlen_k=384;batch=512;n_heads_q=16";

static std::string replace(std::string s, const std::string& from, const std::string& to) {
  size_t pos = s.find(from);
  if (pos == std::string::npos) throw std::logic_error("test bug: " + from);
  return s.replace(pos, from.size(), to);
}

static void test_roundtrip() {
  Entry f = parse_entry(kRoberta);
  CHECK(f.dtype == Dtype::BFloat16);
  CHECK_EQ(f.hdim_qk, 64);
  CHECK_EQ(f.hdim_v, 64);
  CHECK(!f.causal);
  CHECK_EQ(f.window_left, -1);
  CHECK_EQ(f.window_right, -1);
  CHECK_EQ(f.dropout_p, 0.1);
  CHECK_EQ(f.bias_type, 1);
  CHECK(f.bias_shape == BiasShape::B1QK);
  CHECK_EQ(f.gqa_ratio, 1);
  CHECK(!f.varlen);
  CHECK(!f.storage_flip);
  CHECK_EQ(f.seqlen_q, 384);
  CHECK_EQ(f.seqlen_k, 384);
  CHECK_EQ(f.batch, 512);
  CHECK_EQ(f.n_heads_q, 16);
}

static void test_enum_values() {
  CHECK(dtype_from_name("fp16") == Dtype::Float16);
  CHECK(dtype_from_name("bf16") == Dtype::BFloat16);
  CHECK(dtype_from_name("fp32") == Dtype::Float32);
  CHECK_THROWS(dtype_from_name("float16"), "unknown dtype");
  CHECK(bias_shape_from_name("-") == BiasShape::None);
  CHECK(bias_shape_from_name("bhqk") == BiasShape::BHQK);
  CHECK(bias_shape_from_name("b1qk") == BiasShape::B1QK);
  CHECK(bias_shape_from_name("11qk") == BiasShape::B11QK);
  CHECK_THROWS(bias_shape_from_name("b1q1"), "unknown bias shape");
}

static void test_flags_and_numbers() {
  std::string s = replace(kRoberta, "causal=False", "causal=True");
  s = replace(s, "varlen=False", "varlen=0");
  s = replace(s, "storage_flip=False", "storage_flip=1");
  s = replace(s, "dropout_p=0.1", "dropout_p=0");
  Entry f = parse_entry(s);
  CHECK(f.causal);
  CHECK(!f.varlen);
  CHECK(f.storage_flip);
  CHECK_EQ(f.dropout_p, 0.0);
  CHECK_EQ(parse_entry(replace(kRoberta, "dropout_p=0.1", "dropout_p=1e-05")).dropout_p, 1e-05);
  CHECK_EQ(parse_entry(replace(kRoberta, "window_left=-1", "window_left=4096")).window_left, 4096);
  // Order does not matter.
  std::string reordered = "n_heads_q=16;" + replace(kRoberta, ";n_heads_q=16", "");
  CHECK_EQ(parse_entry(reordered).n_heads_q, 16);
}

static void test_rejects() {
  CHECK_THROWS(parse_entry(replace(kRoberta, ";batch=512", "")), "\"batch\": missing");
  CHECK_THROWS(parse_entry(kRoberta + ";N_HEADS=16"), "\"N_HEADS\": unknown key");
  CHECK_THROWS(parse_entry(kRoberta + ";batch=512"), "\"batch\" is repeated");
  CHECK_THROWS(parse_entry(replace(kRoberta, "causal=False", "causal=2")), "\"causal\"");
  CHECK_THROWS(parse_entry(replace(kRoberta, "causal=False", "causal='x'")), "\"causal\"");
  CHECK_THROWS(parse_entry(replace(kRoberta, "T_io='bf16'", "T_io=bf16")), "T_io");
  CHECK_THROWS(parse_entry(replace(kRoberta, "T_io='bf16'", "T_io=1")), "single-quoted");
  CHECK_THROWS(parse_entry(replace(kRoberta, "hdim_qk=64", "hdim_qk=64.0")), "integer");
  CHECK_THROWS(parse_entry(replace(kRoberta, "batch=512", "batch=0")), "positive");
  CHECK_THROWS(parse_entry(replace(kRoberta, "gqa_ratio=1", "gqa_ratio=3")), "divisible");
  CHECK_THROWS(parse_entry(replace(kRoberta, "dropout_p=0.1", "dropout_p=1.0")), "[0, 1)");
  CHECK_THROWS(parse_entry(replace(kRoberta, "dropout_p=0.1", "dropout_p=nan")), "dropout_p");
  CHECK_THROWS(parse_entry(replace(kRoberta, "bias_shape='b1qk'", "bias_shape='-'")),
               "bias_shape");
  CHECK_THROWS(parse_entry(replace(kRoberta, "window_right=-1", "window_right=-2")),
               "window_right");
  CHECK_THROWS(parse_entry(replace(kRoberta, "batch=512", "batch=99999999999")),
               "out of range");
  CHECK_THROWS(parse_entry(replace(kRoberta, "causal=False", "causal=False ")), "\"causal\"");
  CHECK_THROWS(parse_entry(replace(kRoberta, ";causal=False", ";causal")), "malformed");
}

int main() {
  test_roundtrip();
  test_enum_values();
  test_flags_and_numbers();
  test_rejects();
  CHECK_MAIN_RETURN();
}
