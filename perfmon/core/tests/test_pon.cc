// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "check.h"
#include "pon.h"

#include <string>
#include <string_view>
#include <vector>

using namespace perfmon;

// The lookup semantics below are those of the AOTriton runtime's reader
// (v3src/common/pon.cc); a divergence there should show up here.

static void test_lookup() {
  const Pon p("a=1;bb=-7;s='transposed';t=True;f=False;n=None;block_mask=9");
  CHECK(p.contains("a"));
  CHECK(!p.contains("zz"));
  CHECK(p.find("a").value() == "1");
  CHECK_EQ(p.get_int("a").value(), 1);
  CHECK_EQ(p.get_int("bb").value(), -7);
  CHECK(p.get_bool("t").value());
  CHECK(!p.get_bool("f").value());
  CHECK(p.get_str("s").value() == "transposed");
  // Full-key match: 'block' must not match 'block_mask'.
  CHECK(!p.contains("block"));
  CHECK(!p.contains("a="));
  // None is a miss for every accessor.
  CHECK(!p.contains("n"));
  CHECK(!p.find("n").has_value());
  CHECK(!p.get_int("n").has_value());
  CHECK(!p.get_str("n").has_value());
  // Defaults.
  CHECK_EQ(p.get_int("zz", 42), 42);
  CHECK_EQ(p.get_int("n", 5), 5);
  CHECK_EQ(p.get_int("a", 42), 1);
  CHECK(p.get_bool("zz", true));
  CHECK(p.get_str("zz", "d") == "d");
}

static void test_misses() {
  CHECK(!Pon("a=1x").get_int("a").has_value());
  CHECK(!Pon("a=").get_int("a").has_value());
  CHECK(!Pon("a=1.5").get_int("a").has_value());
  CHECK(!Pon("a=99999999999999999999").get_int("a").has_value());
  CHECK(!Pon("a=true").get_bool("a").has_value());
  CHECK(!Pon("a=1").get_bool("a").has_value());
  CHECK(!Pon("a=True ").get_bool("a").has_value());
  CHECK(Pon("a=").contains("a"));
  CHECK(!Pon().contains("a"));
  // First wins on a duplicate; a None first occurrence hides the second.
  CHECK_EQ(Pon("a=1;a=2").get_int("a").value(), 1);
  CHECK(!Pon("a=None;a=2").contains("a"));
  // A pair without '=' does not stop the scan.
  CHECK_EQ(Pon("junk;a=3").get_int("a").value(), 3);
  // Strings: one quote pair is stripped, anything else is returned as is.
  CHECK(Pon("s=''").get_str("s").value().empty());
  CHECK(Pon("s=bare").get_str("s").value() == "bare");
  CHECK(Pon("s='").get_str("s").value() == "'");
  // Bounded: a view cut out of a longer buffer never reads past its end.
  const std::string whole = "a=12;b=34";
  CHECK_EQ(Pon(std::string_view(whole).substr(0, 4)).get_int("a").value(), 12);
  CHECK_EQ(Pon(std::string_view(whole).substr(0, 3)).get_int("a").value(), 1);
  CHECK(!Pon(std::string_view(whole).substr(0, 4)).contains("b"));
}

// --- Extensions ----------------------------------------------------------------

static void test_get_double() {
  const Pon p("d=0.5;i=2;e=1e-05;n=-0.25;s='x';z=nan;h=inf;x=1e999;b=True;w= 1");
  CHECK_EQ(p.get_double("d").value(), 0.5);
  CHECK_EQ(p.get_double("i").value(), 2.0);  // an integer widens
  CHECK_EQ(p.get_double("e").value(), 1e-05);
  CHECK_EQ(p.get_double("n").value(), -0.25);
  CHECK(!p.get_double("s").has_value());
  CHECK(!p.get_double("z").has_value());
  CHECK(!p.get_double("h").has_value());
  CHECK(!p.get_double("x").has_value());
  CHECK(!p.get_double("b").has_value());
  CHECK(!p.get_double("w").has_value());
  CHECK(!p.get_double("missing").has_value());
}

static void test_for_each_pair() {
  std::vector<std::string> seen;
  Pon("a=1;b='x';a=2;n=None").for_each_pair([&](std::string_view k, std::string_view v) {
    seen.push_back(std::string(k) + ":" + std::string(v));
  });
  CHECK_EQ(seen.size(), 4u);
  CHECK_EQ(seen[0], "a:1");
  CHECK_EQ(seen[1], "b:'x'");
  CHECK_EQ(seen[2], "a:2");
  CHECK_EQ(seen[3], "n:None");
  seen.clear();
  Pon().for_each_pair([&](std::string_view, std::string_view) { seen.push_back("x"); });
  CHECK(seen.empty());
}

static void test_check() {
  CHECK_EQ(Pon("a=1;b='x';c=True;d=None").check(), "");
  CHECK_EQ(Pon("a=").check(), "");  // an empty value is the codec's business
  CHECK(Pon("").check().find("empty") != std::string::npos);
  CHECK(Pon("a").check().find("malformed") != std::string::npos);
  CHECK(Pon("=1").check().find("empty key") != std::string::npos);
  CHECK(Pon("a=1;").check().find("malformed") != std::string::npos);
  CHECK(Pon("a=1;;b=2").check().find("malformed") != std::string::npos);
  CHECK(Pon("a=1;b").check().find("no '='") != std::string::npos);
  CHECK(Pon("a=1;b=2;a=3").check().find("\"a\" is repeated") != std::string::npos);
  // Not a repeat: 'a' is a prefix of 'ab'.
  CHECK_EQ(Pon("a=1;ab=2;b=3").check(), "");
}

static void test_seed() {
  CHECK_EQ(fnv1a64(""), 0xcbf29ce484222325ULL);
  CHECK_EQ(fnv1a64("a"), 0xaf63dc4c8601ec8cULL);
  const std::string pon = "T_io='bf16';batch=512";
  const uint64_t fwd = entry_seed(pon, "op_attn_fwd");
  // Pinned: the seed is part of what makes releases comparable, so the hash
  // must never change. fnv1a64(pon + "|" + iface).
  CHECK_EQ(fwd, fnv1a64("T_io='bf16';batch=512|op_attn_fwd"));
  CHECK_EQ(fwd, 0x99208f08ef090d2bULL);  // computed independently in Python
  CHECK(fwd != entry_seed(pon, "op_attn_bwd"));
  CHECK(fwd != entry_seed("T_io='bf16';batch=256", "op_attn_fwd"));
}

int main() {
  test_lookup();
  test_misses();
  test_get_double();
  test_for_each_pair();
  test_check();
  test_seed();
  CHECK_MAIN_RETURN();
}
