// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "check.h"
#include "protocol.h"

#include <sstream>
#include <stdexcept>

using namespace perfmon;

static void test_parse_command() {
  Command c = parse_command("  measure  a;b=1   op_attn_fwd   -  ");
  CHECK_EQ(c.name, "measure");
  CHECK_EQ(c.args.size(), 3u);
  CHECK_EQ(c.args[0], "a;b=1");
  CHECK_EQ(c.args[2], "-");
  CHECK(parse_command("   ").name.empty());
  CHECK(parse_command("platform").args.empty());
}

static void test_emitters() {
  std::ostringstream out;
  emit_ok(out, "{\"a\":1}");
  emit_unavailable(out, "no\nsuch backend");
  emit_error(out, "OK this is not success");
  emit_overheating(out, "temp=90");
  CHECK_EQ(out.str(),
           "OK {\"a\":1}\n"
           "UNAVAILABLE no such backend\n"
           "ERROR: OK this is not success\n"
           "OVERHEATING: temp=90\n");
}

static void test_loop() {
  Handlers h;
  int measured = 0;
  h.measure = [&](const std::vector<std::string>& args) {
    ++measured;
    if (args.size() != 3) throw std::runtime_error("bad args");
    if (args[2] == "aiter") return Reply::unavailable("launch failed");
    return Reply::ok("{\"n\":100}");
  };
  h.platform = [](const std::vector<std::string>&) { return Reply::ok("{}"); };
  // enumerate deliberately left unset.
  std::istringstream in(
      "platform\r\n"
      "\n"
      "measure e op_attn_fwd triton\n"
      "measure e op_attn_fwd aiter\n"
      "measure e\n"
      "enumerate e op_attn_fwd\n"
      "bogus\n"
      "exit\n"
      "platform\n");
  std::ostringstream out;
  CHECK_EQ(protocol_loop(in, out, h), 0);
  CHECK_EQ(measured, 3);
  CHECK_EQ(out.str(),
           "OK {}\n"
           "OK {\"n\":100}\n"
           "UNAVAILABLE launch failed\n"
           "ERROR: bad args\n"
           "ERROR: no handler for command: enumerate\n"
           "ERROR: unknown command: bogus\n");

  // EOF ends the loop like exit does.
  std::istringstream eof("platform");
  std::ostringstream out2;
  CHECK_EQ(protocol_loop(eof, out2, h), 0);
  CHECK_EQ(out2.str(), "OK {}\n");
}

static void test_json() {
  CHECK_EQ(json_string("a\"b\\c\nd\x01"), "\"a\\\"b\\\\c\\nd\\u0001\"");
  CHECK_EQ(json_string_array({}), "[]");
  CHECK_EQ(json_string_array({"triton_split", "aiter"}), "[\"triton_split\",\"aiter\"]");
  CHECK_EQ(json_double(1.5), "1.5");
  CHECK_EQ(json_double(0.1), "0.10000000000000001");
  CHECK_EQ(json_double(1.0 / 0.0), "null");
  CHECK_EQ(std::stod(json_double(123.456789012345678)), 123.456789012345678);
}

int main() {
  test_parse_command();
  test_emitters();
  test_loop();
  test_json();
  CHECK_MAIN_RETURN();
}
