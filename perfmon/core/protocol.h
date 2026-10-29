// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// The runner's line protocol, as pgdef's ExaidProxy speaks it.
//
//   * One command per line on stdin, tokens separated by whitespace:
//       enumerate <entry_pon> <iface>
//       measure <entry_pon> <iface> <backend>
//       platform
//       exit
//     An entry PON never contains a space (python/utils/pon.py guarantees
//     it), so it is always exactly one token.
//   * Replies, one line each, flushed:
//       OK <json>              success
//       UNAVAILABLE <message>  this release or arch cannot run this (entry,
//                              backend): prepare said unsupported, or the
//                              untimed first launch of a FORCED backend
//                              failed (a result, not a failure). A failed
//                              first launch of the unforced operator
//                              (backend "-") is an error instead: the
//                              default dispatch failing on a supported
//                              entry is a regression
//       OVERHEATING: <detail>  keep-alive while the thermal gate waits;
//                              ExaidProxy skips these without resetting its
//                              timeout
//       anything else          an error; this module writes "ERROR: <message>"
//
// Standard library only, so it is unit-tested without ROCm.

#ifndef PERFMON_CORE_PROTOCOL_H
#define PERFMON_CORE_PROTOCOL_H

#include <functional>
#include <istream>
#include <ostream>
#include <string>
#include <vector>

namespace perfmon {

struct Command {
  std::string name;               // empty for a blank line
  std::vector<std::string> args;
};

Command parse_command(const std::string& line);

// What a handler returns. Throwing std::exception instead yields an error
// line carrying what().
struct Reply {
  enum Status { kOk, kUnavailable };
  Status status = kOk;
  std::string text;  // JSON for kOk, a message for kUnavailable

  static Reply ok(std::string json) { return Reply{kOk, std::move(json)}; }
  static Reply unavailable(std::string message) {
    return Reply{kUnavailable, std::move(message)};
  }
};

// Each writes one '\n'-terminated line and flushes it: the proxy blocks on
// readline, so an unflushed reply would hang it. Newlines inside a message
// are replaced by spaces, so one reply is always one line.
void emit_ok(std::ostream& out, const std::string& json_body);
void emit_unavailable(std::ostream& out, const std::string& message);
void emit_error(std::ostream& out, const std::string& message);
void emit_overheating(std::ostream& out, const std::string& detail);

using Handler = std::function<Reply(const std::vector<std::string>&)>;

struct Handlers {
  Handler measure;
  Handler enumerate_cmd;
  Handler platform;
};

// Serves commands from `in` until "exit" or EOF. Unknown commands and
// handler exceptions become error lines; the loop keeps going. Returns 0.
int protocol_loop(std::istream& in, std::ostream& out, const Handlers& handlers);

// --- Minimal JSON writing (the project has no JSON library) --------------

// `s` as a quoted JSON string.
std::string json_string(const std::string& s);
// A finite double in a form that reads back bit-identically ("null" if not
// finite, which JSON cannot carry).
std::string json_double(double v);
std::string json_string_array(const std::vector<std::string>& items);

}  // namespace perfmon

#endif  // PERFMON_CORE_PROTOCOL_H
