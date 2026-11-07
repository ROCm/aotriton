// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "protocol.h"

#include <cmath>
#include <cstdio>
#include <sstream>
#include <stdexcept>

namespace perfmon {

namespace {

std::string one_line(const std::string& s) {
  std::string out = s;
  for (char& c : out) {
    if (c == '\n' || c == '\r') c = ' ';
  }
  return out;
}

void write_line(std::ostream& out, const std::string& line) {
  out << line << "\n";
  out.flush();
}

}  // namespace

Command parse_command(const std::string& line) {
  Command cmd;
  std::istringstream iss(line);
  std::string tok;
  if (iss >> tok) cmd.name = tok;
  while (iss >> tok) cmd.args.push_back(tok);
  return cmd;
}

void emit_ok(std::ostream& out, const std::string& json_body) {
  write_line(out, "OK " + one_line(json_body));
}

void emit_unavailable(std::ostream& out, const std::string& message) {
  write_line(out, "UNAVAILABLE " + one_line(message));
}

void emit_error(std::ostream& out, const std::string& message) {
  // The "ERROR: " prefix keeps an arbitrary message from being misread as
  // one of the other reply types.
  write_line(out, "ERROR: " + one_line(message));
}

void emit_overheating(std::ostream& out, const std::string& detail) {
  write_line(out, "OVERHEATING: " + one_line(detail));
}

int protocol_loop(std::istream& in, std::ostream& out, const Handlers& handlers) {
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    Command cmd = parse_command(line);
    if (cmd.name.empty()) continue;
    if (cmd.name == "exit") return 0;
    try {
      const Handler* h = nullptr;
      if (cmd.name == "measure") {
        h = &handlers.measure;
      } else if (cmd.name == "enumerate") {
        h = &handlers.enumerate_cmd;
      } else if (cmd.name == "platform") {
        h = &handlers.platform;
      } else {
        throw std::runtime_error("unknown command: " + cmd.name);
      }
      if (!*h) throw std::runtime_error("no handler for command: " + cmd.name);
      Reply r = (*h)(cmd.args);
      if (r.status == Reply::kOk) {
        emit_ok(out, r.text);
      } else {
        emit_unavailable(out, r.text);
      }
    } catch (const std::exception& e) {
      emit_error(out, e.what());
    }
  }
  return 0;
}

std::string json_string(const std::string& s) {
  std::string out = "\"";
  for (unsigned char c : s) {
    switch (c) {
      case '"':  out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += static_cast<char>(c);
        }
    }
  }
  out += "\"";
  return out;
}

std::string json_double(double v) {
  if (!std::isfinite(v)) return "null";
  // %.17g round-trips every double exactly.
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.17g", v);
  return buf;
}

std::string json_string_array(const std::vector<std::string>& items) {
  std::string out = "[";
  for (size_t i = 0; i < items.size(); ++i) {
    if (i) out += ",";
    out += json_string(items[i]);
  }
  out += "]";
  return out;
}

}  // namespace perfmon
