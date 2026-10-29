// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Implementation of perfmon::Pon, see pon.h.

#include "pon.h"

#include <charconv>
#include <cmath>
#include <cstdlib>

namespace perfmon {

namespace {

// The value of the FIRST pair whose full key is `key`; "None" is a miss. Never
// reads outside `text`.
std::optional<std::string_view> find_value(std::string_view text,
                                         std::string_view key) noexcept {
  size_t pos = 0;
  while (pos <= text.size()) {
    const size_t semi = text.find(';', pos);
    const std::string_view pair = (semi == std::string_view::npos)
                                      ? text.substr(pos)
                                      : text.substr(pos, semi - pos);
    const size_t eq = pair.find('=');
    if (eq != std::string_view::npos && pair.substr(0, eq) == key) {
      const std::string_view value = pair.substr(eq + 1);
      if (value == "None") return std::nullopt;
      return value;
    }
    if (semi == std::string_view::npos) break;
    pos = semi + 1;
  }
  return std::nullopt;
}

}  // namespace

bool Pon::contains(std::string_view key) const noexcept {
  return find_value(text_, key).has_value();
}

std::optional<std::string_view> Pon::find(std::string_view key) const noexcept {
  return find_value(text_, key);
}

std::optional<int64_t> Pon::get_int(std::string_view key) const noexcept {
  const auto v = find(key);
  if (!v) return std::nullopt;
  int64_t out = 0;
  const char* end = v->data() + v->size();
  const auto [ptr, ec] = std::from_chars(v->data(), end, out);
  if (ec == std::errc() && ptr == end) return out;
  return std::nullopt;
}

std::optional<bool> Pon::get_bool(std::string_view key) const noexcept {
  const auto v = find(key);
  if (!v) return std::nullopt;
  if (*v == "True") return true;
  if (*v == "False") return false;
  return std::nullopt;
}

std::optional<std::string_view> Pon::get_str(std::string_view key) const noexcept {
  const auto v = find(key);
  if (!v) return std::nullopt;
  if (v->size() >= 2 && v->front() == '\'' && v->back() == '\'') {
    return v->substr(1, v->size() - 2);
  }
  return *v;
}

int64_t Pon::get_int(std::string_view key, int64_t dflt) const noexcept {
  return get_int(key).value_or(dflt);
}

bool Pon::get_bool(std::string_view key, bool dflt) const noexcept {
  return get_bool(key).value_or(dflt);
}

std::string_view Pon::get_str(std::string_view key, std::string_view dflt) const noexcept {
  return get_str(key).value_or(dflt);
}

std::optional<double> Pon::get_double(std::string_view key) const {
  const auto v = find(key);
  if (!v || v->empty()) return std::nullopt;
  if (v->find_first_not_of("0123456789+-.eE") != std::string_view::npos) return std::nullopt;
  const std::string copy(*v);  // strtod needs the NUL the view lacks
  char* end = nullptr;
  const double d = std::strtod(copy.c_str(), &end);
  if (end != copy.c_str() + copy.size() || !std::isfinite(d)) return std::nullopt;
  return d;
}

std::string Pon::check() const {
  if (text_.empty()) return "empty PON";
  size_t pos = 0;
  for (;;) {
    const size_t semi = text_.find(';', pos);
    const std::string_view pair = (semi == std::string_view::npos)
                                      ? text_.substr(pos)
                                      : text_.substr(pos, semi - pos);
    const size_t eq = pair.find('=');
    if (eq == std::string_view::npos) {
      return "malformed pair at offset " + std::to_string(pos) + ": no '='";
    }
    const std::string_view key = pair.substr(0, eq);
    if (key.empty()) return "malformed pair at offset " + std::to_string(pos) + ": empty key";
    // Earlier pairs only: quadratic, but entries have a few dozen of them.
    const std::string_view before = text_.substr(0, pos);
    size_t p = 0;
    while (p < before.size()) {
      const size_t s = before.find(';', p);
      const std::string_view prev = (s == std::string_view::npos)
                                        ? before.substr(p)
                                        : before.substr(p, s - p);
      if (prev.substr(0, prev.find('=')) == key) {
        return "key \"" + std::string(key) + "\" is repeated";
      }
      if (s == std::string_view::npos) break;
      p = s + 1;
    }
    if (semi == std::string_view::npos) return "";
    pos = semi + 1;
  }
}

uint64_t fnv1a64(std::string_view s) {
  uint64_t h = 0xcbf29ce484222325ULL;
  for (unsigned char c : s) {
    h ^= c;
    h *= 0x100000001b3ULL;
  }
  return h;
}

uint64_t entry_seed(std::string_view entry_pon, std::string_view iface) {
  return fnv1a64(std::string(entry_pon) + "|" + std::string(iface));
}

}  // namespace perfmon
