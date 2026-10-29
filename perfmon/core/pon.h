// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// perfmon::Pon reads the PON text (';'-separated 'k=v', python/utils/pon.py's
// render_pon) of a perfmon entry. It is a COPY of the AOTriton runtime's reader
// (include/aotriton/_internal/pon.h, v3src/common/pon.cc), kept API-compatible
// and semantically identical on purpose: full-key match, first match wins,
// `None` is a miss, bounded reads, exact True/False, one pair of single quotes
// stripped. The two are maintained separately: perfmon must build against any
// AOTriton release, so it cannot share the runtime's code.
//
// Perfmon-only extensions (the runtime has none of them), marked below:
// get_double, for_each_pair and check.
//
// The central core knows no family's keys. A family's
// codec, e.g. one under modules/<family>/perfmon/core_ext, enumerates and reads them.
//
// Header-only and standard library only (no HIP, no AOTriton), on purpose. The
// core is built once per ROCm and shared by every subject, while an adapter is
// built per subject, so a C++ type crossing that compiled boundary would tie
// both to one standard library ABI; the core only passes plain-C text over
// perfmon_abi.h. An adapter includes this header from the core's installed
// include/perfmon/ and compiles its own copy.

#ifndef PERFMON_CORE_PON_H
#define PERFMON_CORE_PON_H

#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>

namespace perfmon {

namespace pon_detail {

// The value of the FIRST pair whose full key is `key`; "None" is a miss. Never
// reads outside `text`.
inline std::optional<std::string_view> find_value(std::string_view text,
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

}  // namespace pon_detail

// A borrowed view: the text must outlive the Pon, never construct it from a
// temporary std::string.
class Pon {
 public:
  constexpr explicit Pon(std::string_view text = {}) noexcept : text_(text) {}

  bool contains(std::string_view key) const noexcept {
    return pon_detail::find_value(text_, key).has_value();
  }

  std::optional<std::string_view> find(std::string_view key) const noexcept {
    return pon_detail::find_value(text_, key);
  }

  // Empty optional on a missing key, an unparsable value, "None", or a range
  // error.
  std::optional<int64_t> get_int(std::string_view key) const noexcept {
    const auto v = find(key);
    if (!v) return std::nullopt;
    int64_t out = 0;
    const char* end = v->data() + v->size();
    const auto [ptr, ec] = std::from_chars(v->data(), end, out);
    if (ec == std::errc() && ptr == end) return out;
    return std::nullopt;
  }

  std::optional<bool> get_bool(std::string_view key) const noexcept {
    const auto v = find(key);
    if (!v) return std::nullopt;
    if (*v == "True") return true;
    if (*v == "False") return false;
    return std::nullopt;
  }

  std::optional<std::string_view> get_str(std::string_view key) const noexcept {
    const auto v = find(key);
    if (!v) return std::nullopt;
    if (v->size() >= 2 && v->front() == '\'' && v->back() == '\'') {
      return v->substr(1, v->size() - 2);
    }
    return *v;
  }

  int64_t get_int(std::string_view key, int64_t dflt) const noexcept {
    return get_int(key).value_or(dflt);
  }
  bool get_bool(std::string_view key, bool dflt) const noexcept {
    return get_bool(key).value_or(dflt);
  }
  std::string_view get_str(std::string_view key, std::string_view dflt) const noexcept {
    return get_str(key).value_or(dflt);
  }

  // --- Perfmon extensions ----------------------------------------------------

  // Finite decimal numbers only (strtod alone would take nan, inf, hex floats
  // and leading spaces); an integer literal is accepted too. Same miss rules
  // as get_int.
  std::optional<double> get_double(std::string_view key) const {
    const auto v = find(key);
    if (!v || v->empty()) return std::nullopt;
    if (v->find_first_not_of("0123456789+-.eE") != std::string_view::npos) return std::nullopt;
    const std::string copy(*v);  // strtod needs the NUL the view lacks
    char* end = nullptr;
    const double d = std::strtod(copy.c_str(), &end);
    if (end != copy.c_str() + copy.size() || !std::isfinite(d)) return std::nullopt;
    return d;
  }

  // Calls fn(key, raw_value) for every pair in order, including duplicates and
  // "None" values. Lets a family's codec reject keys it does not know. A pair
  // without '=' is skipped; check() reports it.
  template <typename Fn>
  void for_each_pair(Fn&& fn) const {
    size_t pos = 0;
    while (pos <= text_.size()) {
      const size_t semi = text_.find(';', pos);
      const std::string_view pair = (semi == std::string_view::npos)
                                        ? text_.substr(pos)
                                        : text_.substr(pos, semi - pos);
      const size_t eq = pair.find('=');
      if (eq != std::string_view::npos) fn(pair.substr(0, eq), pair.substr(eq + 1));
      if (semi == std::string_view::npos) break;
      pos = semi + 1;
    }
  }

  // Strict syntax check, which the lenient lookups above do not do: "" if the
  // text is well formed, else a message. Rejects empty text, an empty pair, an
  // empty key, a pair without '=', and a repeated key (the lookups would
  // silently let the first win).
  std::string check() const {
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

 private:
  std::string_view text_;
};

// FNV-1a 64-bit.
inline uint64_t fnv1a64(std::string_view s) {
  uint64_t h = 0xcbf29ce484222325ULL;
  for (unsigned char c : s) {
    h ^= c;
    h *= 0x100000001b3ULL;
  }
  return h;
}

// The data seed of a measurement: a hash of the entry and the iface. It
// leaves the subject out on purpose, so every release sees identical inputs.
inline uint64_t entry_seed(std::string_view entry_pon, std::string_view iface) {
  return fnv1a64(std::string(entry_pon) + "|" + std::string(iface));
}

}  // namespace perfmon

#endif  // PERFMON_CORE_PON_H
