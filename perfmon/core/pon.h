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
// Standard library only (no HIP, no AOTriton). The implementation is pon.cc,
// compiled into libperfmon_core+rocm<ver>.so and exported with default
// visibility. An adapter includes this header from the core's installed
// include/perfmon/ and links the core's copy; it must not compile its own.
// So perfmon::Pon crosses the core/adapter boundary as a C++ API. That is
// acceptable because a subject's runner and adapter are always built with the
// same ROCm toolchain (and libstdc++) as the core of that ROCm.

#ifndef PERFMON_CORE_PON_H
#define PERFMON_CORE_PON_H

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace perfmon {

// A borrowed view: the text must outlive the Pon, never construct it from a
// temporary std::string.
class Pon {
 public:
  constexpr explicit Pon(std::string_view text = {}) noexcept : text_(text) {}

  bool contains(std::string_view key) const noexcept;
  std::optional<std::string_view> find(std::string_view key) const noexcept;

  // Empty optional on a missing key, an unparsable value, "None", or a range
  // error.
  std::optional<int64_t> get_int(std::string_view key) const noexcept;
  std::optional<bool> get_bool(std::string_view key) const noexcept;
  std::optional<std::string_view> get_str(std::string_view key) const noexcept;

  int64_t get_int(std::string_view key, int64_t dflt) const noexcept;
  bool get_bool(std::string_view key, bool dflt) const noexcept;
  std::string_view get_str(std::string_view key, std::string_view dflt) const noexcept;

  // --- Perfmon extensions ----------------------------------------------------

  // Finite decimal numbers only (strtod alone would take nan, inf, hex floats
  // and leading spaces); an integer literal is accepted too. Same miss rules
  // as get_int.
  std::optional<double> get_double(std::string_view key) const;

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
  std::string check() const;

 private:
  std::string_view text_;
};

// FNV-1a 64-bit.
uint64_t fnv1a64(std::string_view s);

// The data seed of a measurement: a hash of the entry and the iface. It
// leaves the subject out on purpose, so every release sees identical inputs.
uint64_t entry_seed(std::string_view entry_pon, std::string_view iface);

}  // namespace perfmon

#endif  // PERFMON_CORE_PON_H
