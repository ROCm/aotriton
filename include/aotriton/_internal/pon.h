// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AOTRITON_V2_API_PON_H
#define AOTRITON_V2_API_PON_H

#include <aotriton/config.h>

#include <cstdint>
#include <optional>
#include <string_view>

namespace AOTRITON_NS {

// A borrowed, read-only view over a ';'-separated 'k=v' PON (Plain / Python
// Object Notation) string. Holds no storage: the backing text is the
// compiled-in packed_string, which has static storage duration. Never
// construct from a temporary std::string.
//
// Grammar (measured from the packed strings actually emitted; the Python
// producer is python/utils/pon.py's `render_pon`):
//   section := pair (';' pair)*
//   pair    := key '=' value
//   value   := Python repr -- 0 | -1 | True | False | None | 'transposed' | 'auto'
//
// Every value is a plain `repr()`: a string is always single-quoted with
// nothing escaped (render_pon asserts this at build time), so `get_str` only
// ever needs to strip exactly one leading and one trailing quote.
//
// Parsing uses std::from_chars exclusively (never strtol/atoi/sscanf): the
// backing string_view is not NUL-terminated and may point into the middle of a
// larger concatenated buffer, so any NUL-seeking libc parser risks reading past
// this view's own end into unrelated data. Every lookup here is strictly
// bounded to [text_.begin(), text_.end()).
//
// `None` (any field can render it -- every knob is `T | None`) is treated as a
// lookup failure by every accessor, exactly like a missing key: it means "this
// knob was not computed for this image", not a literal three-letter result.
// EVERY accessor includes `contains()` and `find()`, not just the three `get_*`
// families -- otherwise `if (pon.contains("x")) use(*pon.get_int("x"));`, the
// pairing the two-form API invites, would dereference an empty optional on
// exactly the not-computed case. It is enforced in one place, `find_value()` in
// v3src/pon/pon.cc, so the promise cannot be true of some accessors only.
//
// Each accessor comes in TWO forms, and the difference is deliberate:
//
//   get_int(key)        -> std::optional<int64_t>   caller MUST handle a miss
//   get_int(key, dflt)  -> int64_t                  caller has named a fallback
//
// Neither aborts. The one-argument form returns an empty optional, which the
// type system will not let a caller spend as an integer -- so "what if this key
// is absent?" has to be answered at the call site, in code, before the value can
// be used. That is a strictly stronger guarantee than the abort this replaced:
// an abort catches the mistake at run time on whichever machine happens to hit
// the missing key, while an optional catches it at compile time on every
// machine. The two-argument form is for callers that genuinely have a sensible
// default, and it says so by naming it.
class Pon {
public:
  constexpr explicit Pon(std::string_view text = {}) noexcept : text_(text) {}

  bool contains(std::string_view key) const noexcept;
  std::optional<std::string_view> find(std::string_view key) const noexcept;

  // Empty optional on a missing key, an unparsable value, "None", or a range
  // error. There is no failure mode that is not one of those.
  std::optional<int64_t>          get_int (std::string_view key) const noexcept;
  std::optional<bool>             get_bool(std::string_view key) const noexcept;
  std::optional<std::string_view> get_str (std::string_view key) const noexcept;

  // Same lookup, with a caller-supplied fallback for every one of those cases.
  int64_t          get_int (std::string_view key, int64_t dflt) const noexcept;
  bool             get_bool(std::string_view key, bool dflt) const noexcept;
  std::string_view get_str (std::string_view key, std::string_view dflt) const noexcept;

private:
  std::string_view text_;
};

}

#endif
