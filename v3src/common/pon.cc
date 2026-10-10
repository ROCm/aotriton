// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/_internal/pon.h>

#include <charconv>

namespace AOTRITON_NS {

namespace {

// Find the value substring of the FIRST 'key=value' pair matching `key` in
// `text` (';'-separated). Duplicate keys: first wins. Matches on the FULL key
// -- 'block_m' must not match a pair keyed 'block_mask'. Never reads outside
// [text.begin(), text.end()); if `text` is a sub-view deliberately cut out of
// a larger buffer (no NUL at its own end), this still only ever inspects
// text[0, text.size()).
//
// A value of "None" is a MISS, here rather than in each accessor, which is
// what makes pon.h's "None is treated as a lookup failure by every accessor,
// exactly like a missing key" true of contains()/find() as well as of the
// three get_* families. Filtered at the match rather than by continuing the
// scan, so first-wins still holds for a duplicated key.
//
// Behaviour of get_int/get_bool/get_str is unchanged by this: from_chars
// already rejected "None", and the other two matched it exactly. What changes
// is that `if (pon.contains("x")) use(*pon.get_int("x"));` -- the idiomatic
// pairing -- no longer dereferences an empty optional on precisely the "this
// knob was not computed for this image" case the class exists to carry.
std::optional<std::string_view> find_value(std::string_view text, std::string_view key) noexcept {
  size_t pos = 0;
  while (pos <= text.size()) {
    size_t semi = text.find(';', pos);
    std::string_view pair = (semi == std::string_view::npos)
                             ? text.substr(pos)
                             : text.substr(pos, semi - pos);
    size_t eq = pair.find('=');
    if (eq != std::string_view::npos) {
      std::string_view pkey = pair.substr(0, eq);
      if (pkey == key) {
        std::string_view value = pair.substr(eq + 1);
        if (value == "None")
          return std::nullopt;
        return value;
      }
    }
    if (semi == std::string_view::npos)
      break;
    pos = semi + 1;
  }
  return std::nullopt;
}

}  // anonymous namespace

bool Pon::contains(std::string_view key) const noexcept {
  return find_value(text_, key).has_value();
}

std::optional<std::string_view> Pon::find(std::string_view key) const noexcept {
  return find_value(text_, key);
}

std::optional<int64_t> Pon::get_int(std::string_view key) const noexcept {
  auto v = find_value(text_, key);
  if (!v)
    return std::nullopt;
  int64_t out = 0;
  const char* begin = v->data();
  const char* end = v->data() + v->size();
  auto [ptr, ec] = std::from_chars(begin, end, out);
  // Bounded, no trailing garbage, overflow (result_out_of_range) is a miss.
  // "None" fails here too: from_chars rejects it, so it needs no special case.
  if (ec == std::errc() && ptr == end)
    return out;
  return std::nullopt;
}

std::optional<bool> Pon::get_bool(std::string_view key) const noexcept {
  auto v = find_value(text_, key);
  if (!v)
    return std::nullopt;
  // Exact, case-sensitive match on Python's True/False -- not "1"/"0", not
  // case-insensitive. A typo (e.g. "true") is a miss, not silently accepted.
  if (*v == "True")
    return true;
  if (*v == "False")
    return false;
  return std::nullopt;
}

std::optional<std::string_view> Pon::get_str(std::string_view key) const noexcept {
  // No "None" test here: find_value already reports it as a miss, for every
  // accessor at once (see the comment there).
  auto v = find_value(text_, key);
  if (!v)
    return std::nullopt;
  // render_pon (python/utils/pon.py) emits every str value through repr(),
  // which -- for every string this grammar carries -- is exactly one pair of
  // surrounding single quotes and nothing escaped (render_pon raises at build
  // time otherwise). Recovering the original string is therefore always this
  // plain two-character strip, never a general repr unescaper.
  if (v->size() >= 2 && v->front() == '\'' && v->back() == '\'')
    return v->substr(1, v->size() - 2);
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

}
