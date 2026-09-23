// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/_internal/lszip.h>
#include <aotriton/_internal/fd.h>
#include <aotriton/_internal/log.h>
#include <cstring>
#include <algorithm>
#include <vector>

// ZIP format constants (PKWARE spec §4.3)
static constexpr uint32_t ZIP_CENTRAL_DIR_SIG = 0x02014b50;
static constexpr uint32_t ZIP_EOCD_SIG        = 0x06054b50;

static constexpr int EOCD_MIN_SIZE   = 22;
static constexpr int EOCD_SEARCH_MAX = 65536 + EOCD_MIN_SIZE;

static inline uint16_t le16(const uint8_t* p) {
  return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}
static inline uint32_t le32(const uint8_t* p) {
  return static_cast<uint32_t>(p[0])
       | (static_cast<uint32_t>(p[1]) << 8)
       | (static_cast<uint32_t>(p[2]) << 16)
       | (static_cast<uint32_t>(p[3]) << 24);
}

namespace AOTRITON_NS {

bool
lszip(fd_t fd,
      std::function<void(std::string_view, uint64_t, uint64_t)> visitor) {
  off_t file_size = fd_seek(fd, 0, SEEK_END);
  if (file_size < EOCD_MIN_SIZE)
    return false;

  int search_len = static_cast<int>(std::min<off_t>(file_size, EOCD_SEARCH_MAX));
  std::vector<uint8_t> tail(search_len);
  fd_seek(fd, file_size - search_len, SEEK_SET);
  if (fd_read(fd, tail.data(), search_len) != search_len)
    return false;

  int eocd_off = -1;
  for (int i = search_len - EOCD_MIN_SIZE; i >= 0; --i) {
    if (le32(tail.data() + i) == ZIP_EOCD_SIG) {
      eocd_off = i;
      break;
    }
  }
  if (eocd_off < 0)
    return false;

  const uint8_t* eocd    = tail.data() + eocd_off + 4;
  uint16_t total_entries = le16(eocd + 6);
  uint32_t cd_size       = le32(eocd + 8);
  uint32_t cd_offset     = le32(eocd + 12);

  // Reject corrupted EOCD pointing past file_size or describing a CD that
  // would overflow allocation / read past end-of-file.
  if (static_cast<uint64_t>(cd_offset) + cd_size > static_cast<uint64_t>(file_size))
    return false;

  std::vector<uint8_t> cd(cd_size);
  fd_seek(fd, static_cast<off_t>(cd_offset), SEEK_SET);
  if (fd_read(fd, cd.data(), cd_size) != static_cast<ssize_t>(cd_size))
    return false;

  const uint8_t* p   = cd.data();
  const uint8_t* end = p + cd_size;

  for (uint16_t i = 0; i < total_entries; ++i) {
    if (p + 46 > end || le32(p) != ZIP_CENTRAL_DIR_SIG)
      return false;
    uint16_t gp_flag      = le16(p + 8);
    uint16_t comp_method  = le16(p + 10);
    uint32_t comp_size    = le32(p + 20);
    uint16_t fname_len    = le16(p + 28);
    uint16_t extra_len    = le16(p + 30);
    uint16_t comment_len  = le16(p + 32);
    uint32_t local_offset = le32(p + 42);

    // Bounds check the variable-length tail (name + extra + comment) before
    // reading the entry name — fname_len/extra_len/comment_len all come from
    // file bytes and a truncated central directory must not produce an OOB read.
    if (p + 46 + static_cast<size_t>(fname_len)
              + static_cast<size_t>(extra_len)
              + static_cast<size_t>(comment_len) > end)
      return false;

    // flatzip is STORED-only; reject anything compressed or encrypted so the
    // caller never gets back offsets that point at non-raw data.
    if (comp_method != 0 || (gp_flag & 0x0001) != 0) {
      AOTRITON_LOG(LOG_DEBUG,
                   "lszip: rejecting non-STORED/encrypted entry %.*s (comp_method=%d, gp_flag=%#06x)",
                   int(fname_len), reinterpret_cast<const char*>(p + 46),
                   int(comp_method), unsigned(gp_flag));
      return false;
    }

    std::string_view entry_name(reinterpret_cast<const char*>(p + 46), fname_len);

    // Local header's extra field length may differ from central directory's.
    uint8_t lhdr[4];
    fd_seek(fd, static_cast<off_t>(local_offset) + 26, SEEK_SET);
    if (fd_read(fd, lhdr, 4) != 4)
      return false;
    uint16_t local_fname_len = le16(lhdr + 0);
    uint16_t local_extra_len = le16(lhdr + 2);
    uint64_t data_offset = static_cast<uint64_t>(local_offset) + 30
                           + local_fname_len + local_extra_len;

    visitor(entry_name, data_offset, static_cast<uint64_t>(comp_size));

    p += 46 + fname_len + extra_len + comment_len;
  }
  return true;
}

// Considerations about Zip Directory Loading
//
// PackedKernel::open() calls lszip() to populate an InnerMap
// (unordered_map<string, CachedEntry>) on first access to a ZIP file.
// Each insert allocates a map node (key + value + next-pointer) and a heap
// string for the entry name (unified_signature strings exceed SSO length).
// reserve(n) on the map before insertion eliminates O(log n) bucket-array
// rehashes but does not affect per-node allocations.
//
// A fully arena-based design would eliminate all per-node allocs:
//   1. First pass over ZIP central directory: count total_entries and sum of
//      entry-name byte lengths.
//   2. Allocate one contiguous chunk sized to hold all string data, all
//      CachedEntry structs, all map nodes, and the bucket array
//      (~next_power_of_2(n) * sizeof(void*)).
//   3. Use a monotonic bump allocator over the chunk for the map and its nodes.
//   4. Key the map on string_view pointing into the chunk (no owned string).
//   5. A wrapper class ties chunk lifetime to the map.
//
// This is functionally equivalent to std::pmr::monotonic_buffer_resource +
// std::pmr::unordered_map<std::pmr::string, CachedEntry>, with the advantage
// of exact buffer sizing from the two-pass count instead of a guessed initial
// size.  The remaining portability obstacle is that map node size
// (std::_Hash_node<...>) is implementation-defined, requiring either a fudge
// factor or a runtime probe.  Given that InnerMap is populated once per ZIP
// at library load time and n is typically < 500, the per-node allocation
// overhead is not measurable in practice, so the added complexity is not
// currently justified.

} // namespace AOTRITON_NS
