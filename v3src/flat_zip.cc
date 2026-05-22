// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/_internal/flat_zip.h>
#include <cstring>
#include <mutex>
#include <vector>

#if defined(_WIN32)
#include "packed_kernel_win32.h"
#else
#include "packed_kernel_unix.h"
#endif

// ZIP format constants (PKWARE spec §4.3)
static constexpr uint32_t ZIP_LOCAL_HEADER_SIG   = 0x04034b50;
static constexpr uint32_t ZIP_CENTRAL_DIR_SIG     = 0x02014b50;
static constexpr uint32_t ZIP_EOCD_SIG            = 0x06054b50;

// EOCD record layout (excluding signature):
// disk number         2
// central dir disk    2
// entries this disk   2
// total entries       2
// central dir size    4
// central dir offset  4
// comment length      2
static constexpr int EOCD_MIN_SIZE   = 22;  // signature (4) + fixed fields (18)
static constexpr int EOCD_SEARCH_MAX = 65536 + EOCD_MIN_SIZE;

namespace AOTRITON_NS {

std::shared_mutex FlatZip::registry_mutex_;
FlatZip::Registry  FlatZip::registry_;

// Read a little-endian uint16/uint32 from a byte buffer at offset.
static inline uint16_t le16(const uint8_t* p) {
  return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}
static inline uint32_t le32(const uint8_t* p) {
  return static_cast<uint32_t>(p[0])
       | (static_cast<uint32_t>(p[1]) << 8)
       | (static_cast<uint32_t>(p[2]) << 16)
       | (static_cast<uint32_t>(p[3]) << 24);
}
static inline uint64_t le64(const uint8_t* p) {
  uint64_t lo = le32(p), hi = le32(p + 4);
  return lo | (hi << 32);
}

FlatZip::DirectoryMap
FlatZip::parse_central_directory(int fd) {
  DirectoryMap dir;

  // Find EOCD record by scanning backwards from end of file.
  off_t file_size = fd_seek(fd, 0, SEEK_END);
  if (file_size < EOCD_MIN_SIZE)
    return dir;

  int search_len = static_cast<int>(std::min<off_t>(file_size, EOCD_SEARCH_MAX));
  std::vector<uint8_t> tail(search_len);
  fd_seek(fd, file_size - search_len, SEEK_SET);
  if (fd_read(fd, tail.data(), search_len) != search_len)
    return dir;

  // Scan backwards for EOCD signature.
  int eocd_off = -1;
  for (int i = search_len - EOCD_MIN_SIZE; i >= 0; --i) {
    if (le32(tail.data() + i) == ZIP_EOCD_SIG) {
      eocd_off = i;
      break;
    }
  }
  if (eocd_off < 0)
    return dir;

  const uint8_t* eocd = tail.data() + eocd_off + 4;  // skip signature
  // uint16_t disk_num       = le16(eocd + 0);
  // uint16_t cd_disk        = le16(eocd + 2);
  // uint16_t entries_here   = le16(eocd + 4);
  uint16_t total_entries  = le16(eocd + 6);
  uint32_t cd_size        = le32(eocd + 8);
  uint32_t cd_offset      = le32(eocd + 12);

  // Read the central directory.
  std::vector<uint8_t> cd(cd_size);
  fd_seek(fd, static_cast<off_t>(cd_offset), SEEK_SET);
  if (fd_read(fd, cd.data(), cd_size) != static_cast<ssize_t>(cd_size))
    return dir;

  const uint8_t* p   = cd.data();
  const uint8_t* end = p + cd_size;

  for (uint16_t i = 0; i < total_entries; ++i) {
    if (p + 46 > end || le32(p) != ZIP_CENTRAL_DIR_SIG)
      break;
    // Central directory entry fixed fields (after 4-byte signature):
    // version made by     2
    // version needed      2
    // flags               2
    // compression         2
    // mod time            2
    // mod date            2
    // crc32               4
    // compressed size     4
    // uncompressed size   4
    // file name length    2
    // extra field length  2
    // file comment length 2
    // disk number start   2
    // int file attrs      2
    // ext file attrs      4
    // local header offset 4
    uint32_t comp_size    = le32(p + 20);
    uint32_t uncomp_size  = le32(p + 24);
    uint16_t fname_len    = le16(p + 28);
    uint16_t extra_len    = le16(p + 30);
    uint16_t comment_len  = le16(p + 32);
    uint32_t local_offset = le32(p + 42);

    const char* fname_ptr = reinterpret_cast<const char*>(p + 46);
    std::string entry_name(fname_ptr, fname_len);

    // Compute data offset: local header is 30 bytes + fname + extra.
    // We need to read the local header's extra field length separately
    // because it may differ from the central directory's extra field length.
    // Read 2 bytes at local_offset+28 and local_offset+30 for fname/extra lengths.
    uint8_t lhdr[4];
    fd_seek(fd, static_cast<off_t>(local_offset) + 26, SEEK_SET);
    fd_read(fd, lhdr, 4);
    uint16_t local_fname_len  = le16(lhdr + 0);
    uint16_t local_extra_len  = le16(lhdr + 2);
    uint64_t data_offset = static_cast<uint64_t>(local_offset) + 30
                           + local_fname_len + local_extra_len;

    dir[entry_name] = EntryLocation{ data_offset, static_cast<uint64_t>(comp_size) };
    (void)uncomp_size;  // STORED: comp_size == uncomp_size

    p += 46 + fname_len + extra_len + comment_len;
  }

  return dir;
}

void
FlatZip::warm(pstring_view zip_path, int fd) {
  {
    std::shared_lock lock(registry_mutex_);
    if (registry_.contains(zip_path))
      return;
  }
  std::unique_lock lock(registry_mutex_);
  if (registry_.contains(zip_path))
    return;
  registry_[pstring_type(zip_path)] = parse_central_directory(fd);
}

std::optional<FlatZip::EntryLocation>
FlatZip::lookup(pstring_view zip_path, std::string_view entry_name) {
  std::shared_lock lock(registry_mutex_);
  auto outer = registry_.find(zip_path);
  if (outer == registry_.end())
    return std::nullopt;
  auto inner = outer->second.find(entry_name);
  if (inner == outer->second.end())
    return std::nullopt;
  return inner->second;
}

} // namespace AOTRITON_NS
