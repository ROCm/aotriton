# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Build-time tool: pack AKS2 blobs into an uncompressed per-kernel ZIP archive.
#
# The ZIP is stored (no compression) so entries are accessible via bare
# lseek+read at fixed offsets, which massively simplifies the C++ reader.
#
# Usage:
#   python -m aotriton.flatzip \
#     --manifest <kernel>.nsv \
#     -o <kernel>.zip

import argparse
import struct
import os
import sys
from pathlib import Path

# ZIP format constants (PKWARE spec §4.3)
_LOCAL_HEADER_SIG   = b'PK\x03\x04'
_CENTRAL_DIR_SIG    = b'PK\x01\x02'
_EOCD_SIG           = b'PK\x05\x06'
_COMPRESSION_STORED = 0
# 2.0 is over-spec for STORED (which only needs 1.0 / version 10) but matches
# what most ZIP writers in the wild emit, so consumers don't trip on a value
# below their hardcoded minimum.
_VERSION_NEEDED     = 20
_VERSION_MADE_BY    = 20

# All sizes/offsets in the headers below are packed as 32-bit unsigned ('I').
# This archive intentionally does NOT use ZIP64: a cumulative offset or entry
# size exceeding 4 GiB will raise struct.error at build time (loud failure,
# not silent truncation). The STORED layout means decompressed == compressed,
# so the ceiling is the sum of all .aks2 sizes packed into one .zip.
# Bit 11 (0x0800) tells ZIP readers to decode entry names as UTF-8 (PKWARE §4.4.4)
_GP_FLAG_UTF8       = 0x0800


def _encode_name(name: str) -> bytes:
    return name.encode('utf-8')


def _local_file_header(name_bytes: bytes, data_size: int, crc32: int) -> bytes:
    return struct.pack(
        '<4sHHHHHIIIHH',
        _LOCAL_HEADER_SIG,
        _VERSION_NEEDED,
        _GP_FLAG_UTF8,      # general purpose bit flag (UTF-8 name)
        _COMPRESSION_STORED,
        0,                  # last mod time
        0,                  # last mod date
        crc32,
        data_size,          # compressed size == uncompressed for STORED
        data_size,
        len(name_bytes),
        0,                  # extra field length
    ) + name_bytes


def _central_dir_entry(name_bytes: bytes, data_size: int, crc32: int, local_offset: int) -> bytes:
    return struct.pack(
        '<4sHHHHHHIIIHHHHHII',
        _CENTRAL_DIR_SIG,
        _VERSION_MADE_BY,
        _VERSION_NEEDED,
        _GP_FLAG_UTF8,      # general purpose bit flag (UTF-8 name)
        _COMPRESSION_STORED,
        0,                  # last mod time
        0,                  # last mod date
        crc32,
        data_size,          # compressed size
        data_size,          # uncompressed size
        len(name_bytes),
        0,                  # extra field length
        0,                  # file comment length
        0,                  # disk number start
        0,                  # internal file attributes
        0,                  # external file attributes
        local_offset,
    ) + name_bytes


def _eocd(central_dir_offset: int, central_dir_size: int, num_entries: int) -> bytes:
    return struct.pack(
        '<4sHHHHIIH',
        _EOCD_SIG,
        0,                  # disk number
        0,                  # disk with central dir start
        num_entries,        # entries on this disk
        num_entries,        # total entries
        central_dir_size,
        central_dir_offset,
        0,                  # comment length
    )


def build_zip(manifest_path: Path, output_path: Path) -> None:
    import zlib

    # manifest.nsv: lines of  <abs_aks2_path>\x00<unified_signature>\x00\n
    entries: dict[str, str] = {}
    with open(manifest_path, encoding='utf-8') as mf:
        for line in mf:
            line = line.rstrip('\n')
            parts = line.split('\x00')
            if len(parts) < 2:
                continue
            abs_path, entry_name = parts[0], parts[1]
            if abs_path:
                entries[abs_path] = entry_name  # dedup by abs_path key

    output_path.parent.mkdir(parents=True, exist_ok=True)
    central_dir_entries = []

    with open(output_path, 'wb') as zf:
        for abs_path, entry_name in entries.items():
            data = Path(abs_path).read_bytes()
            crc32 = zlib.crc32(data) & 0xFFFFFFFF
            name_bytes = _encode_name(entry_name)
            local_offset = zf.tell()
            zf.write(_local_file_header(name_bytes, len(data), crc32))
            zf.write(data)
            central_dir_entries.append((name_bytes, len(data), crc32, local_offset))

        central_dir_offset = zf.tell()
        central_dir_size = 0
        for name_bytes, data_size, crc32, local_offset in central_dir_entries:
            entry = _central_dir_entry(name_bytes, data_size, crc32, local_offset)
            zf.write(entry)
            central_dir_size += len(entry)

        zf.write(_eocd(central_dir_offset, central_dir_size, len(central_dir_entries)))


def main():
    parser = argparse.ArgumentParser(description='Pack AKS2 blobs into an uncompressed ZIP archive')
    parser.add_argument('--manifest', required=True, type=Path,
                        help='Path to .nsv manifest (abs_aks2_path\\x00unified_sig per line)')
    parser.add_argument('-o', '--output', required=True, type=Path,
                        help='Output .zip path')
    args = parser.parse_args()

    build_zip(args.manifest, args.output)


if __name__ == '__main__':
    main()
