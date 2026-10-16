#!/usr/bin/env python
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import sys
import lzma
from argparse import ArgumentParser
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from .aks2 import AKS2_MAGIC, AKS2_DirectoryEntry, _AKS2_DirectoryEntry_BaseSize

desc = """
List or extract AOTriton Kernel Storage V2 (AKS2) files, like unzip.
"""

def parse():
    parser = ArgumentParser(description=desc)
    parser.add_argument("aks2", help="AKS2 file")
    parser.add_argument("files", nargs='*',
                        help="Only process entries matching these names (shell wildcards allowed)")
    parser.add_argument("-l", help="List entries instead of extracting", action='store_true')
    parser.add_argument("-d", help="Extract files into this directory", default='.')
    return parser.parse_args()

def u32(blob, pos):
    return int.from_bytes(blob[pos:pos+4], byteorder=sys.byteorder, signed=False)

def read_aks2(path):
    '''Return (directory entries, blob region) of an AKS2 file.'''
    raw = Path(path).read_bytes()
    if raw[:4] != AKS2_MAGIC:
        sys.exit(f'{path}: not an AKS2 file')
    total_size, nkernels, directory_size = u32(raw, 4), u32(raw, 8), u32(raw, 12)
    data = lzma.decompress(raw[16:])
    if len(data) != total_size:
        sys.exit(f'{path}: uncompressed size {len(data)} != header {total_size}')
    directory = []
    pos = 0
    for _ in range(nkernels):
        vals = [u32(data, pos + 4 * i) for i in range(5)]
        name_start = pos + _AKS2_DirectoryEntry_BaseSize
        name = data[name_start:name_start + vals[4] - 1]  # strip trailing NUL
        directory.append(AKS2_DirectoryEntry(*vals, filename=name.decode('utf-8')))
        pos = name_start + vals[4]
    assert pos == directory_size, f'directory size {pos} != header {directory_size}'
    return directory, data[directory_size:]

def safe_path(extdir, name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts:
        sys.exit(f'refusing to extract unsafe path: {name}')
    return Path(extdir) / p

def main():
    args = parse()
    directory, images = read_aks2(args.aks2)
    if args.files:
        directory = [e for e in directory if any(fnmatchcase(e.filename, pat) for pat in args.files)]
        if not directory:
            sys.exit(f'{args.aks2}: no entries match {" ".join(args.files)}')
    if args.l:
        print(f'{"Length":>10} {"Shared":>8} {"Threads":>7}  Name')
        for e in directory:
            print(f'{e.image_size:>10} {e.shared_memory_size:>8} {e.block_threads:>7}  {e.filename}')
        print(f'{sum(e.image_size for e in directory):>10} {"":>8} {"":>7}  {len(directory)} file(s)')
        return
    for e in directory:
        out = safe_path(args.d, e.filename)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(images[e.offset:e.offset + e.image_size])
        print(f'  inflating: {out}')

if __name__ == "__main__":
    main()
