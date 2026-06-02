#!/usr/bin/env python
# Copyright © 2024-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import sys
from argparse import ArgumentParser
from pathlib import Path
import lzma
from dataclasses import dataclass, fields

desc = """
AOTriton Kernel Storage V2 (AKS2) utility
"""

def parse():
    parser = ArgumentParser(description=desc)
    parser.add_argument("-o", help="Output AKS2 file")
    parser.add_argument("--ignore_json", help="Ignore JSON files", action='store_true')
    parser.add_argument("--hsaco_manifest", required=True,
                        help="NSV file mapping abs HSACO path to its AKS2 entry name")
    args = parser.parse_args()
    return args

AKS2_MAGIC = b'AKS2'

@dataclass
class AKS2_DirectoryEntry:
    shared_memory_size : int = 0
    block_threads : int = 0
    offset : int = 0
    image_size : int = 0
    filename_length : int = 0
    filename : bytes = 0

_AKS2_DirectoryEntry_BaseSize = (len(fields(AKS2_DirectoryEntry)) - 1) * 4

def directory_entry_size(entry):
    return entry.filename_length + _AKS2_DirectoryEntry_BaseSize

def read_manifest(manifest_path: str) -> dict[str, str]:
    '''Read NUL-separated manifest, return {abs_hsaco_path: aks2_entry_name}.'''
    text = Path(manifest_path).read_text(encoding='utf-8')
    result = {}
    for line in text.splitlines():
        parts = line.split('\x00')
        if len(parts) >= 2 and parts[0]:
            result[parts[0]] = parts[1]
    return result

def load_hsaco(hsaco_path: str, entry_name: str, offset: int, ignore_json: bool):
    hsaco = Path(hsaco_path)
    filename = entry_name.encode('utf-8')
    with open(hsaco, 'rb') as f:
        blob = f.read()
    if ignore_json:
        shared_memory_size = 0
        block_threads = 0
    else:
        with open(hsaco.with_suffix('.json')) as jf:
            j = json.load(jf)
            if len(blob) > 0:
                shared_memory_size = j['shared']
                block_threads = j['num_warps'] * j['warp_size']
            else:
                shared_memory_size = 0
                block_threads = 0
                assert j['compile_status'] != 'Complete'
    entry = AKS2_DirectoryEntry(shared_memory_size=shared_memory_size,
                                block_threads=block_threads,
                                offset=offset,
                                image_size=len(blob),
                                filename_length=len(filename)+1,
                                filename=filename)
    return entry, blob

def u32(val):
    return val.to_bytes(4, byteorder=sys.byteorder, signed=False)

def write_u32(val, f):
    f.write(u32(val))

def directory_entry_blob(entry):
    blob = b"".join([u32(entry.shared_memory_size),
                     u32(entry.block_threads),
                     u32(entry.offset),
                     u32(entry.image_size),
                     u32(entry.filename_length),
                     entry.filename,
                     b'\0'])
    assert len(blob) == directory_entry_size(entry), f'blob size {len(blob)} != directory entry size {directory_entry_size(entry)}'
    return blob

class AKS2(object):
    def __init__(self):
        self.total_uncompressed_size : int = 0
        self.number_of_kernels : int = 0
        self.directory_size : int = 0
        self.directory = []
        self.hsaco_blobs = []
        self.current_offset = 0

    def load(self, manifest: dict[str, str], ignore_json: bool):
        self.number_of_kernels = len(manifest)
        for hsaco_path, entry_name in manifest.items():
            entry, blob = load_hsaco(hsaco_path, entry_name, self.current_offset, ignore_json)
            self.current_offset += len(blob)
            self.directory.append(entry)
            self.hsaco_blobs.append(blob)
        self.directory_size = sum([directory_entry_size(e) for e in self.directory])
        self.total_uncompressed_size = self.directory_size + sum([len(blob) for blob in self.hsaco_blobs])

    def write(self, f):
        f.write(AKS2_MAGIC)
        write_u32(self.total_uncompressed_size, f)
        write_u32(self.number_of_kernels, f)
        write_u32(self.directory_size, f)
        lzc = lzma.LZMACompressor()
        for entry in self.directory:
            entry_blob = directory_entry_blob(entry)
            f.write(lzc.compress(entry_blob))
        for blob in self.hsaco_blobs:
            f.write(lzc.compress(blob))
        f.write(lzc.flush())

def do_create(args):
    manifest = read_manifest(args.hsaco_manifest)
    if not manifest:
        print("No input file, exit")
        return
    aks2 = AKS2()
    aks2.load(manifest, args.ignore_json)
    with open(Path(args.o).with_suffix('.aks2'), "wb") as f:
        aks2.write(f)

def main():
    args = parse()
    # Leave do_xxx() for other operation modes like
    #   aks2 -l xx: list content from xx
    do_create(args)

if __name__ == "__main__":
    main()
