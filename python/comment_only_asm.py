#!/usr/bin/env python
# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
from pathlib import Path

def parse():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-o', help="Output", type=Path, required=True)
    p.add_argument('major')
    p.add_argument('minor')
    p.add_argument('patch')
    args = p.parse_args()
    return args

def write_assembly(args):
    string = f"AOTriton {args.major}.{args.minor}.{args.patch}"
    with open(args.o, 'w') as f:
        print(f""".section .note.GNU-stack,"",@progbits
.section .comment
msg: .ascii "{string}\\0"
len = . - msg""", file=f)


def main():
    args = parse()
    write_assembly(args)

if __name__ == "__main__":
    main()
