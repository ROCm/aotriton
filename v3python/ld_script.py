#!/usr/bin/env python

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

def write_linker_script(args):
    string = f"AOTriton {args.major}.{args.minor}.{args.patch}"
    string = string.encode('ascii')
    with open(args.o, 'w') as f:
        print("""SECTIONS
{
  .comment : {
    KEEP(*(.comment))
    BYTE(0)""", file=f)
        for c in string:
            print(f"    BYTE(0x{c:02x})", file=f)
        print("""    BYTE(0)
  }
}
INSERT AFTER .comment;
""", file=f)

def main():
    args = parse()
    write_linker_script(args)

if __name__ == "__main__":
    main()
