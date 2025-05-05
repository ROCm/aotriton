# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from .rules import (
    kernels as triton_kernels,
    operators as dispatcher_operators,
)
from .codegen.registry import (
    ClusterRegistry,
)
from .codegen import (
    RootGenerator
)
import argparse
from pathlib import Path
from .gpu_targets import AOTRITON_SUPPORTED_GPUS

def parse():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--target_gpus", type=str, default=None, nargs='+', choices=AOTRITON_SUPPORTED_GPUS,
                   help="Ahead of Time (AOT) Compile Architecture.")
    p.add_argument("--build_dir", type=Path, default='build/', help="build directory")
    p.add_argument("--archive_only", action='store_true', help='Only generate archive library instead of shared library. No linking with dependencies.')
    p.add_argument("--library_suffix", type=str, default='', help="Add suffix to the library name 'aotriton' to avoid symbol conflicts")
    # Always True
    # p.add_argument("--bare_mode", action='store_true', help="Instead of generating a proper Makefile, only generate a list of source files and leave the remaining tasks to cmake.")
    p.add_argument("--noimage_mode", action='store_true', help="Expect the GPU kernel images are built separately.")
    p.add_argument("--build_for_tuning", action='store_true', help="Include all GPU kernels in the dispatcher for performance tuning.")
    p.add_argument("--build_for_tuning_but_skip_kernel", type=str, default='', nargs='*',
                   help="Excluse certain GPU kernels for performance tuning when --build_for_tuning=True.")
    # Always True
    # p.add_argument("--generate_cluster_info", action='store_true', help="Generate Bare.functionals for clustering.")
    p.add_argument("--verbose", action='store_true', help="Print debugging messages")
    p.add_argument("--lut_sanity_check", action='store_true', help="Do not raise exceptions when the look up table (lut) is incomplete.")
    # Handled by CMake
    # p.add_argument("--timeout", type=float, default=8.0, help='Maximal time the compiler can run. Passing < 0 for indefinite. No effect in bare mode (handled separately)')
    args = p.parse_args()
    args._sanity_check_exceptions = []
    args.build_for_tuning_but_skip_kernel = args.build_for_tuning_but_skip_kernel
    args._cluster_registry = ClusterRegistry()
    args._object_file_registry = []
    # print(args)
    return args

def main():
    args = parse()
    gen = RootGenerator(args)
    gen.generate()
    for e in args._sanity_check_exceptions:
        raise e

if __name__ == '__main__':
    main()
