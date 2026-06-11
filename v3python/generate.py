# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
from .rules import (
    kernels as triton_kernels,
    operators as dispatcher_operators,
)
from .codegen import (
    RootGenerator
)
from .utils import LazyFile
import argparse
from pathlib import Path
from .gpu_targets import AOTRITON_SUPPORTED_GPUS

SKIPPED_LUT_CHECK = os.getenv('AOTRITON_SKIP_LUT_CHECK', default='').split(',')

def should_raise_for_lut(args, f : 'Functional'):
    if args.lut_sanity_check:
        return False
    iface = f.meta_object
    if iface.UNTYPED_FULL_NAME in SKIPPED_LUT_CHECK:
        return False
    return True

def parse():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--target_gpus", type=str, default=None, nargs='+', choices=AOTRITON_SUPPORTED_GPUS,
                   help="Ahead of Time (AOT) Compile Architecture.")
    p.add_argument("--build_dir", type=Path, default='build/', help="build directory")
    p.add_argument("--alt_venv_config", type=Path, default=None, help="alternative venv YAML config file")
    p.add_argument("--root_dir", type=Path, default='.', help="Root of repository directory")
    p.add_argument("--archive_only", action='store_true', help='Only generate archive library instead of shared library. No linking with dependencies.')
    p.add_argument("--library_suffix", type=str, default='', help="Add suffix to the library name 'aotriton' to avoid symbol conflicts")
    # Always True
    # p.add_argument("--bare_mode", action='store_true', help="Instead of generating a proper Makefile, only generate a list of source files and leave the remaining tasks to cmake.")
    p.add_argument("--noimage_mode", action='store_true', help="Expect the GPU kernel images are built separately.")
    p.add_argument("--build_for_tuning", action='store_true', help="Include all GPU kernels in the dispatcher for performance tuning.")
    p.add_argument("--build_for_tuning_second_pass", action='store_true', help="Only re-generate autotune files. Ignore HSACO kernels in the autotune files that failed to compile.")
    p.add_argument("--build_for_tuning_but_skip_kernel", type=str, default='', nargs='*',
                   help="Excluse certain GPU kernels for performance tuning when --build_for_tuning=True.")
    # Always True
    # p.add_argument("--generate_cluster_info", action='store_true', help="Generate Bare.functionals for clustering.")
    p.add_argument("--selective", type=str, default=None,
                   help="Only generate the item matching this path or glob pattern, "
                        "e.g. flash/triton/attn_fwd or 'flash/affine/*'. "
                        "Paths containing <family>/affine MUST be a glob pattern ending in *; "
                        "use quotes to prevent shell expansion of *.")
    p.add_argument("--verbose", action='store_true', help="Print debugging messages")
    p.add_argument("--lut_sanity_check", action='store_true', help="By default, an exception will ba raised when any the look up table (LUT) is broken. With this option the exception is not raised, and diagnose information is printed for developers to re-run the tuning script in order to fix the database.")
    p.add_argument("--sanity_check_only", action='store_true', help="Run --lut_sanity_check but suppress all code generation. No files are written. Implies --lut_sanity_check.")
    p.add_argument("--sancheck", action='store_true', help="Validate the ATI descriptions (completeness, override scope, no arg in two axes, derive-target resolution, operator merge cycles) and exit. Read-only: no code is generated. Exit non-zero if any error.")
    # Handled by CMake
    # p.add_argument("--timeout", type=float, default=8.0, help='Maximal time the compiler can run. Passing < 0 for indefinite. No effect in bare mode (handled separately)')
    args = p.parse_args()
    if args.sanity_check_only:
        args.lut_sanity_check = True
        LazyFile.suppress_writes = True
    args._sanity_check_exceptions = []
    args.build_for_tuning_but_skip_kernel = args.build_for_tuning_but_skip_kernel
    args._object_file_registry = []
    args._should_raise_for_lut = lambda f : should_raise_for_lut(args, f)
    # print(args)
    return args

def run_sancheck() -> int:
    """Validate all ATI descriptions; print every error; return process exit code."""
    import sys
    from .rules import kernels as triton_kernels, operators as dispatcher_operators
    from .template_instantiation.tools import sancheck_report
    ok, errors = sancheck_report(kernels=triton_kernels,
                                 operators=dispatcher_operators)
    if ok:
        print('ATI sancheck: OK (no errors)')
        return 0
    print(f'ATI sancheck: {len(errors)} error(s):', file=sys.stderr)
    for e in errors:
        print(f'  {e}', file=sys.stderr)
    return 1

def main():
    args = parse()
    if args.sancheck:
        import sys
        sys.exit(run_sancheck())
    gen = RootGenerator(args)
    gen.generate()
    for e in args._sanity_check_exceptions:
        raise e

if __name__ == '__main__':
    main()
