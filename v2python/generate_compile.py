# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from .rules import kernels as triton_kernels
from .tuning_database import (
    KernelTuningDatabase,
    ARCH_TO_DIRECTORY,
)
from .gpu_targets import AOTRITON_SUPPORTED_GPUS
import io
import shutil
import argparse
import json
from pathlib import Path
from collections import defaultdict
import itertools

SOURCE_PATH = Path(__file__).resolve()
COMPILER = SOURCE_PATH.parent / 'compile.py'

# TODO: unify with generate_shim
def is_tuning_on_for_kernel(args, k : 'KernelDescription'):
    if not args.build_for_tuning:
        return False
    elif k.FULL_KERNEL_NAME in args.build_for_tuning_but_skip_kernel:
        return False
    else:
        return True


class ClusterKernel(object):
    def __init__(self):
        self._registry = []

    def collect_object_file(self, ofd : 'ObjectFileDescription'):
        self._registry.append(ofd)

    def calc_clustering_scheme(self, n_combination):
        cluster_by = {}
        if n_combination == 0:  # No need to change functional(s) to 'Any'
            dic = defaultdict(list)
            for ofd in self._registry:
                fonly = ofd.functional_signature + '_' + ofd.target_arch
                dic[fonly].append(ofd)
            cluster_by[None] = dic
            return cluster_by
        kdesc = self._registry[0]._triton_kernel_desc
        keys = []
        for m in kdesc._func_meta:
            if m.nchoices <= 1:
                continue
            keys.append(m.repr_name)
        for keycomb in itertools.combinations(keys, n_combination):
            cluster_by[keycomb] = defaultdict(list)
        for by, dic in cluster_by.items():
            if isinstance(by, str):
                sans = set([by])
            else:
                sans = set(by)
            for ofd in self._registry:
                ksig = ofd._signature
                fonly = ksig.get_partial_functional_signature(sans) + ksig._gpu
                dic[fonly].append(ofd)
        return cluster_by

class ClusterKernelFamily(object):
    def __init__(self):
        self._registry = defaultdict(ClusterKernel)

    def collect_object_file(self, ofd : 'ObjectFileDescription'):
        self._registry[ofd.SHIM_KERNEL_NAME].collect_object_file(ofd)

    def gen_clusters(self):
        for kernel_name, registry_0 in self._registry.items():
            yield kernel_name, registry_0

class ClusterRegistry(object):
    def __init__(self):
        self._registry = defaultdict(ClusterKernelFamily)

    def collect_object_file(self, ofd : 'ObjectFileDescription'):
        self._registry[ofd.KERNEL_FAMILY].collect_object_file(ofd)

    def gen_clusters(self, n_combination):
        for family, registry_0 in self._registry.items():
            for kernel_name, registry_1 in registry_0.gen_clusters():
                yield family, kernel_name, registry_1.calc_clustering_scheme(n_combination=n_combination)

    def write_clustering_tests(self, f):
        for family, kernel_name, cluster in self.gen_clusters(n_combination=2):
            print(f'mkdir -p {family}/{kernel_name}', file=f)
            for by, clusters in cluster.items():
                bypath = '-'.join(by)
                print(f'mkdir -p {family}/{kernel_name}/{bypath}', file=f)
                for fonly, obj_list in clusters.items():
                    tar = f'{family}/{kernel_name}/{bypath}/{fonly}.tar'
                    print(f'tar cf {tar} ', ' '.join([str(o.obj.absolute()) for o in obj_list]), file=f)
                    print(f'zstd {tar}', file=f)

    def write_bare(self, args, f):
        for family, kernel_name, cluster_bys in self.gen_clusters(n_combination=0):
            # cluster_bys[None]: only cluster psels and copts
            # Experiment shows it is not needed to cluster by one or more functionals.
            # XZ + clustering psels+copts is good enough
            clusters = cluster_bys[None]
            for fonly, obj_list in clusters.items():
                first_obj = obj_list[0]
                dir_arch = ARCH_TO_DIRECTORY[first_obj.target_arch]
                print(dir_arch, family, kernel_name, fonly, end=';', sep=';', file=f)
                path_list = [str(o.obj.absolute()) for o in obj_list]
                print(*path_list, sep=';', file=f)

def parse():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--target_gpus", type=str, default=None, nargs='+', choices=AOTRITON_SUPPORTED_GPUS,
                   help="Ahead of Time (AOT) Compile Architecture. PyTorch is required for autodetection if --targets is missing.")
    p.add_argument("--build_dir", type=str, default='build/', help="build directory")
    p.add_argument("--python", type=str, default=None, help="python binary to run compile.py")
    p.add_argument("--bare_mode", action='store_true', help="Instead of generating a proper Makefile, only generate compiler options and leave the remaining tasks to cmake.")
    p.add_argument("--test_clustering", action='store_true', help="Generate TestClustering.sh to find the optimal clustering scheme.")
    p.add_argument("--generate_cluster_info", action='store_true', help="Generate Bare.functionals for clustering.")
    p.add_argument("--build_for_tuning", action='store_true', help="Build all possible GPU kernels for performance tuning.")
    p.add_argument("--build_for_tuning_but_skip_kernel", type=str, default='', nargs='*',
                   help="Excluse certain GPU kernels for performance tuning when --build_for_tuning=True.")
    p.add_argument("--timeout", type=float, default=8.0, help='Maximal time the compiler can run. Passing < 0 for indefinite. No effect in bare mode (handled separately)')
    # p.add_argument("--autotune_data", type=str, default=None, help="Autotune results generated by tune_flash.py")
    args = p.parse_args()
    if args.test_clustering or args.generate_cluster_info:
        args._cluster_registry = ClusterRegistry()
    # print(args)
    return args

def gen_from_object(args, o : 'ObjectFileDescription', makefile):
    if o.is_functional_disabled():
        return
    if args.test_clustering or args.generate_cluster_info:
        args._cluster_registry.collect_object_file(o)
    if args.bare_mode:
        print(o.obj.absolute(), o.src.absolute(), o.entrance, o.num_warps, o.num_stages, o.waves_per_eu, o.target_arch, o.signature, sep=';', file=makefile)
        return
    print('#', o.human_readable_signature, file=makefile)
    target_fn = f'{o.KERNEL_FAMILY}/gpu_kernel_image.{o.SHIM_KERNEL_NAME}/{o._hsaco_kernel_path.name}'
    print(target_fn, ':', o.src.absolute(), COMPILER.absolute(), file=makefile)
    cmd  = f'LD_PRELOAD=$(LIBHSA_RUNTIME64) {COMPILER} {o.src.absolute()} --kernel_name {o.entrance} -o {o.obj.absolute()}'
    cmd += f' -g 1,1,1 --num_warps {o.num_warps} --num_stages {o.num_stages} --waves_per_eu {o.waves_per_eu}'
    cmd += f" --target '{o.target_arch}'"
    cmd += f" --signature '{o.signature}'"
    cmd += f" --timeout {args.timeout}"
    print('\t', cmd, file=makefile)
    print('', file=makefile)
    return target_fn

def gen_from_kernel(args, k, build_dir, makefile):
    outpath = build_dir / k.KERNEL_FAMILY / f'gpu_kernel_image.{k.SHIM_KERNEL_NAME}'
    outpath.mkdir(parents=True, exist_ok=True)
    target_all = f'compile_{k.SHIM_KERNEL_NAME}'
    all_targets = []
    object_rules = io.StringIO()
    # ktd = None if args.build_for_tuning else KernelTuningDatabase(SOURCE_PATH.parent / 'rules', k)
    tuning = is_tuning_on_for_kernel(args, k)
    ktd = KernelTuningDatabase(build_dir, k, build_for_tuning=tuning)
    if False: # Debugging
        if k.SHIM_KERNEL_NAME == 'attn_fwd':
            assert not ktd.empty
    k.set_target_gpus(args.target_gpus)
    for o in k.gen_all_object_files(outpath, tuned_db=ktd):
        all_targets.append(gen_from_object(args, o, object_rules))
    if not args.bare_mode:
        print(target_all, ': ', end='', file=makefile)
        for t in all_targets:
            print(t, end=' ', file=makefile)
        print('\n\n', file=makefile)
    object_rules.seek(0)
    shutil.copyfileobj(object_rules, makefile)
    return target_all

def main():
    args = parse()
    build_dir = Path(args.build_dir)
    fn = 'Bare.compile' if args.bare_mode else 'Makefile.compile'
    with open(build_dir / fn, 'w') as f:
        if not args.bare_mode:
            print('LIBHSA_RUNTIME64=/opt/rocm/lib/libhsa-runtime64.so\n', file=f)
        makefile_content = io.StringIO()
        per_kernel_targets = []
        try:
            for k in triton_kernels:
                k.set_target_gpus(args.target_gpus)
                per_kernel_targets.append(gen_from_kernel(args, k, build_dir, makefile_content))
        except Exception as e:
            raise e
        if not args.bare_mode:
            print('all: ', end='', file=f)
            for t in per_kernel_targets:
                print(t, end=' ', file=f)
            print('\n', file=f)
        makefile_content.seek(0)
        shutil.copyfileobj(makefile_content, f)
        if not args.bare_mode:
            print('.PHONY: all ', ' '.join(per_kernel_targets), file=f)
    if args.test_clustering:
        archve_sh = 'TestClustering.sh'
        with open(build_dir / archve_sh, 'w') as f:
            args._cluster_registry.write_clustering_tests(f)
    if args.generate_cluster_info:
        with open(build_dir / 'Bare.cluster', 'w') as f:
            args._cluster_registry.write_bare(args, f)

if __name__ == '__main__':
    main()
