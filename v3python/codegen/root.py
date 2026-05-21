# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Root of the Generation process

from pathlib import Path, PurePath
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import subprocess
from ..rules import (
    kernels as triton_kernels,
    operators as dispatcher_operators,
    affine_kernels,
)
from ..affine import (
    AffineKernelDescription,
    SlimAffineKernelDescription,
)
from .kernel import KernelShimGenerator
from .affine import AffineGenerator
from .slim_affine import SlimAffineGenerator
from .operator import OperatorGenerator
from ..utils import (
    LazyFile,
    RegistryRepository,
    log,
)
from .common import (
    hsaco_dir,
    hsaco_ondisk_name,
    hsaco_inaks2_name,
)
from ..gpu_targets import AOTRITON_ARCH_TO_DIRECTORY
import sys
import os
import re
import yaml

# DO NOT USE Path.absolute(), which does not resolve '..' in the path
REL_PYTHON = Path(os.path.abspath(sys.executable)).relative_to(Path(sys.exec_prefix))

class StrMatcher(object):
    def __init__(self, pattern):
        self._pattern = pattern

    def match(self, value) -> bool:
        return value == self._pattern

class GlobMatcher(StrMatcher):
    def match(self, value) -> bool:
        return PurePath(value).match(self._pattern)

class RegexMatcher(object):
    def __init__(self, pattern):
        self._pattern = re.compile(pattern)

    def match(self, value) -> bool:
        return self._pattern.match(value) is not None

class RuleMatcher(object):
    def __init__(self, name, goal, matchers):
        self._matchers = matchers
        self._name = name
        self._goal = goal

    def match(self, f: "Functional"):
        for attr, matcher in self._matchers:
            value = getattr(f, attr)
            matched = matcher.match(value)
            if not matched:
                return None
        return self._name, self._goal

class RootGenerator(object):
    def __init__(self, args):
        self._args = args
        if args.alt_venv_config:
            with open(args.alt_venv_config) as yamlfile:
                self._altrules_dict = yaml.load(yamlfile, Loader=yaml.Loader)
        else:
            self._altrules_dict = {}
        self._load_altwheel_config(self._altrules_dict)

    def generate(self):
        if self._args.selective:
            self.do_generate()
        else:
            self.launch_workers()

    def do_generate(self):
        args = self._args
        sel = args.selective  # Path or None
        if sel is not None:
            sel_kind, sel_family, sel_name = sel.parts

        hsaco_for_kernels = []
        asms_for_kernels = []
        shims = []
        # (arch, op_name, lut_value) -> count
        all_lut_stats: dict[tuple, int] = {}
        # (arch, op_name) -> {'trivial': N, 'non_trivial': N}
        all_trivial_stats: dict[tuple, dict[str, int]] = {}

        ops = dispatcher_operators
        if sel is not None and sel_kind != 'op':
            ops = []
        elif sel is not None:
            ops = [op for op in ops if op.FAMILY == sel_family and op.NAME == sel_name]
        for op in ops:
            opg = OperatorGenerator(self._args, op, parent_repo=None)
            opg.generate()
            shims += opg.shim_files
            op_name = op.NAME
            for (arch, _, lut_value), count in opg.lut_stats.items():
                key = (arch, op_name, lut_value)
                all_lut_stats[key] = all_lut_stats.get(key, 0) + count
            for (arch, _), ts in opg.trivial_stats.items():
                ak = (arch, op_name)
                d = all_trivial_stats.setdefault(ak, {'trivial': 0, 'non_trivial': 0})
                d['trivial']     += ts['trivial']
                d['non_trivial'] += ts['non_trivial']
        self._print_lut_stats(all_lut_stats, all_trivial_stats)

        kerns = triton_kernels
        if sel is not None and sel_kind != 'triton':
            kerns = []
        elif sel is not None:
            kerns = [k for k in kerns if k.FAMILY == sel_family and k.NAME == sel_name]
        for k in kerns:
            ksg = KernelShimGenerator(self._args, k, parent_repo=None)
            ksg.generate()
            hsacos = ksg.this_repo.get_data('hsaco')
            hsaco_for_kernels.append((k, hsacos))
            shims += ksg.shim_files

        # TODO: Fix this for Windows
        # On Windows, you get "KeyError: 'validator_function'"
        # See discussion in https://discord.com/channels/1239631572886491286/1401853302139912222/1401862203845378201
        affs = affine_kernels
        if sel is not None and sel_kind != 'affine':
            affs = []
        elif sel is not None:
            affs = [ak for ak in affs if ak.FAMILY == sel_family and ak.NAME == sel_name]
        for ak in affs:
            log(lambda : f'{ak.__class__=}')
            if isinstance(ak, SlimAffineKernelDescription):
                aksg = SlimAffineGenerator(self._args, ak, parent_repo=None)
            elif isinstance(ak, AffineKernelDescription):
                aksg = AffineGenerator(self._args, ak, parent_repo=None)
            aksg.generate()
            asms = aksg.this_repo.get_data('asms', return_none=True)
            if asms is not None:
                asms_for_kernels.append((ak, asms))
            shims += aksg.shim_files

        if args.build_for_tuning_second_pass:
            return

        out_dir = args.build_dir / 'Bare.shards' / sel if sel is not None else args.build_dir
        if sel is not None:
            out_dir.mkdir(parents=True, exist_ok=True)

        with LazyFile(out_dir / 'Bare.shim') as shimfile:
            for shim in shims:
                print(shim.absolute().as_posix(), file=shimfile)
        if args.noimage_mode:
            return

        cluster_dict: dict[Path, dict[str, str]] = {}
        with LazyFile(out_dir / 'Bare.compile') as rulefile:
            for kdesc, hsacos in hsaco_for_kernels:
                image_path = hsaco_dir(args.build_dir, kdesc)
                image_path.mkdir(parents=True, exist_ok=True)
                for functional, signatures in hsacos.items():
                    log(lambda : f'{signatures=}')
                    for ksig in signatures:
                        self.write_hsaco(kdesc, image_path, functional, ksig, rulefile)
                    ffp = functional.full_filepack_path
                    cluster_dict.setdefault(ffp, {}).update(
                        {self._absobjfn(image_path, kdesc, ksig): hsaco_inaks2_name(kdesc, ksig)
                         for ksig in signatures}
                    )
        with LazyFile(out_dir / 'Bare.cluster') as clusterfile:
            for ffp, aol_map in cluster_dict.items():
                self.write_cluster(ffp, aol_map, clusterfile)

        '''
        Note: Affine kernel's functionals have residual choices, so it is not
        completely the same with Triton kernel/Interface's functionals

        However, Affine kernel's functionals do not use residual choices to
        compute full_filepack_path, so eventually .hsaco and .co files will be
        consolated into the same .aks2 file, which is intentional.
        '''
        affine_dict: dict[Path, dict[str, str]] = {}

        def parse_rule(asm: str) -> tuple[str, str]:
            assert asm.startswith(':')
            _, inaks2, ondisk = asm.split(':', 2)
            return Path(ondisk).absolute().as_posix(), inaks2

        for akdesc, asm_registry in asms_for_kernels:
            for package_path, asms in asm_registry.items():
                ffp = Path(package_path)
                affine_dict.setdefault(ffp, {}).update(dict(parse_rule(r) for r in asms))
        with LazyFile(out_dir / 'Affine.cluster') as clusterfile:
            for ffp, aol_map in affine_dict.items():
                self.write_cluster(ffp, aol_map, clusterfile)

    def launch_workers(self):
        args = self._args
        items: list[Path] = []
        items += [Path('op')     / op.FAMILY / op.NAME  for op in dispatcher_operators]
        items += [Path('triton') / k.FAMILY  / k.NAME   for k in triton_kernels]
        items += [Path('affine') / ak.FAMILY / ak.NAME  for ak in affine_kernels]

        base_argv = [x for x in sys.argv[1:] if not x.startswith('--selective')]

        def run_one(item: Path):
            cmd = [sys.executable, '-m', 'v3python.generate',
                   '--selective', item.as_posix()] + base_argv
            result = subprocess.run(cmd, capture_output=False)
            if result.returncode != 0:
                raise RuntimeError(f'Worker failed for {item}: exit {result.returncode}')

        with ThreadPoolExecutor() as pool:
            futures = [pool.submit(run_one, item) for item in items]
        for f in futures:
            f.result()  # re-raise any worker exception

        shard_names = ['Bare.shim', 'Bare.compile', 'Bare.cluster', 'Affine.cluster']
        out_files = {name: args.build_dir / name for name in shard_names}
        # Truncate output files before appending
        for path in out_files.values():
            path.unlink(missing_ok=True)
        for item in items:
            shard_dir = args.build_dir / 'Bare.shards' / item
            for name, out_path in out_files.items():
                shard_file = shard_dir / name
                if shard_file.exists():
                    with open(out_path, 'ab') as out, open(shard_file, 'rb') as src:
                        out.write(src.read())

    def _absobjfn(self, path, kdesc, ksig):
        full = path / hsaco_ondisk_name(kdesc, ksig)
        return full.absolute().as_posix()

    def write_hsaco(self, kdesc, path, functional, ksig, rulefile):
        log(lambda : f'{ksig=}')
        srcfn = kdesc.triton_source_path
        venv, python = self._get_venv_and_python(functional)
        print(venv,
              python.as_posix(),
              self._absobjfn(path, kdesc, ksig),
              str(srcfn.absolute()),
              kdesc.triton_kernel_name,
              ksig.num_warps,
              ksig.num_stages,
              ksig.waves_per_eu,
              functional.arch,
              ksig.triton_signature_string,  # Functional is not Triton-specific
              sep=';', file=rulefile)

    def write_cluster(self, ffp, aol_map, clusterfile):
        manifest_path = (self._args.build_dir / 'aotriton.images' / ffp).with_suffix('.nsv')
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, 'w', encoding='utf-8') as mf:
            for abs_path, entry_name in aol_map.items():
                mf.write(abs_path + '\x00' + entry_name + '\x00\n')
        print(*ffp.parts, *aol_map.keys(), sep=';', file=clusterfile)

    # TODO: deprecate this, the generator should return the full path directly
    def _absasmfn(self, asm_rule):
        if asm_rule.startswith(':'):
            return asm_rule
        full = self._args.root_dir / asm_rule
        return full.absolute().as_posix()

    def _load_altwheel_config(self, d: dict):
        venvs = d.get("venvs", {})
        rules = d.get("rules", [])
        self._altwheels = {}
        self._venvpython = {}
        for name, value in venvs.items():
            if value.startswith("python:"):
                # Use the provided Python executable directly
                python_path = value[len("python:"):]
                self._venvpython[name] = Path(python_path).absolute()
            else:
                self._altwheels[name] = Path(value)
                self._venvpython[name] = (self._args.build_dir.parent / "altvenvs" / name / REL_PYTHON).absolute()
        if "default" not in self._venvpython:
            # Use VIRTUAL_ENV environment variable if set (by CMake in v3src/CMakeLists.txt)
            # This ensures we use an existing build environment before falling back to venv
            venv_dir = os.getenv('VIRTUAL_ENV')
            if venv_dir:
                self._venvpython['default'] = (Path(venv_dir) / REL_PYTHON).absolute()
            else:
                self._venvpython['default'] = (self._args.build_dir.parent / 'venv' / REL_PYTHON).absolute()
        self._altrules = [ self._create_rule_function(rule) for rule in rules ]

    def _create_rule_function(self, rule):
        def gen_matcher():
            for key, matcher_factory in zip(["matches", "rmatches", "gmatches"],
                                            [StrMatcher, RegexMatcher, GlobMatcher]):
                for attr, pattern in rule.get(key, {}).items():
                    yield (attr, matcher_factory(pattern))
        venv_name = rule['venv']
        venvpython = self._venvpython[venv_name]
        return RuleMatcher(venv_name, venvpython, list(gen_matcher()))

    def _get_venv_and_python(self, f: 'Interface'):
        for rule in self._altrules:
            matched = rule.match(f)
            if matched is not None:
                return matched
        # We can add an always-true matcher to self._altrules but let's be explicit
        return "default", self._venvpython["default"]

    def _print_lut_stats(self, all_lut_stats: dict, all_trivial_stats: dict):
        def _table(title, header, rows):
            sep = '-' * len(header)
            print(sep)
            print(title)
            print(sep)
            print(header)
            print(sep)
            for row in rows:
                print(row)
            print(sep)

        if all_trivial_stats:
            sorted_pairs = sorted(all_trivial_stats.keys())
            col_arch = max(len('arch'),    max(len(p[0]) for p in sorted_pairs))
            col_op   = max(len('op_name'), max(len(p[1]) for p in sorted_pairs))
            col_t    = len('trivial')
            col_nt   = len('non_trivial')
            header = (f'{"arch":<{col_arch}}  {"op_name":<{col_op}}'
                      f'  {"trivial":>{col_t}}  {"non_trivial":>{col_nt}}')
            rows = [
                f'{arch:<{col_arch}}  {op_name:<{col_op}}'
                f'  {d["trivial"]:>{col_t}}  {d["non_trivial"]:>{col_nt}}'
                for (arch, op_name), d in sorted(all_trivial_stats.items())
            ]
            _table('Op tuning: trivial vs non-trivial functionals', header, rows)

        if all_lut_stats:
            sorted_keys = sorted(all_lut_stats.keys())
            col_arch = max(len('arch'),      max(len(k[0]) for k in sorted_keys))
            col_op   = max(len('op_name'),   max(len(k[1]) for k in sorted_keys))
            col_lut  = max(len('lut_value'), max(len(str(k[2])) for k in sorted_keys))
            col_cnt  = max(len('count'),     max(len(str(all_lut_stats[k])) for k in sorted_keys))
            header = (f'{"arch":<{col_arch}}  {"op_name":<{col_op}}'
                      f'  {"lut_value":>{col_lut}}  {"count":>{col_cnt}}')
            rows = [
                f'{arch:<{col_arch}}  {op_name:<{col_op}}'
                f'  {str(lv) + (" (no backend)" if lv == -1 else ""):>{col_lut}}'
                f'  {all_lut_stats[k]:>{col_cnt}}'
                for k in sorted_keys
                for arch, op_name, lv in [k]
            ]
            _table('Op tuning LUT coverage statistics', header, rows)

    # def write_cluster(self, kdesc, path, functional, signatures, clusterfile):
    #     full_filepack_path = functional.full_filepack_path
    #     print(*full_filepack_path.parts,
    #           end=';', sep=';', file=clusterfile)
    #     path_list = [self._absobjfn(path, kdesc, ksig) for ksig in signatures]
    #     print(*path_list, sep=';', file=clusterfile)
