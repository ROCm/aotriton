# Copyright © 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
KernelSignature — the perf + compiler-option signature of a single compiled kernel
instance (one Functional bound to a perf config). Pure formatting/hashing over a
Functional; no description-class dependency. Relocated from the legacy
aotriton.kernel.ksignature during the ATI migration.
"""

from functools import cached_property
import hashlib

from aotriton.utils import log

COMPACT_COMPILER_OPTIONS = {
    'waves_per_eu' : 'wave',
    'num_warps': 'warp',
    'num_stages': 'stg',
}
COMPILER_OPTIONS = [ 'waves_per_eu', 'num_warps', 'num_stages' ]
DEFAULT_COPT = [ 2, 4, 1 ]
COPT_WAVES_INDEX = 0
COPT_NWARPS_INDEX = 1
COPT_NSTAGES_INDEX = 2
assert COMPILER_OPTIONS[COPT_WAVES_INDEX] == 'waves_per_eu'
assert COMPILER_OPTIONS[COPT_NWARPS_INDEX] == 'num_warps'
assert COMPILER_OPTIONS[COPT_NSTAGES_INDEX] == 'num_stages'

class KernelSignature(object):

    def __init__(self, f : 'Functional', perf_struct, copt_values : list,
                 gfx1250_warp_workaround : bool = True):
        # perf_struct: a synthesized perf-struct INSTANCE (specs/tune.PerfStructBase),
        # one bind row whose fields are settled TypedChoices. Iterated via .items().
        # Perf params are non-conditional constexprs, so there is nothing to settle.
        self._functional = f
        self._perfs = perf_struct
        self._copts = list(copt_values)
        # TODO: Remove when gfx1250 database is created.
        # gfx1250 falls back to gfx942's tuning database, but with 2x number
        # of warps to avoid compiler issues. This does NOT apply when building
        # for tuning: those signatures come from gen_autotune_configs, which
        # already emits the intended num_warps for gfx1250.
        if gfx1250_warp_workaround and f.arch == 'gfx1250':
            self._copts[COPT_NWARPS_INDEX] *= 2

    @property
    def perf_cdict(self):
        return { name : tc.json_value for name, tc in self._perfs.items() }

    @property
    def perf_compact_dict(self):
        return { name : tc for name, tc in self._perfs.items() }

    @property
    def copt_dict(self):
        return { oname : v for oname, v in zip(COMPILER_OPTIONS, self._copts) }

    @cached_property
    def perf_section(self) -> str:
        parts = []
        for name, tc in self._perfs.items():
            parts.append(f'{name}={tc.testrun_entry_signature}')
        return ';'.join(parts)

    @cached_property
    def copt_section(self) -> str:
        return ';'.join(f'{k}={v}' for k, v in self.copt_dict.items())

    @cached_property
    def hsaco_entry_name(self) -> str:
        return (
            f';;#F;{self._functional.unified_signature}'
            f';;#P;{self.perf_section}'
            f';;#CO;{self.copt_section}'
            f';;arch={self._functional.arch}'
        )


    def blake2b_hash(self, package_path):
        entry = self.hsaco_entry_name
        raw = (package_path + entry).encode('utf-8')
        h = hashlib.blake2b(raw, digest_size=8)
        return h.hexdigest(), raw

    @property
    def num_warps(self):
        return self._copts[COPT_NWARPS_INDEX]

    @property
    def num_stages(self):
        return self._copts[COPT_NSTAGES_INDEX]

    @property
    def waves_per_eu(self):
        return self._copts[COPT_WAVES_INDEX]

    @property
    def triton_signature_string(self):
        complete_dict = self._functional.build_complete_tc_dict()
        for aname, tc in self._perfs.items():
            complete_dict[aname] = tc
        kdesc = self._functional.meta_object
        ARGUMENTS = kdesc.ARGUMENTS
        log(lambda: f'{kdesc.NAME=}')
        log(lambda: f'{kdesc.ARGUMENTS=}')
        log(lambda: f'{complete_dict=}')
        return ', '.join([str(complete_dict[aname].triton_compile_signature) for aname in ARGUMENTS])
