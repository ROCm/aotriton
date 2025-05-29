# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from ..base import Functional
from ..utils import log
import hashlib

# Move to a dedicated file
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

    def __init__(self, f : Functional, perf_values : 'list[Bind]', copt_values : list):
        self._functional = f
        tc_dict = f.build_tc_dict()
        for bind in perf_values:
            bind.settle_unresolved(tc_dict)
        self._perfs = perf_values
        self._copts = list(copt_values)

    def functional(self):
        return self._functional

    @property
    def perf_cdict(self):
        return { aname : tc.json_value for bind in self._perfs for aname, tc in bind }

    @property
    def perf_compact_dict(self):
        return { bind.name : bind.get_typed_value(bind.name) for bind in self._perfs }

    @property
    def perf_signature(self):
        kdesc = self._functional.meta_object
        # TODO: Add prefix?
        lp = [str(p.value) for p in self._perfs]
        psel = '_'.join([x for x in lp if x is not None])
        return psel

    @property
    def copt_dict(self):
        kdesc = self._functional.meta_object
        return { oname : v for oname, v in zip(COMPILER_OPTIONS, self._copts) }

    @property
    def copt_signature(self):
        lc = [f"{COMPACT_COMPILER_OPTIONS[oname]}{v}" for oname, v in self.copt_dict.items()]
        return '_'.join(lc)

    @property
    def both_signature(self):
        perf = self.perf_signature
        copt = self.copt_signature
        return '__P__' + perf + '__CO__' + copt

    @property
    def full_compact_signature(self):
        return self._functional.compact_signature_noarch + self.both_signature + '--Arch_' + self._functional.arch

    def blake2b_hash(self, package_path):
        raw = package_path.encode('utf-8')
        s = self.both_signature + '--Arch_' + self._functional.arch
        raw += s.encode('utf-8')
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
        for perf in self._perfs:
            for aname, tc in perf:
                complete_dict[aname] = tc
        kdesc = self._functional.meta_object
        ARGUMENTS = kdesc.ARGUMENTS
        log(lambda: f'{kdesc.NAME=}')
        log(lambda: f'{kdesc.ARGUMENTS=}')
        log(lambda: f'{kdesc.TYPE_CHOICES=}')
        log(lambda: f'{kdesc.FEAT_CHOICES=}')
        log(lambda: f'{complete_dict=}')
        return ', '.join([str(complete_dict[aname].triton_compile_signature) for aname in ARGUMENTS])
