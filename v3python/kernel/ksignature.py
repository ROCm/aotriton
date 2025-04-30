# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from ..base import Functional

# Move to a dedicated file
COMPACT_COMPILER_OPTIONS = {
    'waves_per_eu' : 'wave',
    'num_warps': 'warp',
    'num_stages': 'stg',
}

class KernelSignature(object):

    def __init__(self, f : Functional, perf_values : 'list[Argument]', copt_values : list):
        self._functional = f
        self._perfs = perf_values
        self._copts = copt_values

    def functional(self):
        return self._functional

    @property
    def perf_signature(self):
        kdesc = self._functional.meta_object
        # TODO: Add prefix?
        lp = [str(p.value) for p in self._perfs]
        psel = '_'.join([x for x in lp if x is not None])
        return psel

    @property
    def copt_signature(self):
        kdesc = self._functional.meta_object
        lc = [f"{COMPACT_COMPILER_OPTIONS[oname]}{v}" for oname, v in zip(kdesc.COMPILER_OPTIONS, self._copts)]
        return '_'.join(lc)

    @property
    def both_signature(self):
        perf = self.perf_signature
        copt = self.copt_signature
        return '__P__' + psel + '__CO__' + copts

    @property
    def full_compact_signature(self):
        return 'F__' + self._functional.compact_signature + self.both_signature

    @property
    def blake2b_hash(self, package_path):
        raw = package_path.encode('utf-8')
        _, psel, copts = self.compact_signature_components
        s = '__P__' + psel + '__CO__' + copts + '-Gpu-' + self._signature.target_arch
        raw += s.encode('utf-8')
        h = hashlib.blake2b(raw, digest_size=8)
        return h.hexdigest(), raw
