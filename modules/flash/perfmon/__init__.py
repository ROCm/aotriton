# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
The flash family's perfmon plugin.

Loaded by path by `aotriton.perfmon.registry.load_family_perfmon('flash')`,
like the tune block: `modules/flash` stays a plain directory, not a package.
Exports `PerfDesc`, a `aotriton.perfmon.pdesc.PerfDescription`.

Must not import aotriton.tune, aotriton.pgdef, torch or triton
(python/test/pgdef/test_layering.py): it runs on control hosts with none of
them, and the native runner, not Python, touches the GPU.
"""

import dataclasses

from aotriton.perfmon.pdesc import LLMCase, PerfDescription, PrimeTable
from aotriton.utils.pon import render_pon

from . import entry as _entry
from . import size as _size
from . import tflops as _tflops
from . import workloads as _workloads

IFACES = ('op_attn_fwd', 'op_attn_bwd')

#: Largest seqlen each arch can hold, filled from measurement only. An arch
#: absent here is unconstrained, which fails loudly at run time rather than
#: silently skipping entries.
_MAX_SEQLEN_BY_ARCH: dict[str, int] = {}


class PerfDesc(PerfDescription):
    PROBLEM_COLUMNS = _entry.PROBLEM_COLUMNS
    ENTRY_CLASS = _entry.FlashPerfEntry

    def ifaces(self):
        return IFACES

    def phase_ifaces(self, phase):
        if phase == 'fwd':
            return ('op_attn_fwd',)
        if phase == 'fwd_bwd':
            return IFACES
        raise ValueError(f'unknown phase {phase!r}')

    def entry_sets(self):
        return {'prime': _entry.prime_axes(), 'coverage': _entry.coverage_axes()}

    def entries(self, axes, arch=None, dropped=None):
        return _entry.expand(axes, self.size, dropped)

    def llm_cases(self):
        return [LLMCase(model=row['model'],
                        phase=row['phase'],
                        entry=_workloads.row_entry(row),
                        pending=bool(row['pending']))
                for row in _workloads.read_workloads()]

    def validate_hw_feature(self, arch, entry):
        # gfx11xx has no kernels for hdim > 256 (LDS and register limits).
        if arch.startswith('gfx11') and max(entry.hdim_qk, entry.hdim_v) > 256:
            return False, f'{arch} does not support hdim > 256'
        limit = _MAX_SEQLEN_BY_ARCH.get(arch)
        if limit is not None and max(entry.seqlen_q, entry.seqlen_k) > limit:
            return False, f'{arch} does not support seqlen > {limit}'
        return True, ''

    def size(self, problem, seqlen_q, seqlen_k):
        if dataclasses.is_dataclass(problem):
            problem = {c: getattr(problem, c) for c in self.PROBLEM_COLUMNS}

        def make_entry(batch, n_heads_q):
            return _entry.FlashPerfEntry(**problem,
                                         seqlen_q=seqlen_q,
                                         seqlen_k=seqlen_k,
                                         batch=batch,
                                         n_heads_q=n_heads_q)

        return _size.size_rule(make_entry, problem['hdim_qk'], problem['gqa_ratio'])

    def input_bytes(self, entry, iface):
        return _size.input_bytes(entry, iface)

    def fits_vram(self, entry, iface, vram_total_bytes):
        return _size.fits_vram(entry, iface, vram_total_bytes)

    def tflops(self, iface, entry, time_us):
        # flash.js takes one head dim; with asymmetric head dims the two
        # matmuls are Dqk and Dv wide, which is their mean for both.
        hdim = (entry.hdim_qk + entry.hdim_v) / 2
        fn = _tflops.attn_bwd_tflops if iface == 'op_attn_bwd' else _tflops.attn_fwd_tflops
        return fn(entry.seqlen_q,
                  entry.seqlen_k,
                  hdim,
                  bool(entry.causal),
                  time_us * 1e-3,
                  entry.batch,
                  entry.n_heads_q)

    def entry_pon(self, entry):
        return render_pon(dataclasses.asdict(entry))

    def prime_tables(self, iface):
        """Rows hdim, columns seqlen (seqlen_q = seqlen_k), one table per
        (causal, dropout_p); each cell bf16, with fp16 as the alternative."""
        axes = self.entry_sets()['prime']
        index = {}
        for e in self.entries(axes):
            if e.seqlen_q == e.seqlen_k:
                index[(e.T_io, e.hdim_qk, e.causal, e.dropout_p, e.seqlen_q)] = e
        seqlens = tuple(sorted(set(axes['seqlen'])))
        tables = []
        for causal in axes['causal']:
            for dropout_p in axes['dropout_p']:
                rows = []
                for hdim in axes['hdim']:
                    cells = tuple((index.get(('bf16', hdim, causal, float(dropout_p), s)),
                                   index.get(('fp16', hdim, causal, float(dropout_p), s)))
                                  for s in seqlens)
                    rows.append((hdim, cells))
                tables.append(PrimeTable(title=f'causal={causal}, dropout_p={dropout_p:g}',
                                         row_label='hdim',
                                         column_label='seqlen',
                                         main_label='bf16',
                                         alt_label='fp16',
                                         columns=seqlens,
                                         rows=tuple(rows)))
        return tables

    def case_columns(self, entry):
        hdim = (str(entry.hdim_qk) if entry.hdim_qk == entry.hdim_v
                else f'{entry.hdim_qk}/{entry.hdim_v}')
        window = ('-' if (entry.window_left, entry.window_right) == (-1, -1)
                  else f'{entry.window_left},{entry.window_right}')
        return {
            'dtype': entry.T_io,
            'hdim': hdim,
            'causal': str(entry.causal),
            'window': window,
            'dropout_p': f'{entry.dropout_p:g}',
            'bias': entry.bias_shape,
            'heads q/kv': f'{entry.n_heads_q}/{entry.n_heads_kv}',
            'batch': str(entry.batch),
            'seqlen_q': str(entry.seqlen_q),
            'seqlen_k': str(entry.seqlen_k),
        }
