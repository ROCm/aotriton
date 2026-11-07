# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A tiny fake perfmon family and database rows, for the perfdb and export
tests: small enough to read a golden file by eye."""

import itertools
from dataclasses import dataclass
from datetime import datetime, timezone

from aotriton.perfmon.pdesc import LLMCase, PerfDescription, PrimeTable

IFACES = ('op_a', 'op_b')


@dataclass(frozen=True)
class FakeEntry:
    T_io: str
    hdim: int
    causal: int
    seqlen_q: int
    seqlen_k: int
    batch: int
    n_heads_q: int


class FakeDesc(PerfDescription):
    PROBLEM_COLUMNS = ('T_io', 'hdim', 'causal')
    ENTRY_CLASS = FakeEntry

    AXES = {'T_io': ['bf16', 'fp16'], 'hdim': [16, 256], 'causal': [0], 'seqlen': [16, 32]}

    def ifaces(self):
        return IFACES

    def phase_ifaces(self, phase):
        return ('op_a',) if phase == 'fwd' else IFACES

    def entry_sets(self):
        return {'prime': dict(self.AXES)}

    def entries(self, axes, arch=None, dropped=None):
        combos = itertools.product(axes['T_io'], axes['hdim'], axes['causal'], axes['seqlen'])
        for t_io, hdim, causal, seqlen in combos:
            yield FakeEntry(t_io, hdim, causal, seqlen, seqlen, 2, 8)

    def llm_cases(self):
        return [LLMCase('model-x', 'fwd_bwd', FakeEntry('fp16', 64, 1, 128, 128, 4, 12)),
                LLMCase('model-y', 'fwd', FakeEntry('bf16', 64, 0, 100, 100, 3, 6)),
                LLMCase('model-z', 'fwd_bwd', FakeEntry('bf16', 64, 0, 1, 4096, 3, 6), pending=True)]

    def validate_hw_feature(self, arch, entry):
        if arch == 'gfx1100' and entry.hdim > 128:
            return False, 'hdim > 128'
        return True, ''

    def size(self, problem, seqlen_q, seqlen_k):
        return 2, 8

    def input_bytes(self, entry, iface):
        return 1

    def fits_vram(self, entry, iface, vram_total_bytes):
        return True, ''

    def tflops(self, iface, entry, time_us):
        # 1 TFLOPS at 1000 us for hdim 100 and seqlen 100: simple and exact
        return entry.hdim * entry.seqlen_q / time_us / 10

    def entry_pon(self, entry):
        return f'hdim={entry.hdim}'

    def prime_tables(self, iface):
        entries = {(e.T_io, e.hdim, e.seqlen_q): e for e in self.entries(self.AXES)}
        seqlens = tuple(self.AXES['seqlen'])
        rows = tuple((hdim, tuple((entries[('bf16', hdim, s)], entries[('fp16', hdim, s)]) for s in seqlens))
                     for hdim in self.AXES['hdim'])
        return [PrimeTable(title='causal=0',
                           row_label='hdim',
                           column_label='seqlen',
                           main_label='bf16',
                           alt_label='fp16',
                           columns=seqlens,
                           rows=rows)]

    def case_columns(self, entry):
        return {'dtype': entry.T_io, 'hdim': str(entry.hdim), 'batch': str(entry.batch),
                'seqlen': str(entry.seqlen_q)}


def stats(median):
    return {'median_us': median, 'mean_us': median + 0.01, 'stddev_us': 0.05, 'min_us': median - 0.2,
            'p05_us': median - 0.1, 'p95_us': median + 0.3}


def row(desc, entry, iface='op_a', backend=None, median=100.0, status='OK', variant='any',
        subject='aotriton-0.13b+rocm10.1.0', day=1, gpu_device='0x75a0', **patch):
    out = {
        'variant': variant, 'subject': subject, 'module': 'fake', 'iface': iface, 'backend': backend,
        'entry': desc.entry_to_dict(entry), 'status': status,
        'timing_method': 'hipgraph_ev100', 'iterations': 100, 'l2_flush': 1,
        'platform': {'pci_device_id': gpu_device, 'vram_total_bytes': 288 << 30, 'thermal_gate': 'off',
                     'cpu': 'Test CPU', 'os': 'Debian 13', 'kernel': '6.12', 'gpus_per_node': 8},
        'harness': {'commit': 'abc1234', 'arches': ['gfx950']},
        'created_at': datetime(2026, 11, day, 12, 0, tzinfo=timezone.utc),
    }
    out.update(stats(median) if status == 'OK' else {k: None for k in stats(1)})
    out.update(patch)
    return out
