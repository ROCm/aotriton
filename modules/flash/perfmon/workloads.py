# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
The `llm` entry set: `workloads.csv`, one row per real LLM workload.

The file is edited by hand and reviewed like code. Its rows are entries with
the workload's own sizes (never the size rule's), plus:

    model       labels the case in rendered tables; not part of the entry
    phase       fwd (forward only) or fwd_bwd (forward and backward)
    n_heads_q, n_heads_kv
                the workload's head counts; the entry stores n_heads_q and
                gqa_ratio = n_heads_q / n_heads_kv
    pending     1 for a workload the adapters cannot run yet (decode, varlen,
                asymmetric head dims, ...): listed, not dispatched

Rows may be reordered or edited freely: the perfdb matches entries by content,
and the order here is only the order of the rendered LLM tables.
"""

import csv
from pathlib import Path

from .entry import PROBLEM_COLUMNS, FlashPerfEntry

WORKLOADS_CSV = Path(__file__).with_name('workloads.csv')

COLUMNS = (('model', 'phase')
           + tuple(c for c in PROBLEM_COLUMNS if c != 'gqa_ratio')
           + ('n_heads_q', 'n_heads_kv', 'batch', 'seqlen_q', 'seqlen_k', 'pending'))

_INT_COLUMNS = ('hdim_qk', 'hdim_v', 'causal', 'window_left', 'window_right', 'bias_type', 'varlen',
                'storage_flip', 'n_heads_q', 'n_heads_kv', 'batch', 'seqlen_q', 'seqlen_k', 'pending')


def read_workloads(path: Path = WORKLOADS_CSV) -> list[dict]:
    """The rows as typed dicts, in file order. Lines starting with '#' are
    comments."""
    with open(path, newline='', encoding='utf-8') as f:
        numbered = [(n, line) for n, line in enumerate(f, start=1)
                    if line.strip() and not line.startswith('#')]
    # numbered[0] is the header; each later entry is one row (no multi-line fields).
    reader = csv.DictReader(line for _, line in numbered)
    if set(reader.fieldnames or ()) != set(COLUMNS):
        raise ValueError(f'{path}: columns must be {list(COLUMNS)}, got {reader.fieldnames}')
    rows = []
    for index, raw in enumerate(reader, start=1):
        n = numbered[index][0]      # the row's real line in the file
        row = dict(raw)
        try:
            for c in _INT_COLUMNS:
                row[c] = int(row[c])
            row['dropout_p'] = float(row['dropout_p'])
            if row['phase'] not in ('fwd', 'fwd_bwd'):
                raise ValueError(f"phase must be fwd or fwd_bwd, got {row['phase']!r}")
            if row['n_heads_kv'] <= 0:
                raise ValueError(f"n_heads_kv must be positive, got {row['n_heads_kv']}")
            if row['n_heads_q'] % row['n_heads_kv']:
                raise ValueError('n_heads_q must be a multiple of n_heads_kv')
        except ValueError as e:
            raise ValueError(f'{path}, line {n}: {e}') from None
        rows.append(row)
    return rows


def row_entry(row: dict) -> FlashPerfEntry:
    problem = {c: row[c] for c in PROBLEM_COLUMNS if c != 'gqa_ratio'}
    return FlashPerfEntry(**problem,
                          gqa_ratio=row['n_heads_q'] // row['n_heads_kv'],
                          seqlen_q=row['seqlen_q'],
                          seqlen_k=row['seqlen_k'],
                          batch=row['batch'],
                          n_heads_q=row['n_heads_q'])
