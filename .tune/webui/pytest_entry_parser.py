# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Parse a pytest node ID string into FlashEntry fields suitable for a
task_queue lookup.

Supported test functions (from test/test_backward.py):
  test_regular_bwd, test_fast
  test_irregulars
  test_op_bwd_with_matrix_bias  (best-effort)

Bracket parameter order for test_regular_bwd / test_fast (left to right):
  BWDOP, storage_flip, sm_scale, dtype, dropout_p, causal,
  seqlen_k, seqlen_q, D_HEAD, N_HEADS, BATCH

Usage as CLI:
  python pytest_entry_parser.py \
      "test/test_backward.py::test_regular_bwd[Split-False-l1-dtype2-0.0-CausalOff-256-8192-hdim8-5-3]"
"""

import os
import re
import json
import sys
import argparse

# dtype token → dtype string
DTYPE_MAP: dict[str, str] = {
    'dtype0': 'float16',
    'dtype1': 'bfloat16',
    'dtype2': 'float32',
}

# Rounding tables — copied from .tune/libexec/broken_entries_to_db.
# hdim and seqlen values from pytest must be ceiling-rounded to the nearest
# compiled entry because the tuning DB only stores rows for these values.
_BLOCK_DMODEL_DEFAULT = '16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 512'
_BLOCK_DMODEL: list[int] = sorted(
    int(x.strip())
    for x in os.getenv('AOTRITON_FLASH_BLOCK_DMODEL', _BLOCK_DMODEL_DEFAULT).split(',')
)
_SEQLEN_ENTRIES: list[int] = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]


def _round_up(value: int, table: list[int], name: str) -> int:
    for entry in table:
        if entry >= value:
            return entry
    raise ValueError(f'{name} {value} exceeds maximum tuning table entry {table[-1]}')


def parse_hdim(token: str) -> int | tuple[int, int]:
    """Parse 'hdimN' -> N  or  'hdimNxM' -> (N, M)."""
    if not token.startswith('hdim'):
        raise ValueError(f'Expected token starting with "hdim", got {token!r}')
    body = token[len('hdim'):]
    if 'x' in body:
        a, b = body.split('x', 1)
        return (int(a), int(b))
    return int(body)


def parse_nheads(token: str) -> int | tuple[int, int]:
    """Parse 'N' -> N  or  'NxM' -> (N, M)."""
    if 'x' in token:
        a, b = token.split('x', 1)
        return (int(a), int(b))
    return int(token)


def parse_pytest_node_id(node_id: str) -> dict:
    """
    Parse a pytest node ID into a dict of FlashEntry fields.

    Returns a dict with keys:
      dtype, hdim, seqlen_q, seqlen_k, causal, dropout_p, bias_type
      (same fields as FlashEntry)

    On failure raises ValueError with a human-readable message.
    """
    node_id = node_id.strip()

    m = re.search(r'::(\w+)\[(.+)\]$', node_id)
    if not m:
        raise ValueError(
            'Could not parse pytest node ID — '
            'expected format: path::test_name[params]'
        )

    test_name = m.group(1)
    bracket = m.group(2)

    # Split on '-'. None of the expected values contain '-'.
    parts = bracket.split('-')

    if test_name in ('test_regular_bwd', 'test_fast'):
        # Positions: 0=BWDOP, 1=storage_flip, 2=sm_scale, 3=dtype,
        #            4=dropout_p, 5=causal, 6=seqlen_k, 7=seqlen_q,
        #            8=D_HEAD, 9=N_HEADS, 10=BATCH
        if len(parts) < 11:
            raise ValueError(
                f'Expected 11 bracket params for {test_name}, '
                f'got {len(parts)}: {bracket}'
            )
        dtype_token = parts[3]
        dropout_p = float(parts[4])
        causal_str = parts[5]
        seqlen_k = int(parts[6])
        seqlen_q = int(parts[7])
        hdim_token = parts[8]
        # N_HEADS and BATCH are at [9] and [10] but not stored in task_config['entry']

        dtype = DTYPE_MAP.get(dtype_token)
        if dtype is None:
            raise ValueError(
                f'Unknown dtype token: {dtype_token!r}. '
                'Expected dtype0 (float16), dtype1 (bfloat16), dtype2 (float32).'
            )

        causal_map = {'CausalOff': False, 'CausalOn': True}
        if causal_str not in causal_map:
            raise ValueError(
                f'Unknown causal token: {causal_str!r}. '
                'Expected CausalOff or CausalOn.'
            )
        causal = causal_map[causal_str]
        raw_hdim = parse_hdim(hdim_token)
        hdim = (
            (_round_up(raw_hdim[0], _BLOCK_DMODEL, 'hdim_qk'),
             _round_up(raw_hdim[1], _BLOCK_DMODEL, 'hdim_vo'))
            if isinstance(raw_hdim, tuple)
            else _round_up(raw_hdim, _BLOCK_DMODEL, 'hdim')
        )
        seqlen_q = _round_up(seqlen_q, _SEQLEN_ENTRIES, 'seqlen_q')
        seqlen_k = _round_up(seqlen_k, _SEQLEN_ENTRIES, 'seqlen_k')
        bias_type = 0

    elif test_name == 'test_irregulars':
        # Positions: 0=BWDOP, 1=bias_type, 2=storage_flip, 3=sm_scale, 4=dtype,
        #            5=dropout_p, 6=causal, 7=seqlen_k, 8=seqlen_q,
        #            9=D_HEAD, 10=N_HEADS, 11=BATCH
        if len(parts) < 12:
            raise ValueError(
                f'Expected 12 bracket params for {test_name}, '
                f'got {len(parts)}: {bracket}'
            )
        bias_str = parts[1]
        dtype_token = parts[4]
        dropout_p = float(parts[5])
        causal_str = parts[6]
        seqlen_k = int(parts[7])
        seqlen_q = int(parts[8])
        hdim_token = parts[9]

        bias_map = {'BiasOff': 0, 'BiasOn': 1}
        if bias_str not in bias_map:
            raise ValueError(f'Unknown bias_type token: {bias_str!r}. Expected BiasOff or BiasOn.')
        bias_type = bias_map[bias_str]

        dtype = DTYPE_MAP.get(dtype_token)
        if dtype is None:
            raise ValueError(f'Unknown dtype token: {dtype_token!r}')

        causal_map = {'CausalOff': False, 'CausalOn': True}
        if causal_str not in causal_map:
            raise ValueError(f'Unknown causal token: {causal_str!r}. Expected CausalOff or CausalOn.')
        causal = causal_map[causal_str]
        raw_hdim = parse_hdim(hdim_token)
        hdim = (
            (_round_up(raw_hdim[0], _BLOCK_DMODEL, 'hdim_qk'),
             _round_up(raw_hdim[1], _BLOCK_DMODEL, 'hdim_vo'))
            if isinstance(raw_hdim, tuple)
            else _round_up(raw_hdim, _BLOCK_DMODEL, 'hdim')
        )
        seqlen_q = _round_up(seqlen_q, _SEQLEN_ENTRIES, 'seqlen_q')
        seqlen_k = _round_up(seqlen_k, _SEQLEN_ENTRIES, 'seqlen_k')

    elif test_name == 'test_op_bwd_with_matrix_bias':
        # Parametrize order (no causal param; causal=False hardcoded):
        #   BWDOP(0), storage_flip(1), sm_scale(2), dtype(3), dropout_p(4),
        #   seqlen_k(5), seqlen_q(6), D_HEAD(7), N_HEADS(8), BATCH(9)
        if len(parts) < 9:
            raise ValueError(
                f'Expected at least 9 bracket params for {test_name}, '
                f'got {len(parts)}: {bracket}'
            )
        dtype_token = parts[3]
        dropout_p = float(parts[4])
        seqlen_k = int(parts[5])
        seqlen_q = int(parts[6])
        hdim_token = parts[7]

        dtype = DTYPE_MAP.get(dtype_token)
        if dtype is None:
            raise ValueError(f'Unknown dtype token: {dtype_token!r}')
        raw_hdim = parse_hdim(hdim_token)
        hdim = (
            (_round_up(raw_hdim[0], _BLOCK_DMODEL, 'hdim_qk'),
             _round_up(raw_hdim[1], _BLOCK_DMODEL, 'hdim_vo'))
            if isinstance(raw_hdim, tuple)
            else _round_up(raw_hdim, _BLOCK_DMODEL, 'hdim')
        )
        seqlen_q = _round_up(seqlen_q, _SEQLEN_ENTRIES, 'seqlen_q')
        seqlen_k = _round_up(seqlen_k, _SEQLEN_ENTRIES, 'seqlen_k')
        causal = False
        bias_type = 1

    else:
        raise ValueError(
            f'Unsupported test name: {test_name!r}. '
            'Supported: test_regular_bwd, test_fast, test_irregulars, test_op_bwd_with_matrix_bias.'
        )

    return {
        'dtype': dtype,
        'hdim': hdim,
        'seqlen_q': seqlen_q,
        'seqlen_k': seqlen_k,
        'causal': causal,
        'dropout_p': dropout_p,
        'bias_type': bias_type,
    }


def entry_to_sql_clauses(entry: dict) -> tuple[list[str], list]:
    """
    Convert a FlashEntry field dict into (clauses, params) for a psycopg
    WHERE clause on task_config->'entry'.

    Does NOT include an arch filter — pytest IDs do not encode arch.
    """
    clauses = []
    params: list = []
    for field, value in entry.items():
        col = f"task_config->'entry'->>'{field}'"
        if isinstance(value, tuple):
            # hdim stored as JSON array [a, b]
            json_val = json.dumps(list(value))
            clauses.append(f"task_config->'entry'->'{field}' = %s::jsonb")
            params.append(json_val)
        elif isinstance(value, bool):
            clauses.append(f"({col})::boolean = %s")
            params.append(value)
        elif isinstance(value, int):
            clauses.append(f"({col})::integer = %s")
            params.append(value)
        elif isinstance(value, float):
            clauses.append(f"({col})::float = %s")
            params.append(value)
        else:
            clauses.append(f"{col} = %s")
            params.append(value)
    return clauses, params


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Parse a pytest node ID and show the equivalent FlashEntry fields.',
        epilog=(
            'Example:\n'
            '  %(prog)s '
            '"test/test_backward.py::test_regular_bwd'
            '[Split-False-l1-dtype2-0.0-CausalOff-256-8192-hdim8-5-3]"'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'node_id',
        help='Pytest node ID string (e.g. test/test_backward.py::test_name[params])',
    )
    parser.add_argument(
        '--json',
        action='store_true',
        dest='as_json',
        help='Output as JSON instead of human-readable text',
    )
    args = parser.parse_args()

    try:
        entry = parse_pytest_node_id(args.node_id)
    except ValueError as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)

    if args.as_json:
        # tuples are not JSON-serializable natively
        serialisable = {
            k: list(v) if isinstance(v, tuple) else v
            for k, v in entry.items()
        }
        print(json.dumps(serialisable, indent=2))
    else:
        print('Parsed FlashEntry fields:')
        for k, v in entry.items():
            print(f'  {k} = {v!r}')
        print()
        clauses, params = entry_to_sql_clauses(entry)
        print('SQL WHERE clauses (no arch filter — pytest IDs do not encode arch):')
        for clause, param in zip(clauses, params):
            print(f'  {clause}  [param={param!r}]', end=' ')
        print()


if __name__ == '__main__':
    main()
