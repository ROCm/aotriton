# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Translate a parsed pytest node ID into flash tuning-entry fields.

Everything here is a property of flash's own test parametrization -- the
order of the bracket parameters, the spelling of their tokens, and the axes
the tuning database is built on. Splitting the node ID is family-neutral and
lives in aotriton.tune.pytest_node.

Torch-free and import-light: reached from the web UI, which runs outside any
GPU container.
"""

import os
import re

# dtype token -> dtype name. pytest renders an unnamed parametrize value as
# '<argname><index>', so these track the order of the dtype list in
# modules/flash/tests/.
DTYPE_MAP: dict[str, str] = {
    'dtype0': 'float16',
    'dtype1': 'bfloat16',
    'dtype2': 'float32',
}

CAUSAL_MAP: dict[str, bool] = {'CausalOff': False, 'CausalOn': True}
BIAS_MAP: dict[str, int] = {'BiasOff': 0, 'BiasOn': 1}

# Tuning-database axes. A pytest case uses arbitrary shapes, but the database
# only holds rows at these values, so a lookup has to round up to the entry
# that would have covered the case.
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
    """'hdim8' -> 8; 'hdim(64,128)' / 'hdim64x128' -> (64, 128)."""
    body = token[len('hdim'):] if token.startswith('hdim') else token
    nums = [int(n) for n in re.findall(r'\d+', body)]
    if not nums:
        raise ValueError(f'Could not parse hdim token: {token!r}')
    return nums[0] if len(nums) == 1 else (nums[0], nums[1])


def parse_nheads(token: str) -> int | tuple[int, int]:
    """Same shape as parse_hdim, for the N_HEADS parameter."""
    nums = [int(n) for n in re.findall(r'\d+', token)]
    if not nums:
        raise ValueError(f'Could not parse N_HEADS token: {token!r}')
    return nums[0] if len(nums) == 1 else (nums[0], nums[1])


def _hdim_rounded(token: str):
    raw = parse_hdim(token)
    if isinstance(raw, tuple):
        return (_round_up(raw[0], _BLOCK_DMODEL, 'hdim_qk'),
                _round_up(raw[1], _BLOCK_DMODEL, 'hdim_vo'))
    return _round_up(raw, _BLOCK_DMODEL, 'hdim')


def _lookup(mapping: dict, token: str, what: str):
    try:
        return mapping[token]
    except KeyError:
        raise ValueError(
            f'Unknown {what} token: {token!r}. '
            f'Expected one of {sorted(mapping)}.') from None


# Bracket-parameter positions per test, left to right. Kept as data so a
# parametrize change is a one-line edit against the test's own signature.
#
#   n:      minimum number of parameters
#   idx:    field -> position
#   const:  fields the test fixes rather than parametrizes
_LAYOUTS: dict[str, dict] = {
    # BWDOP, storage_flip, sm_scale, dtype, dropout_p, causal,
    # seqlen_k, seqlen_q, D_HEAD, N_HEADS, BATCH
    'test_regular_bwd': {
        'n': 11,
        'idx': {'dtype': 3, 'dropout_p': 4, 'causal': 5,
                'seqlen_k': 6, 'seqlen_q': 7, 'hdim': 8},
        'const': {'bias_type': 0},
    },
    # BWDOP, bias_type, storage_flip, sm_scale, dtype, dropout_p, causal,
    # seqlen_k, seqlen_q, D_HEAD, N_HEADS, BATCH
    'test_irregulars': {
        'n': 12,
        'idx': {'bias_type': 1, 'dtype': 4, 'dropout_p': 5, 'causal': 6,
                'seqlen_k': 7, 'seqlen_q': 8, 'hdim': 9},
        'const': {},
    },
    # BWDOP, storage_flip, sm_scale, dtype, dropout_p,
    # seqlen_k, seqlen_q, D_HEAD, N_HEADS, BATCH  -- no causal parameter
    'test_op_bwd_with_matrix_bias': {
        'n': 9,
        'idx': {'dtype': 3, 'dropout_p': 4,
                'seqlen_k': 5, 'seqlen_q': 6, 'hdim': 7},
        'const': {'causal': False, 'bias_type': 1},
    },
}
_LAYOUTS['test_fast'] = _LAYOUTS['test_regular_bwd']


def supported_tests() -> list[str]:
    return sorted(_LAYOUTS)


def entry_from_pytest_node(node) -> dict:
    """ParsedNode (aotriton.tune.pytest_node) -> FlashEntry field dict.

    Shapes are rounded up to the nearest tuning-database axis value, so the
    result identifies the entry that covers the pytest case rather than the
    case's own shape.
    """
    layout = _LAYOUTS.get(node.test)
    if layout is None:
        raise ValueError(
            f'Unsupported test name: {node.test!r}. '
            f'Supported: {", ".join(supported_tests())}.')
    node.require(layout['n'])
    at = {name: node.params[i] for name, i in layout['idx'].items()}

    entry = dict(layout['const'])
    entry['dtype'] = _lookup(DTYPE_MAP, at['dtype'], 'dtype')
    entry['hdim'] = _hdim_rounded(at['hdim'])
    entry['seqlen_q'] = _round_up(int(at['seqlen_q']), _SEQLEN_ENTRIES, 'seqlen_q')
    entry['seqlen_k'] = _round_up(int(at['seqlen_k']), _SEQLEN_ENTRIES, 'seqlen_k')
    entry['dropout_p'] = float(at['dropout_p'])
    if 'causal' in at:
        entry['causal'] = _lookup(CAUSAL_MAP, at['causal'], 'causal')
    if 'bias_type' in at:
        entry['bias_type'] = _lookup(BIAS_MAP, at['bias_type'], 'bias_type')

    # Fixed field order, matching FlashEntry, so callers building SQL or
    # comparing dicts see a stable shape regardless of which test produced it.
    return {k: entry[k] for k in
            ('dtype', 'hdim', 'seqlen_q', 'seqlen_k', 'causal', 'dropout_p', 'bias_type')}
