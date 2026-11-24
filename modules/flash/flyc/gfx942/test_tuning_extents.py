#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The grid extents the tuning policies record (`block_m`, `block_kv`) follow the
geometry overrides, and match what the built kernels cover.

The C++ grid calculator divides the sequence length by the recorded extent, so
an extent larger than the kernel's leaves rows or keys unwritten. The resolve
tests are flydsl-free; the build tests read the extent from the builder's traits
(no compile, no GPU) and skip without flydsl.

    AOTRITON_FLYDSL_KERNEL_ROOT=<flydsl-checkout> \\
        python -m pytest modules/flash/flyc/gfx942/test_tuning_extents.py
"""

import os
import sys

import pytest

import tuning_dkdv
import tuning_dq
import tuning_fwd
from fmha_meta import FmhaInputMetadata

TUNING = {'fwd': tuning_fwd, 'dq': tuning_dq, 'dkdv': tuning_dkdv}
EXTENT = {'fwd': 'block_m', 'dq': 'block_m', 'dkdv': 'block_kv'}

# (kernel, head_dim, overrides, extents by causal: (non-causal, causal)), with
# the default build of every body and the overrides the policies used to miss.
CASES = [
    ('fwd', 64, {}, (256, 256)),
    ('fwd', 128, {'num_waves': 4}, (128, 128)),
    ('fwd', 192, {}, (128, 256)),
    ('fwd', 512, {}, (128, 128)),
    ('dq', 64, {}, (256, 256)),
    ('dq', 128, {'num_waves': 4}, (128, 128)),
    ('dq', 192, {}, (128, 256)),
    ('dq', 192, {'wave_m': 16}, (64, 128)),
    ('dq', 512, {}, (64, 128)),
    ('dkdv', 64, {}, (128, 128)),
    ('dkdv', 64, {'kv_blocks': 1}, (64, 64)),
    ('dkdv', 128, {'num_waves': 2}, (64, 64)),
    ('dkdv', 192, {}, (128, 128)),
    ('dkdv', 256, {}, (128, 128)),
    ('dkdv', 512, {}, (48, 48)),
]


def _meta(head_dim, causal):
    return FmhaInputMetadata(num_heads=2,
                             dtype_str='bf16',
                             causal_type=2 if causal else 0,
                             block_dmodel=head_dim,
                             padded_head=False)


def _resolve(kernel, head_dim, causal, overrides):
    return TUNING[kernel].resolve_knobs('gfx942', _meta(head_dim, causal), overrides=overrides)


def _ids(case):
    kernel, head_dim, overrides, _ = case
    return '-'.join([kernel, f'hd{head_dim}'] + [f'{k}{v}' for k, v in overrides.items()])


@pytest.mark.parametrize('causal', (False, True))
@pytest.mark.parametrize('case', CASES, ids=[_ids(c) for c in CASES])
def test_resolved_extent(case, causal):
    """The recorded extent follows the overrides, and an agreeing explicit one passes."""
    kernel, head_dim, overrides, extents = case
    knobs = _resolve(kernel, head_dim, causal, overrides)
    assert getattr(knobs, EXTENT[kernel]) == extents[causal]
    explicit = dict(overrides, **{EXTENT[kernel]: extents[causal]})
    assert _resolve(kernel, head_dim, causal, explicit) == knobs


@pytest.mark.parametrize('kernel, head_dim, overrides', [
    ('fwd', 128, {'num_waves': 4, 'block_m': 256}),
    ('fwd', 192, {'block_m': 128}),             # causal pairs q tiles: 256
    ('dq', 128, {'num_waves': 4, 'block_m': 256}),
    ('dq', 192, {'wave_m': 16, 'block_n': 32}),
    ('dq', 192, {'wave_m': 16, 'block_m': 256}),
    ('dkdv', 64, {'kv_blocks': 1, 'block_kv': 128}),
    ('dkdv', 192, {'kv_subs': 2, 'block_kv': 64}),
])
def test_conflicting_extent_raises(kernel, head_dim, overrides):
    with pytest.raises(ValueError, match='conflicts with the build'):
        _resolve(kernel, head_dim, True, overrides)


def test_wide_dq_block_n_follows_wave_m():
    knobs = _resolve('dq', 192, False, {'wave_m': 16})
    assert (knobs.wave_m, knobs.block_n) == (16, 16)


def _built_extent(kernel, traits):
    """The rows (fwd, dQ) or keys (dK/dV) one workgroup of the built kernel covers."""
    if kernel == 'dkdv':
        return traits.block_kv if hasattr(traits, 'block_kv') else traits.block_n
    return getattr(traits, 'grid_block_m', traits.block_m)


@pytest.mark.parametrize('causal', (False, True))
@pytest.mark.parametrize('case', CASES, ids=[_ids(c) for c in CASES])
def test_built_extent(case, causal):
    """The resolved knobs build a kernel whose workgroup covers the recorded extent."""
    root = os.environ.get('AOTRITON_FLYDSL_KERNEL_ROOT')
    if not root:
        pytest.skip('needs AOTRITON_FLYDSL_KERNEL_ROOT')
    if root not in sys.path:
        sys.path.append(root)
    pytest.importorskip('flydsl')
    kernel, head_dim, overrides, extents = case
    meta = _meta(head_dim, causal)
    knobs = _resolve(kernel, head_dim, causal, overrides)
    launch = TUNING[kernel].builder('gfx942', meta, knobs)
    assert _built_extent(kernel, launch.traits) == getattr(knobs, EXTENT[kernel]) == extents[causal]
