# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fake flyc forward backend: a few kernel arguments plus the BLOCK_DMODEL /
PADDED_HEAD markers. Its gfx942 ladder has 384, which the cited operator axis
(block_dmodel_values()) does not; the generator must still enumerate it."""

import aotriton.template_instantiation as ati

# 384 is on no operator rung; most operator rungs are on no ladder here.
FAKE_FLYC_LADDERS = {
    'gfx942': frozenset({64, 128, 384, 512}),
    'gfx950': frozenset({64, 128, 512}),
}


def _fake_flyc_disabled(f):
    if f.arch not in FAKE_FLYC_LADDERS:
        return True
    return f.choices.BLOCK_DMODEL not in FAKE_FLYC_LADDERS[f.arch]


@ati.start
@ati.disable(when=_fake_flyc_disabled,
             I_understand_this_overrides_cited_disable=True)
@ati.cite('op_attn_fwd.triton.attn_fwd')
@ati.tensor('Q', 'T_io', rank=4, strides='stride_q_*', wires_to='Q')
@ati.tensor('O', 'T_io', rank=4, strides='stride_o_*', wires_to='Out')
@ati.scalar('hdim_qk', 'i32', wires_to='Hdim_qk')
@ati.scalar('hdim_vo', 'i32', wires_to='Hdim_vo')
@ati.scalar('BLOCK_DMODEL',
            options=sorted(set().union(*FAKE_FLYC_LADDERS.values())),
            wires_to=ati.context_helper('flyc_block_dmodel'))
@ati.scalar('PADDED_HEAD', options=[False, True],
            wires_to=ati.context_helper('flyc_padded_head'))
@ati.flyc.kernel()
def flyc_attn_fwd(arch, choices, hints):
    raise AssertionError('the generator must never call a fake flyc builder')
