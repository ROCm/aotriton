#!/usr/bin/env python
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Re-introduce historical attention-kernel mistakes into a COPY of
# modules/flash/kernel, so test_common_mistakes can be shown to catch each one.
# See README.md next to this file; run_mutants.sh drives it.
#
#   python mutants.py make <name>|all     -> $MUTANTS_DIR/<name>/ (full copy + patch)
#   python mutants.py list
#
# MUTANTS_DIR defaults to <tmp>/aotriton-mutants, so nothing lands in the tree.
#
# Every replacement must match exactly once, so a mutant that silently stops
# applying (because the kernel moved on) fails loudly instead of testing nothing.

import os
import sys
import shutil
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
KERNEL_DIR = HERE.parents[1] / 'modules' / 'flash' / 'kernel'
OUT_DIR = Path(os.getenv('MUTANTS_DIR', Path(tempfile.gettempdir()) / 'aotriton-mutants'))

MUTANTS = {
    # 7a165f1e (#46): Q pre-multiplied by sm_scale*log2(e) and rounded back to
    # the input dtype before QK^T, in the forward.
    'fwd_q_prescale': [
        ('fwd_kernel_inner.py',
         'qk += (Qk_scale * tl.dot(q0, k0))',
         'qk += tl.dot((q0 * Qk_scale).to(q0.type.element_ty), k0)'),
        ('fwd_kernel_inner.py',
         'if BLOCK_DMODEL1 > 0 : qk += (Qk_scale * tl.dot(q1, k1))',
         'if BLOCK_DMODEL1 > 0 : qk += tl.dot((q1 * Qk_scale).to(q1.type.element_ty), k1)'),
        ('fwd_kernel_inner.py',
         'if BLOCK_DMODEL2 > 0 : qk += (Qk_scale * tl.dot(q2, k2))',
         'if BLOCK_DMODEL2 > 0 : qk += tl.dot((q2 * Qk_scale).to(q2.type.element_ty), k2)'),
    ],
    # 7a165f1e (#46), backward half: the dq kernel pre-scaled Q and the dk/dv
    # kernel pre-scaled K, each rounded back to the input dtype. Only the first
    # head-dim sub-block is scaled, which covers every power-of-two head dim.
    'bwd_qk_prescale': [
        ('bwd_kernel_dq.py',
         '    qk_scale = sm_scale * RCP_LN2\n',
         '    qk_scale = sm_scale * RCP_LN2\n'
         '    q0 = (q0 * qk_scale).to(q0.type.element_ty)\n'),
        ('bwd_inner_dq.py',
         'p = tl.math.exp2(qk_scale * qk - l_i[:, None])',
         'p = tl.math.exp2(qk - l_i[:, None])'),
        ('bwd_kernel_dk_dv.py',
         '    qk_scale = sm_scale * 1.44269504089\n',
         '    qk_scale = sm_scale * 1.44269504089\n'
         '    kt0 = (kt0 * qk_scale).to(kt0.type.element_ty)\n'),
        ('bwd_inner_dk_dv.py',
         'p = tl.math.exp2(qk_scale * qk - l_i) # (BLOCK_M, BLOCK_N)',
         'p = tl.math.exp2(qk - l_i) # (BLOCK_M, BLOCK_N)'),
    ],
    # The scores S = QK^T kept in the input dtype (e.g. staged through LDS as
    # bf16) before the fp32 softmax. Same exponent error as pre-scaling Q.
    'fwd_s_lowp': [
        ('fwd_kernel_inner.py',
         'qk += (Qk_scale * tl.dot(q0, k0))',
         'qk += (Qk_scale * tl.dot(q0, k0).to(q0.type.element_ty).to(tl.float32))'),
    ],
    # 52a37783 (#108): LSE stored in base 2. Forward and backward stay
    # consistent with each other, so only the LSE itself is wrong.
    'lse_base2': [
        ('fwd_kernel.py',
         '                    logsumexp *= 0.6931471824645996\n',
         ''),
        ('bwd_inner_dk_dv.py',
         '        l_i *= RCP_LN2\n',
         ''),
        ('bwd_kernel_dq.py',
         'mask=d_lse_ptrs_mask, other=0.0) * RCP_LN2',
         'mask=d_lse_ptrs_mask, other=0.0)'),
    ],
    # 6f8cbcac (#57): the running max initialised to -inf. A row with no
    # unmasked key in the first block it meets computes -inf - (-inf) = NaN.
    'fwd_m_init_neg_inf': [
        ('fwd_kernel.py',
         'm_i = tl.full([BLOCK_M], -3.40282e+38, dtype=tl.float32)  # FILEPR',
         'm_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)  # FILEPR'),
    ],
    # aef9087d (#48): no guard for sm_scale == 0. The backward applies the
    # mask before the scale, so a masked score is -inf * 0 = NaN.
    'bwd_no_zero_scale_guard': [
        ('bwd_inner_dk_dv.py',
         '            if qk_scale == 0.0:\n                p = tl.where(libdevice.isnan(p), 0.0, p)\n',
         '            pass\n'),
        ('bwd_inner_dq.py',
         '            if qk_scale == 0.0:\n                p = tl.where(libdevice.isnan(p), 0.0, p)\n',
         '            pass\n'),
    ],
    # Delta = rowsum(dO * O) reduced in the input dtype instead of fp32.
    'bwd_delta_lowp': [
        ('composed_tensors.py',
         '    x = tl.sum(lx.to(tl.float32) * rx.to(tl.float32), axis=axis)\n',
         '    x = tl.sum(lx * rx, axis=axis).to(lx.dtype).to(tl.float32)\n'),
    ],
}


def make(name):
    dst = OUT_DIR / name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(KERNEL_DIR, dst, ignore=shutil.ignore_patterns('__pycache__'))
    for fn, old, new in MUTANTS[name]:
        path = dst / fn
        text = path.read_text()
        n = text.count(old)
        if n != 1:
            raise SystemExit(f'{name}: {fn}: pattern matched {n} times, expected 1:\n{old}')
        path.write_text(text.replace(old, new))
    return dst


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'list'
    if cmd == 'list':
        print('\n'.join(MUTANTS))
    elif cmd == 'make':
        names = list(MUTANTS) if sys.argv[2] == 'all' else sys.argv[2:]
        for name in names:
            print(make(name))
    else:
        raise SystemExit(f'unknown command {cmd}')


if __name__ == '__main__':
    main()
