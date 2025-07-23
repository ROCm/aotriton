#!/usr/bin/env python
# Copyright ©2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import triton
import triton.language as tl
from composed_tensors import (
    composed_ptrs,
    composed_load,
    composed_to,
    composed_store,
)

@triton.jit
def bwd_postprocess(
    DQ_ACC, DQ,
    stride_accz, stride_acch, stride_accm, stride_acck,
    stride_dqz, stride_dqh, stride_dqm, stride_dqk,
    seqlen_q,
    head_dim,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
):
    tl.static_assert(BLOCK_DMODEL > 0, 'BLOCK_DMODEL must be greater than 0')
    BLOCK_DMODEL_R0 : tl.constexpr = BLOCK_DMODEL
    BLOCK_DMODEL0 : tl.constexpr = 2 ** (BLOCK_DMODEL_R0.bit_length() - 1)
    BLOCK_DMODEL_R1 : tl.constexpr = BLOCK_DMODEL_R0 - BLOCK_DMODEL0
    BLOCK_DMODEL1 : tl.constexpr = 2 ** (BLOCK_DMODEL_R1.bit_length() - 1) if BLOCK_DMODEL_R1 > 0 else 0
    BLOCK_DMODEL_R2 : tl.constexpr = BLOCK_DMODEL_R1 - BLOCK_DMODEL1
    BLOCK_DMODEL2 : tl.constexpr = 2 ** (BLOCK_DMODEL_R2.bit_length() - 1) if BLOCK_DMODEL_R2 > 0 else 0
    BLOCK_DMODEL_R3 : tl.constexpr = BLOCK_DMODEL_R2 - BLOCK_DMODEL2

    tl.static_assert(BLOCK_DMODEL_R3 == 0, f'BLOCK_DMODEL = {BLOCK_DMODEL} = 0b{BLOCK_DMODEL:b} cannot be factored into <= 3 power of two values')
    tl.static_assert(BLOCK_DMODEL1 > 0 or BLOCK_DMODEL2 == 0, 'Only trailing BLOCK_DMODELx can be 0')

    off_m = tl.program_id(0) * BLOCK_M
    offs_m = off_m + tl.arange(0, BLOCK_M)
    off_h = tl.program_id(1) # head index
    off_z = tl.program_id(2) # batch index

    acc_ptrs0, acc_ptrs1, acc_ptrs2 = composed_ptrs(DQ_ACC,
                                                    stride_accz, stride_acch, stride_accm, stride_acck,
                                                    off_z, off_h, offs_m,
                                                    BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2)

    if off_m + BLOCK_M > seqlen_q:
        acc0, acc1, acc2 = composed_load(acc_ptrs0, acc_ptrs1, acc_ptrs2,
                                         offs_m,
                                         BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2,
                                         seqlen_q, head_dim,
                                         other=0.0,
                                         PADDED_ROW=True,
                                         PADDED_COL=PADDED_HEAD,
                                         TRANSPOSED=False)
    else:
        acc0, acc1, acc2 = composed_load(acc_ptrs0, acc_ptrs1, acc_ptrs2,
                                         offs_m,
                                         BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2,
                                         seqlen_q, head_dim,
                                         other=0.0,
                                         PADDED_ROW=False,
                                         PADDED_COL=PADDED_HEAD,
                                         TRANSPOSED=False)

    dq_ptrs0, dq_ptrs1, dq_ptrs2 = composed_ptrs(DQ,
                                                 stride_dqz, stride_dqh, stride_dqm, stride_dqk,
                                                 off_z, off_h, offs_m,
                                                 BLOCK_DMODEL0, BLOCK_DMODEL1, BLOCK_DMODEL2)

    acc0, acc1, acc2 = composed_to(acc0, acc1, acc2, DQ.type.element_ty)
    o_base = DQ + off_z * stride_dqz + off_h * stride_dqh
    composed_store(acc0, acc1, acc2,
                   BLOCK_M,
                   BLOCK_DMODEL0,
                   BLOCK_DMODEL1,
                   BLOCK_DMODEL2,
                   o_base=o_base,
                   o_start_row=off_m,
                   o_start_col=0,
                   o_rows=seqlen_q,
                   o_cols=head_dim,
                   stride_row=stride_dqm,
                   stride_col=stride_dqk)

if __name__ == '__main__':
    def test():
        import torch
        BLOCK = 128
        HDIM = 32
        dq_acc = torch.rand((5, 3, 190, HDIM), dtype=torch.float, device='cuda')
        dq = torch.empty_like(dq_acc, dtype=torch.bfloat16)
        grid_post = (triton.cdiv(dq.shape[2], BLOCK), dq.shape[1], dq.shape[0])
        bwd_postprocess[grid_post](dq_acc, dq,
                                   *dq_acc.stride(), *dq.stride(),
                                   dq_acc.shape[2], dq_acc.shape[3],
                                   BLOCK, HDIM, False)
        print(dq_acc)
        print(dq)
        print(torch.max(torch.abs(dq - dq_acc.to(torch.bfloat16))))
    test()
