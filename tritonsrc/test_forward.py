#!/usr/bin/env python

import pytest
import torch

from attn_torch_function import attention

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    # Efficient implementation equivalent to the following:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    """
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    """
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    SPARSE_HEAD_SINCE = 5
    SPARSE_SEQ_SINCE = 5
    print(f'{query=}')
    print(f'{key=}')
    print(f'BEFORE softmax {attn_weight[:,:, :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
    # attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    print(f'BEFORE DROPOUT_MASK {attn_weight[:,:, :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
    if dropout_p > 0.0:
        if dropout_mask is not None:
            print(f'BEFORE DROPOUT_MASK {attn_weight[:,:, :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
            attn_weight.masked_fill_(dropout_mask.logical_not(), float("0.0"))
            print(f'AFTER DROPOUT_MASK {attn_weight[:,:, :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
            value = value / (1 - dropout_p)
        else:
            # assert False, "TESTING dropout_mask code path"
            attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    else:
        # assert False, "TESTING dropout_mask code path"
        pass
    print(f'{value[:,:, :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
    av = attn_weight @ value
    print(f'{av[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
    return av, attn_weight

'''
Flash Attention is batch operator that evaluates sm(QK')V
Q = batch_size x ... x seqlen_q x head_size
K = batch_size x ... x seqlen_k x head_size
    => K' = batch_size x ... x head_size x seqlen_k
V = batch_size x ... x seqlen_k x head_size
sm(.) = softmax(.)
The output size is
batch_size x ... x seqlen_q x head_size

Note: In Flash V2 API the ... is denoted as "num_heads", serving as uniformly sized sequences
but in PyTorch API it does not present at all
'''

@pytest.mark.parametrize('Z, H, N_CTX, D_HEAD',
                         [(4, 48, 1024, 64),
                          (4, 48, 2048, 64),
                          # (4, 48, 4096, 64), # TOO large to fit in memory...
                          (8, 4, 256, 16),
                          (8, 4, 64, 16),
                          (8, 4, 256, 64),
                          (1, 1, 128, 64),
                          (1, 1, 128, 128),
                          # (1, 1, 64, 64),
                          (1, 1, 96, 64),
                          (1, 1, 16, 32),
                          (1, 1, 16, 16),
                          (1, 1, 1, 16),
                          (1, 1, 7, 16),
                          (8, 4, 1, 16),
                          (8, 4, 7, 16),
                          (4, 48, 1, 16),
                          (4, 48, 7, 16),
                          #(4, 48, 8192, 64),
                          #(4, 48, 16384, 64)
                          ])
@pytest.mark.parametrize('causal', [False, True])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('sm_scale', [0.5, 0.0])
@pytest.mark.parametrize('qseqlen_override', [None, 512])
@pytest.mark.parametrize('dropout_p', [0.0, 0.3, 0.5])
def test_op_fwd(Z, H, N_CTX, D_HEAD, causal, sm_scale, dropout_p, dtype, qseqlen_override):
    torch.manual_seed(20)
    qseqlen = N_CTX if qseqlen_override is None else qseqlen_override
    kseqlen = N_CTX
    if qseqlen_override is not None and N_CTX < 16:
        pytest.skip("Do not qseqlen_override + odd seqlen")
    print(f"test_op_fwd {Z=}, {H=}, {qseqlen=}, {kseqlen=}, {D_HEAD=}, {causal=}")
    SPARSE_HEAD_SINCE = 3
    SPARSE_SEQ_SINCE = 3
    if True: # Real UT
        q = (
            torch.empty((Z, H, qseqlen, D_HEAD), dtype=dtype, device="cuda")
            .normal_(mean=0., std=0.5)
            .requires_grad_()
        )
        k = (
            torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda")
            .normal_(mean=0., std=0.5)
            .requires_grad_()
        )
        v = (
            torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda")
            .normal_(mean=0., std=0.5)
            .requires_grad_()
        )
    if False: # Debugging
        q = (
            torch.empty((Z, H, qseqlen, D_HEAD), dtype=dtype, device="cuda")
            .normal_(mean=0., std=0.5)
            .requires_grad_()
        )
        k = torch.ones((Z, H, kseqlen, D_HEAD), dtype=dtype, device="cuda") * 1.0
        v = torch.ones((Z, H, kseqlen, D_HEAD), dtype=dtype, device="cuda") * 1.0
    if False:
        q = torch.ones((Z, H, qseqlen, D_HEAD), dtype=dtype, device="cuda") * 1.0
        k = torch.ones((Z, H, kseqlen, D_HEAD), dtype=dtype, device="cuda") * 2.0
        v = torch.ones((Z, H, kseqlen, D_HEAD), dtype=dtype, device="cuda") * 3.0
    if False:
        import numpy as np
        q = torch.arange(np.prod([Z, H, qseqlen, D_HEAD]), dtype=dtype, device="cuda").reshape((Z, H, qseqlen, D_HEAD))
        k = torch.arange(np.prod([Z, H, kseqlen, D_HEAD]), dtype=dtype, device="cuda").reshape((Z, H, qseqlen, D_HEAD))
        v = torch.arange(np.prod([Z, H, kseqlen, D_HEAD]), dtype=dtype, device="cuda").reshape((Z, H, qseqlen, D_HEAD))
        q = (q - 128.0) * 0.01
        k = (k - 128.0) * 0.01
        v = (v - 128.0) * 0.01
        q[:, :, :, SPARSE_HEAD_SINCE: ] = 0.0
        k[:, :, :, SPARSE_HEAD_SINCE: ] = 0.0
        v[:, :, :, SPARSE_HEAD_SINCE: ] = 0.0
        q[:, :, SPARSE_SEQ_SINCE:, : ] = 0.0
        k[:, :, SPARSE_SEQ_SINCE:, : ] = 0.0
        v[:, :, SPARSE_SEQ_SINCE:, : ] = 0.0

    '''
    dout = torch.randn_like(q)
    # reference implementation
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    ref_out = torch.matmul(p, v)
    '''
    tri_out, encoded_softmax, _ = attention(q, k, v, causal, sm_scale, dropout_p, True)

    dropout_mask = encoded_softmax >= 0
    # assert torch.allclose(dropout_mask, dropout_mask_naive)
    ref_out, ref_softmax = torch.ops.aten._scaled_dot_product_attention_math(q, k, v,
                                                                dropout_p=dropout_p,
                                                                is_causal=causal,
                                                                scale=sm_scale,
                                                                dropout_mask=dropout_mask)
    if False:
        mref_out, mref_softmax = scaled_dot_product_attention(q, k, v,
                                     dropout_p=dropout_p,
                                     is_causal=causal,
                                     scale=sm_scale,
                                     dropout_mask=dropout_mask)
        print(f'{tri_out[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
        print(f'{ref_out[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
        print(f'{mref_out[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
        print(f'{tri_out[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]/ref_out[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
        print(f'{q.shape=} {q.stride()=}')
        print(f'{k.shape=} {k.stride()=}')
        print(f'{v.shape=} {v.stride()=}')
        print(f'{encoded_softmax.shape=} {encoded_softmax.stride()=}')
        print(f'{encoded_softmax[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_SEQ_SINCE]=}')
        print(f'{dropout_mask.shape=} {dropout_mask.stride()=}')
        print(f'{dropout_mask[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
    if dtype==torch.bfloat16:
        ATOL = 1e-1 * (qseqlen / 128.0) if qseqlen >= 16 else 1e-1
    else:
        ATOL = 1e-2 * (qseqlen / 128.0) if qseqlen >= 16 else 1e-2
    print(f'Using ATOL={ATOL}')
    is_allclose = torch.allclose(ref_out, tri_out, atol=ATOL, rtol=0)
    if not is_allclose:
        import numpy as np
        err_idx = np.unravel_index(torch.argmax(torch.abs(ref_out - tri_out)).cpu().numpy(), ref_out.shape)
        print(f'{err_idx=}')
        print(f'{tri_out[err_idx]=} {ref_out[err_idx]=} error: {tri_out[err_idx] - ref_out[err_idx]}')
    # if not is_allclose:
    if False:
        import numpy as np
        err_idx = np.unravel_index(torch.argmax(torch.abs(ref_out - tri_out)).cpu().numpy(), ref_out.shape)
        print(f'{tri_out[0][0][0][:]=}')
        print(f'{ref_out[0][0][0][:]=}')
        print(f'{mref_out[0][0][0][:]=}')
        print(f'{encoded_softmax[0][0][0][:]=}')
        print(f'{ref_softmax[0][0][0][:]=}')
        print(f'{tri_out[-1][0][0][:]=}')
        print(f'{ref_out[-1][0][0][:]=}')
        print(f'{err_idx=}')
        print(f'{tri_out[err_idx]=}')
        print(f'{ref_out[err_idx]=}')
        if dropout_p > 0:
            # print(f'{unmasked_ref_out[0][0][0][:]=}')
            print(f'{dropout_mask[0][0][0][:]=}')
            print(f'{dropout_mask[err_idx]=}')
        # tri_cpu = tri_out[0, 0].cpu().detach().numpy()
        # print(f'{tri_cpu.shape=}')
    # compare
    assert is_allclose
