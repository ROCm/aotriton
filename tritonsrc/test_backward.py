#!/usr/bin/env python

import pytest
import torch

from attn_torch_function import attention

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
                         [(4, 32, 1024, 64),
                          (4, 32, 2048, 64),
                          # (4, 32, 4096, 64),
                          (8, 4, 256, 16),
                          (8, 4, 64, 16),
                          (8, 4, 256, 64),
                          (1, 1, 16, 16),
                          (1, 1, 16, 32),
                          (1, 1, 16, 64),
                          (1, 1, 32, 16),
                          (1, 1, 32, 32),
                          (1, 1, 32, 64),
                          (1, 1, 64, 16),
                          (1, 1, 64, 32),
                          (1, 1, 64, 64),
                          (1, 1, 128, 16),
                          (1, 1, 128, 32),
                          (1, 1, 128, 64),
                          (1, 1, 256, 16),
                          (1, 1, 256, 32),
                          (1, 1, 256, 64),
                          (1, 8, 256, 64),
                          #(4, 48, 8192, 64),
                          #(4, 48, 16384, 64)
                          ])
@pytest.mark.parametrize('causal', [False, True])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('sm_scale', [1.0, 0.5, 0.0])
# @pytest.mark.parametrize('qseqlen_override', [None, 16])
@pytest.mark.parametrize('qseqlen_override', [None])
@pytest.mark.parametrize('dropout_p', [0.0, 0.3, 0.5])
def test_op_bwd(Z, H, N_CTX, D_HEAD, causal, sm_scale, dropout_p, dtype, qseqlen_override):
    torch.manual_seed(20)
    qseqlen = N_CTX if qseqlen_override is None else qseqlen_override
    kseqlen = N_CTX
    SPARSE_HEAD_SINCE = 1
    SPARSE_SEQ_SINCE = 1
    if True: # Real UT
        q = torch.empty((Z, H, qseqlen, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5)
        k = torch.empty((Z, H, kseqlen, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5)
        v = torch.empty((Z, H, kseqlen, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5)
    if False: # Debugging
        q[:, :, :, SPARSE_HEAD_SINCE: ] = 0.0
        k[:, :, :, SPARSE_HEAD_SINCE: ] = 0.0
        v[:, :, :, SPARSE_HEAD_SINCE: ] = 0.0
        q[:, :, SPARSE_SEQ_SINCE:, : ] = 0.0
        k[:, :, SPARSE_SEQ_SINCE:, : ] = 0.0
        v[:, :, SPARSE_SEQ_SINCE:, : ] = 0.0
    if False: # Debugging
        q = torch.ones((Z, H, qseqlen, D_HEAD), dtype=dtype, device="cuda") * 1.0
        k = torch.ones((Z, H, kseqlen, D_HEAD), dtype=dtype, device="cuda") * 2.0
        v = torch.ones((Z, H, kseqlen, D_HEAD), dtype=dtype, device="cuda") * 3.0
        q[:, :, :, SPARSE_HEAD_SINCE: ] = 0.0
        k[:, :, :, SPARSE_HEAD_SINCE: ] = 0.0
        v[:, :, :, SPARSE_HEAD_SINCE: ] = 0.0
        q[:, :, SPARSE_SEQ_SINCE:, : ] = 0.0
        k[:, :, SPARSE_SEQ_SINCE:, : ] = 0.0
        v[:, :, SPARSE_SEQ_SINCE:, : ] = 0.0
    '''
    q = torch.ones((Z, H, qseqlen, D_HEAD), dtype=dtype, device="cuda") * 1.0
    k = torch.ones((Z, H, kseqlen, D_HEAD), dtype=dtype, device="cuda") * 2.0
    v = torch.ones((Z, H, kseqlen, D_HEAD), dtype=dtype, device="cuda") * 3.0
    '''
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    if causal == False:
        split_kernel = False
    else: # split kernel only handles for causal=True
        split_kernel = True
    if True: # Real UT
        dout = torch.randn_like(q)
    if False: # Debugging
        dout = torch.ones_like(q) * 1.0
    if False:
        # OLD reference implementation
        M = torch.tril(torch.ones((qseqlen, kseqlen), device="cuda"))
        p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
        if causal:
            p[:, :, M == 0] = float("-inf")
        p = torch.softmax(p.float(), dim=-1).to(dtype=dtype)
        unmasked_ref_out = torch.matmul(p, v)
    # # triton implementation
    tri_out, encoded_softmax = attention(q, k, v, causal, sm_scale, dropout_p, True, split_kernel)
    dropout_mask = encoded_softmax >= 0
    ref_out, ref_softmax = torch.ops.aten._scaled_dot_product_attention_math(q, k, v,
                                                                dropout_p=dropout_p,
                                                                is_causal=causal,
                                                                scale=sm_scale,
                                                                dropout_mask=dropout_mask)
    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None
    ref_out.backward(dout, None)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None
    # compare
    if False and q.shape[-2] <= 16 and q.shape[-1] <= 16:
        print(f'{tri_out[0][0][:][:]=}')
        print(f'{ref_out[0][0][:][:]=}')
    RTOL=1e-2 if dtype==torch.float16 else 5e-2
    # FIXME: Need to raise tolerance
    is_allclose = torch.allclose(ref_out, tri_out, atol=5e-2, rtol=RTOL)
    if not is_allclose:
        import numpy as np
        err_idx = np.unravel_index(torch.argmax(torch.abs(ref_out - tri_out)).cpu().numpy(), ref_out.shape)
        print(f'{err_idx=}')
        print(f'{tri_out[err_idx]=}')
        print(f'{ref_out[err_idx]=}')
    assert is_allclose
    if False:
        qk_scale = sm_scale * 1.44269504
        qk = q[0,0] @ k[0,0].transpose(-1, -2)
        qk_bad = q[0,0] @ k[0,0]
        print(f'{q[0,0,0,:].dot(k[0,0,0,:])=}')
        print(f'{q[0,0][:4, :4]=}')
        print(f'{k[0,0][:4, :4]=}')
        print(f'Manual {qk[:4, :4]=}')
        print(f'Manual {qk_bad[:4, :4]=}')
        print(f'Triton qk {tri_dq[0,0][:4, :4]=}')
        # assert False
        # l_i = tl.load(l_ptrs + offs_m_curr)
        # p = tl.math.exp2(qk * qk_scale - l_i[:, None])
        # do = tl.load(do_ptrs)
        # dv += tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do)
    if dtype == torch.bfloat16:
        ATOL = 1e-1 * ((qseqlen + D_HEAD) / 32.0)
    if dtype == torch.float32:
        ATOL = 1e-3 * ((qseqlen + D_HEAD) / 32.0)
    else:
        ATOL = 1e-1 * ((qseqlen + D_HEAD) / 32.0)
    print(f"{ATOL=} {RTOL=}")

    dk_allclose = torch.allclose(ref_dk, tri_dk, atol=ATOL, rtol=RTOL)
    if not dk_allclose or True:
        print(f'{tri_out[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
        print(f'{ref_out[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
        print(f'{tri_dk[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
        print(f'{ref_dk[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
        import numpy as np
        err_idx = np.unravel_index(torch.argmax(torch.abs(ref_dk - tri_dk)).cpu().numpy(), ref_dk.shape)
        print(f'{err_idx=}')
        print(f'{tri_dk[err_idx]=} {ref_dk[err_idx]=} error = {torch.abs(tri_dk[err_idx] - ref_dk[err_idx])}')
        print(f'{tri_dk[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]/ref_dk[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
        print(f'{dropout_mask[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
    assert dk_allclose

    dv_allclose = torch.allclose(ref_dv, tri_dv, atol=ATOL, rtol=RTOL)
    if not dv_allclose:
        import numpy as np
        err_idx = np.unravel_index(torch.argmax(torch.abs(ref_dv - tri_dv)).cpu().numpy(), ref_dv.shape)
        print(f'{q.shape=} {q.stride()=}')
        print(f'{k.shape=} {k.stride()=}')
        print(f'{v.shape=} {v.stride()=}')
        print(f'{q[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
        print(f'{k[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
        print(f'{v[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
        # print(f'{dropout_mask[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
        print(f'{dropout_mask.shape=}')
        '''
        print(f'{dropout_mask=}')
        if qseqlen >= 16:
            print(f'{dropout_mask[0, 0, 16:, :]=}')
        '''
        print(f'{err_idx=}')
        print(f'{tri_dv[err_idx]=}')
        print(f'{ref_dv[err_idx]=}')
        print(f'{tri_dv[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
        print(f'{ref_dv[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
        print(f'{tri_dv[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]/ref_dv[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
        if qseqlen >= 128:
            print(f'{tri_dv[:,:, 107]=}')
            print(f'{ref_dv[:,:, 107]=}')
            print(f'{tri_dv[:,:, 96]=}')
            print(f'{ref_dv[:,:, 96]=}')
            print(f'{tri_dv[:,:, 64]=}')
            print(f'{ref_dv[:,:, 64]=}')
            dv_allclose_rowwise = [torch.allclose(ref_dv[0, 0, i], tri_dv[0, 0, i], atol=ATOL, rtol=RTOL) for i in range(qseqlen)]
            dv_maxerr_rowwise = [torch.max(torch.abs(ref_dv[0, 0, i] - tri_dv[0, 0, i])) for i in range(qseqlen)]
            print('dv_allclose_rowwise:', end=' ')
            for i, c in enumerate(dv_allclose_rowwise):
                print(f'{i}: {c}', end=' ')
            print()
            print('dv_maxerr_rowwise:', end=' ')
            for i, c in enumerate(dv_maxerr_rowwise):
                print(f'{i}: {c}', end=' ')
            print()
    assert dv_allclose
    # dq_allclose = torch.allclose(ref_dq, tri_dq, atol=5e-2, rtol=0)
    # FIXME: Need to raise tolerance
    # print(f'{tri_dv[0][0]=}')
    # print(f'{ref_dv[0][0]=}')
    # print(f'{tri_dv[0][0][:4, :4]=}')
    # print(f'{ref_dv[0][0][:4, :4]=}')
    # print(f'{tri_dk[0][0][:4, :4]=}')
    # print(f'{ref_dk[0][0][:4, :4]=}')
    dq_allclose = torch.allclose(ref_dq, tri_dq, atol=ATOL, rtol=RTOL)
    if not dq_allclose:
        import numpy as np
        err_idx = np.unravel_index(torch.argmax(torch.abs(ref_dq - tri_dq)).cpu().numpy(), ref_dq.shape)
        print(f'{err_idx=}')
        # print(f'{tri_dq[err_idx]=}')
        # print(f'{ref_dq[err_idx]=}')
        print(f'{tri_dq[err_idx]=} {ref_dq[err_idx]=} error = {torch.abs(tri_dq[err_idx] - ref_dq[err_idx])}')
        # print(f'{tri_dq[0][0][:4, :]=}')
        # print(f'{ref_dq[0][0][:4, :]=}')
        # print(f'{tri_dq[0][0][:4, :4]=}')
        # print(f'{ref_dq[0][0][:4, :4]=}')
        print(f'{tri_dq[0][0][16, :4]=}')
        print(f'{ref_dq[0][0][16, :4]=}')
        # print(f'{tri_dq[0][0][31, :4]=}')
        # print(f'{ref_dq[0][0][31, :4]=}')
        # print(f'{tri_dq[0][0][32, :4]=}')
        # print(f'{ref_dq[0][0][32, :4]=}')
        # print(f'{tri_dq[0][0][49, :4]=}')
        # print(f'{ref_dq[0][0][49, :4]=}')
    assert dq_allclose
