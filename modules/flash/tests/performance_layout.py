# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Copy-inclusive forward benchmarks; run with pytest -s performance_layout.py.

Timings are observations for the same-machine performance baseline, not portable
pass/fail limits. Long-sequence references sample query rows but use every key.
"""

import json

import pytest
import torch
import triton

from attn_torch_function import attention, AttentionExtraArgs
from pyaotriton.v3.flash import OpAttnFwdBackend


@pytest.mark.parametrize('heads,seqlen', [(4, 256), (32, 2048), (32, 6630),
                                         (32, 15300), (56, 15787), (40, 33600)])
@pytest.mark.parametrize('layout', ['contiguous', 'transposed', 'packed_qkv'])
def test_forward_layout_copy_cost(heads, seqlen, layout):
    torch.manual_seed(0)
    shape = (1, heads, seqlen, 128)
    if layout == 'packed_qkv':
        packed = torch.randn((1, seqlen, 3, heads, 128), device='cuda', dtype=torch.bfloat16)
        inputs = tuple(packed[:, :, i].transpose(1, 2) for i in range(3))
    else:
        inputs = tuple(torch.randn(shape, device='cuda', dtype=torch.bfloat16) for _ in range(3))
        if layout == 'transposed':
            inputs = tuple(x.transpose(1, 2).contiguous().transpose(1, 2) for x in inputs)
    contiguous = tuple(x.contiguous() for x in inputs)
    backend = next(index for index, name in OpAttnFwdBackend.by_index.items() if name == 'triton')
    extra = AttentionExtraArgs(force_fwd_backend_index=backend, is_testing=False)

    def run(tensors):
        return attention(*tensors, None, False, 128 ** -0.5, 0.0, extra)[0]

    paths = {
        'contiguous': lambda: run(contiguous),
        'input_layout': lambda: run(inputs),
        'copy_qkv_attention': lambda: run(tuple(x.contiguous() for x in inputs)),
        'copy_kv_attention': lambda: run((inputs[0], inputs[1].contiguous(), inputs[2].contiguous())),
    }
    expected = paths['contiguous']()
    # Bound CPU memory for long sequences. Every key/value participates for
    # three heads and three query positions, including the final partial tile.
    selected_heads = sorted({0, heads // 2, heads - 1})
    rows = sorted({0, seqlen // 2, seqlen - 1})
    q = inputs[0][:, selected_heads][:, :, rows].cpu().float()
    k, v = (x[:, selected_heads].cpu().float() for x in inputs[1:])
    reference = torch.softmax(q @ k.transpose(-2, -1) * (128 ** -0.5), dim=-1) @ v
    torch.testing.assert_close(expected[:, selected_heads][:, :, rows].cpu().float(),
                               reference,
                               atol=3e-2,
                               rtol=3e-2)
    for path, fn in paths.items():
        actual = fn()
        torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
        del actual
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        memory_before = torch.cuda.memory_allocated()
        output = fn()
        torch.cuda.synchronize()
        peak_bytes = torch.cuda.max_memory_allocated() - memory_before
        del output
        # do_bench synchronizes GPU events; copies remain inside the timed call.
        milliseconds = triton.testing.do_bench(fn, warmup=25, rep=100)
        print(json.dumps({'shape': shape,
                          'dtype': str(inputs[0].dtype),
                          'layout': layout,
                          'strides': [x.stride() for x in inputs],
                          'storage_offsets': [x.storage_offset() for x in inputs],
                          'backend': 'triton',
                          'path': path,
                          'milliseconds': milliseconds,
                          'peak_allocated_bytes': peak_bytes}))
