# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
The byte model and the size rule of flash perfmon.

The byte model counts every device buffer the runner's adapter allocates for
one call, exactly: no reference tensors and no caching allocator stand
between it and the GPU. That is why it is not tune's memory model
(`modules/flash/tune/desc.py`'s `_clamp_memory_usage`): tune's budget is
dominated by the torch reference that materialises Sq x Sk scores for its
accuracy check, which perfmon never computes. The two answer different
questions and are kept apart on purpose; this file is the only copy of the
adapter's model.

    dt          2 for fp16/bf16, 4 for fp32
    q           B * Hq * Sq * Dqk          Q, dQ (and fp32 dq_acc)
    k           B * Hkv * Sk * Dqk         K, dK
    v           B * Hkv * Sk * Dv          V, dV
    o           B * Hq * Sq * Dv           Out, dO
    lse         B * Hq * Sq                L (forward); L and delta (backward), fp32
    bias        b * h * Sq * round8(Sk)    real dimensions only: b = B or 1, h = Hq or 1

    fwd = dt * (q + k + v + o) + 4 * lse + dt * bias + 4 (causal counter)
    bwd = dt * 2 * (q + k + v + o) + 8 * lse + dt * 2 * bias + 4 * q + 4

`dq_acc` is counted on every backward call, though only the fused backend
allocates it, so that every backend of an entry is measured on the same
entry. Dropout costs nothing: the backward regenerates the mask from the
Philox seed and offset.

The size rule gives every grid entry fixed sizes from the problem and the
seqlens alone, chosen to fill a 32 GiB GPU with margin, so that each
measurement runs on a large problem and the same entry is the same call on
every GPU. Changing a constant below changes `entries.csv` (new rows), which is
visible in review; old results keep their meaning.
"""

GiB = 1 << 30

#: n_heads_q * hdim_qk: a constant hidden size keeps head counts realistic.
HIDDEN_SIZE = 2048
#: What the inputs may fill. 32 GiB less the 1 GiB L2-flush buffer, the HIP
#: context and runtime allocations, and allocator slack.
# TODO(perfmon): phase 1 confirms on a 32 GiB GPU that no entry fails to allocate.
BUDGET_BYTES = 26 * GiB
#: Short sequences would otherwise get launch grids beyond what every
#: release's kernels accept.
# TODO(perfmon): phase 1 confirms every release accepts this cap.
BATCH_CAP = 4096

#: The worker's fit check: what the runner holds besides the entry's buffers.
L2_FLUSH_BYTES = 1 * GiB
# TODO(perfmon): HIP context and allocator slack, to be measured in phase 1.
RUNTIME_RESERVE_BYTES = 1 * GiB


def round_to_8x(n: int) -> int:
    """The bias is allocated with its last dimension padded to a multiple of 8
    and viewed as Sk, so the padded size is what occupies memory."""
    return 8 * ((n + 7) // 8)


def dtype_bytes(t_io: str) -> int:
    return 4 if t_io == 'fp32' else 2


def _terms(e) -> dict:
    hq, hkv = e.n_heads_q, e.n_heads_q // e.gqa_ratio
    b, sq, sk = e.batch, e.seqlen_q, e.seqlen_k
    if e.bias_shape == '-':
        bias = 0
    else:
        bias_b = b if e.bias_shape[0] == 'b' else 1
        bias_h = hq if e.bias_shape[1] == 'h' else 1
        bias = bias_b * bias_h * sq * round_to_8x(sk)
    return {
        'dt': dtype_bytes(e.T_io),
        'q': b * hq * sq * e.hdim_qk,
        'k': b * hkv * sk * e.hdim_qk,
        'v': b * hkv * sk * e.hdim_v,
        'o': b * hq * sq * e.hdim_v,
        'lse': b * hq * sq,
        'bias': bias,
    }


def fwd_bytes(e) -> int:
    t = _terms(e)
    return (t['dt'] * (t['q'] + t['k'] + t['v'] + t['o'])
            + 4 * t['lse']
            + t['dt'] * t['bias']
            + 4)


def bwd_bytes(e) -> int:
    t = _terms(e)
    return (t['dt'] * 2 * (t['q'] + t['k'] + t['v'] + t['o'])
            + 8 * t['lse']
            + t['dt'] * 2 * t['bias']
            + 4 * t['q']
            + 4)


def input_bytes(e, iface: str) -> int:
    if iface == 'op_attn_fwd':
        return fwd_bytes(e)
    if iface == 'op_attn_bwd':
        return bwd_bytes(e)
    raise ValueError(f'unknown iface {iface!r}')


def initial_heads(hdim_qk: int, gqa_ratio: int) -> int:
    """HIDDEN_SIZE / hdim_qk, rounded down to a multiple of gqa_ratio, and at
    least gqa_ratio."""
    heads = HIDDEN_SIZE // hdim_qk
    return max(gqa_ratio, heads // gqa_ratio * gqa_ratio)


def size_rule(make_entry, hdim_qk: int, gqa_ratio: int) -> tuple[int, int] | None:
    """(batch, n_heads_q) by the rule, where `make_entry(batch, n_heads_q)`
    builds the candidate entry: the largest power-of-two batch up to
    BATCH_CAP whose larger (backward) byte count fits BUDGET_BYTES; if even
    batch 1 does not fit, halve the heads (keeping them a multiple of
    gqa_ratio) and try again. None when nothing fits."""
    heads = initial_heads(hdim_qk, gqa_ratio)
    while heads >= gqa_ratio:
        batch = BATCH_CAP
        while batch >= 1:
            e = make_entry(batch, heads)
            if max(fwd_bytes(e), bwd_bytes(e)) <= BUDGET_BYTES:
                return batch, heads
            batch //= 2
        heads = heads // 2 // gqa_ratio * gqa_ratio
    return None


def fits_vram(e, iface: str, vram_total_bytes: int) -> tuple[bool, str]:
    """Whether the entry fits the GPU it is about to run on. An entry that does
    not fit is excluded, never shrunk: a shrunk entry is a different entry."""
    need = input_bytes(e, iface)
    available = vram_total_bytes - L2_FLUSH_BYTES - RUNTIME_RESERVE_BYTES
    if need <= available:
        return True, ''
    return False, (f'{iface} needs {need / GiB:.2f} GiB, '
                   f'{available / GiB:.2f} GiB available of {vram_total_bytes / GiB:.2f} GiB VRAM')
