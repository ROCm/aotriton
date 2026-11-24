# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host-side launch glue of the gfx942 kernels: the builders' causal types and
argument checks, and the `_launch` callables wrapping each `@flyc.jit` launcher
in gfx1201's kernel ABI.

gfx942-only for now; to be shared with the other arches once their kernels settle.
Called through the module (`host.make_fwd_launch(...)`). No device code here.
"""

import flydsl.expr as fx

import fmha_utils as fmha


def principal_diagonal_bits(bits, principal_diagonal):
    """`bits` with bit 18 set from a launcher's `principal_diagonal=`; raises if `bits` already says otherwise."""
    if principal_diagonal is None:
        return bits
    choices = (fmha.VARLEN_PRINCIPAL_DIAGONAL_TOP_LEFT, fmha.VARLEN_PRINCIPAL_DIAGONAL_BOTTOM_RIGHT)
    if principal_diagonal not in choices:
        raise ValueError(f"principal_diagonal must be VARLEN_PRINCIPAL_DIAGONAL_* or None: {principal_diagonal!r}")
    bit = fmha.VARLEN_PRINCIPAL_DIAGONAL_BOTTOM_RIGHT
    if bits & bit and principal_diagonal != bit:
        raise ValueError("varlen['bits'] selects BOTTOM_RIGHT but principal_diagonal= selects TOP_LEFT")
    return bits | principal_diagonal



def builder_causal_types(causal, causal_type, bias):
    """`(host_causal_type, causal_type)`: 1/2 are host sentinels, 3 is gSWA; all compile to 3."""
    host_causal_type = (1 if causal else 0) if causal_type is None else causal_type
    if host_causal_type not in (0, 1, 2, 3):
        raise ValueError(f'causal_type must be 0-3, got {host_causal_type}')
    causal_type = 0 if host_causal_type == 0 else 3
    if bias and causal_type:
        raise ValueError('bias with a causal mask is not served (aot/_common.py flash_disabled)')
    return host_causal_type, causal_type


def check_head_dims(meta, head_dim, padded_head):
    """Raise unless `fmha.prep_tensors`' (hdim_qk, hdim_vo) fit a `head_dim` build."""
    if padded_head:
        if not (0 < max(meta[2], meta[3]) <= head_dim):
            raise ValueError(f'head dims {meta[2:]} do not fit a padded {head_dim} build')
    elif meta[2] != head_dim or meta[3] != head_dim:
        raise ValueError(f'head dims {meta[2:]} on an unpadded {head_dim} build; pass padded_head=True')


def _check_dropout_p(dropout, dropout_p):
    if dropout_p and not dropout:
        raise ValueError('dropout_p= on a build without dropout')


def make_fwd_launch(launch, tr, causal_type, host_causal_type, bias, dropout):
    """gfx1201/fwd.py's call interface for the `@flyc.jit` `launch`;
    `tr` gives D (the compiled head dim) and PADDED."""
    compiled = fmha.new_compiled_cache()
    has_bias = bias

    def _launch(Q,
                K,
                V,
                O,  # noqa: E741
                batch_size,
                seqlen_q,
                seqlen_k=None,
                num_seqlens=0,
                scale=None,
                stream=None,
                lse=None,
                window=None,
                varlen=None,
                bias=None,
                dropout_p=None,
                philox_seed=0,
                philox_offset1=None,
                philox_offset2=0,
                philox_seed_output=None,
                philox_offset_output=None,
                principal_diagonal=None):
        seqlen_k = seqlen_q if seqlen_k is None else seqlen_k
        ptrs, meta, st = fmha.prep_tensors([("Q", Q), ("K", K), ("V", V), ("O", O)], q_heads=("O",))
        check_head_dims(meta, tr.D, tr.PADDED)
        lse_p = fmha.lse_args(lse, seqlen_q, varlen, meta[0])
        wl, wr = fmha.window_args(causal_type, host_causal_type, window, seqlen_q, seqlen_k)
        vb, sq0, sq1, sk0, sk1, mq, mk = fmha.varlen_args(False, varlen, seqlen_q, seqlen_k, Q, batch_size, num_seqlens)
        vb = principal_diagonal_bits(vb, principal_diagonal)
        bp, _, sb, _ = fmha.bias_args(has_bias, False, bias, None, Q)
        _check_dropout_p(dropout, dropout_p)
        # `hold` keeps a materialised philox scalar alive across the launch (u64_scalar).
        ps, po1, po2, ip, dsc, hold = fmha.dropout_args(dropout,
                                                        dropout_p,
                                                        philox_seed,
                                                        philox_offset1,
                                                        philox_offset2,
                                                        Q.device,
                                                        stream)
        so, oo = fmha.dropout_outputs(dropout, philox_seed_output, philox_offset_output)
        fmha.run_compiled(compiled,
                          launch,
                          *ptrs[:3],
                          bp,
                          ptrs[3],
                          lse_p,
                          sq0,
                          sq1,
                          sk0,
                          sk1,
                          vb,
                          batch_size,
                          num_seqlens,
                          mq,
                          mk,
                          wl,
                          wr,
                          ps,
                          po1,
                          po2,
                          so,
                          oo,
                          ip,
                          dsc,
                          *meta,
                          fmha.resolve_scale(Q, scale, tr.PADDED, tr.D ** -0.5),
                          *st,
                          *sb,
                          fx.Stream(stream))
        del hold

    _launch.traits = tr
    return _launch


def make_dq_launch(launch, tr, causal_type, host_causal_type, bias, dropout):
    """gfx1201/dq.py's `_launch` interface for the `@flyc.jit` `launch`."""
    compiled = fmha.new_compiled_cache()
    has_bias = bias

    def _launch(Q,
                K,
                V,
                DO,
                DQ,
                lse,
                delta,
                batch_size,
                seqlen_q,
                seqlen_k=None,
                num_seqlens=0,
                scale=None,
                stream=None,
                window=None,
                varlen=None,
                principal_diagonal=None,
                dropout_p=None,
                philox_seed=0,
                philox_offset1=None,
                philox_offset2=0,
                bias=None,
                dbias=None):
        seqlen_k = seqlen_q if seqlen_k is None else seqlen_k
        ptrs, meta, st = fmha.prep_tensors([("Q", Q), ("K", K), ("V", V), ("DO", DO), ("DQ", DQ)], q_heads=("DO", "DQ"))
        check_head_dims(meta, tr.D, tr.PADDED)
        lp = fmha.row_tensor_arg(lse, "logsumexp", meta[0], seqlen_q, varlen)
        dp = fmha.row_tensor_arg(delta, "delta", meta[0], seqlen_q, varlen)
        wl, wr = fmha.window_args(causal_type, host_causal_type, window, seqlen_q, seqlen_k)
        vb, sq0, sq1, sk0, sk1, mq, mk = fmha.varlen_args(False, varlen, seqlen_q, seqlen_k, Q, batch_size, num_seqlens)
        vb = principal_diagonal_bits(vb, principal_diagonal)
        _check_dropout_p(dropout, dropout_p)
        ps, po1, po2, ip, dsc, hold = fmha.dropout_args(dropout,
                                                        dropout_p,
                                                        philox_seed,
                                                        philox_offset1,
                                                        philox_offset2,
                                                        Q.device,
                                                        stream)
        bp, dbp, bs, dbs = fmha.bias_args(has_bias, True, bias, dbias, Q)
        fmha.run_compiled(compiled,
                          launch,
                          *ptrs[:3],
                          bp,
                          ptrs[3],
                          ptrs[4],
                          dbp,
                          lp,
                          dp,
                          sq0,
                          sq1,
                          sk0,
                          sk1,
                          vb,
                          batch_size,
                          num_seqlens,
                          mq,
                          mk,
                          wl,
                          wr,
                          ps,
                          po1,
                          po2,
                          ip,
                          dsc,
                          *meta,
                          fmha.resolve_scale(Q, scale, tr.PADDED, tr.D ** -0.5),
                          *st,
                          *bs,
                          *dbs,
                          fx.Stream(stream))
        del hold

    _launch.traits = tr
    return _launch


def make_dkdv_launch(launch, tr, causal_type, host_causal_type, bias, dropout):
    """gfx1201/dkdv.py's `_launch` interface for the `@flyc.jit` `launch`."""
    compiled = fmha.new_compiled_cache()
    has_bias = bias

    def _launch(Q,
                K,
                V,
                DO,
                DK,
                DV,
                L,
                Delta,
                batch_size,
                seqlen_q,
                seqlen_k=None,
                num_seqlens=0,
                scale=None,
                stream=None,
                window=None,
                varlen=None,
                principal_diagonal=None,
                dropout_p=None,
                philox_seed=0,
                philox_offset1=None,
                philox_offset2=0,
                bias=None):
        seqlen_k = seqlen_q if seqlen_k is None else seqlen_k
        ptrs, meta, st = fmha.prep_tensors([("Q", Q), ("K", K), ("V", V), ("DO", DO), ("DK", DK), ("DV", DV)],
                                           q_heads=("DO",),
                                           k_heads=("DK", "DV"))
        check_head_dims(meta, tr.D, tr.PADDED)
        lp = fmha.row_tensor_arg(L, "logsumexp", meta[0], seqlen_q, varlen)
        dp = fmha.row_tensor_arg(Delta, "delta", meta[0], seqlen_q, varlen)
        wl, wr = fmha.window_args(causal_type, host_causal_type, window, seqlen_q, seqlen_k)
        vb, sq0, sq1, sk0, sk1, mq, mk = fmha.varlen_args(False, varlen, seqlen_q, seqlen_k, Q, batch_size, num_seqlens)
        vb = principal_diagonal_bits(vb, principal_diagonal)
        _check_dropout_p(dropout, dropout_p)
        ps, po1, po2, ip, dsc, hold = fmha.dropout_args(dropout,
                                                        dropout_p,
                                                        philox_seed,
                                                        philox_offset1,
                                                        philox_offset2,
                                                        Q.device,
                                                        stream)
        bp, _, bs, _ = fmha.bias_args(has_bias, False, bias, None, Q)
        fmha.run_compiled(compiled,
                          launch,
                          *ptrs[:3],
                          bp,
                          *ptrs[3:],
                          lp,
                          dp,
                          sq0,
                          sq1,
                          sk0,
                          sk1,
                          vb,
                          batch_size,
                          num_seqlens,
                          mq,
                          mk,
                          wl,
                          wr,
                          ps,
                          po1,
                          po2,
                          ip,
                          dsc,
                          *meta,
                          fmha.resolve_scale(Q, scale, tr.PADDED, tr.D ** -0.5),
                          *st,
                          *bs,
                          fx.Stream(stream))
        del hold

    _launch.traits = tr
    return _launch
