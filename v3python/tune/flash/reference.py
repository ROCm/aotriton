# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import sys
from dataclasses import dataclass, astuple
from ..kftdesc import KernelForTuneDescription as KFTDesc
import torch
from torch.backends.cuda import allow_fp16_bf16_reduction_math_sdp
from torch.ops import aten
from .utils import (
    round_to_8x,
    sdpa_logsumexp,
)
from ..gpu_utils import (
    elike,
    adiff2,
    strip_grad_l1,
)
from pyaotriton.v2.flash import (
    debug_simulate_encoded_softmax as fa_debug_simulate_encoded_softmax,
)

sdpa_math = aten._scaled_dot_product_attention_math

DEFAULT_PHILOX_SEED = 0x1BF52
DEFAULT_PHILOX_OFFSET_1 = 0x1D4000
DEFAULT_PHILOX_OFFSET_2 = 0x000B42
DEFAULT_PHILOX_OFFSET = DEFAULT_PHILOX_OFFSET_1 + DEFAULT_PHILOX_OFFSET_2

class WindowValue:
    NONE = 0
    TOP_LEFT_ALIGNED = -2147483647       # 0x80000001. Special value for varlen
    BOTTOM_RIGHT_ALIGNED = -2147483646   # 0x80000002. Special value for varlen

'''
Note the order is different from Triton Kernel, to follow @dataclass
requirements about non-default arguments
'''
@dataclass
class SdpaBidiInputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    sm_scale: float
    # BWD inputs are also included here, even if they are outputs of FWD
    out: torch.Tensor
    logsumexp: torch.Tensor
    dout: torch.Tensor
    b: torch.Tensor | None = None
    dropout_p: float = 0.0
    seed: torch.Tensor | None = None
    offset1: torch.Tensor | None = None
    offset2: int = 0
    seedout: torch.Tensor | None = None
    offsetout: torch.Tensor | None = None
    encoded_softmax: torch.Tensor | None  = None    # Unused by attn_fwd, but others needs it
    window_sizes: tuple[int, int] | None = 0        # GENERALIZED SWA
    atomic: torch.Tensor|None = None                # Do we need to keep it?

# Note this class captures "golden" outputs, e.g.,
#   fp32 for fp16/bf16 inputs,
#   fp64 for fp32 inputs
@dataclass
class SdpaGoldenOutputs:
    out: tuple[torch.Tensor, float]
    dq: tuple[torch.Tensor, float]
    dk: tuple[torch.Tensor, float]
    dv: tuple[torch.Tensor, float]
    db: tuple[torch.Tensor, float] | None

# ret = attn_bwd(q, k, v, b, sm_scale, o, do, dq, dk, dv, db, dq_acc, L, delta,
#                dropout_p, philox_seed, philox_offset, 0, causal,
#                extargs=extargs, call_operator=V3_API)

'''
Always use default device.
If caller wants to specify device, use torch.device (NOT torch.cuda.device) ctx mananger
'''
class SdpaReference(KFTDesc):

    PT_INPUT_CLASS = SdpaBidiInputs
    PT_REF_CLASS = SdpaGoldenOutputs

    @property
    def device(self):
        return f'cuda:{torch.cuda.current_device()}'

    def create_extargs(self, *, force_kernel_index=None, peek_kernel_numbers=None):
        return None

    def generate_inputs(self, im: 'FlashInputMetadata', *, dry_run=False):
        dtype_str, D_HEAD, seqlen_q, seqlen_k, causal, dropout_p, bias_type, N_HEADS, BATCH, sm_scale, storage_flip, prng_seed = astuple(im)
        dtype = getattr(torch, dtype_str)
        if isinstance(N_HEADS, int):
            Q_HEADS = K_HEADS = N_HEADS
        else:
            Q_HEADS, K_HEADS = N_HEADS
        if isinstance(D_HEAD, int):
            D_HEAD_V = D_HEAD
        else:
            D_HEAD, D_HEAD_V = D_HEAD
        qdims = (BATCH, Q_HEADS, seqlen_q, D_HEAD)
        kdims = (BATCH, K_HEADS, seqlen_k, D_HEAD)
        vdims = (BATCH, K_HEADS, seqlen_k, D_HEAD_V)
        bdims = (BATCH, Q_HEADS, seqlen_q, round_to_8x(seqlen_k))
        odims = (BATCH, Q_HEADS, seqlen_q, D_HEAD_V)
        if storage_flip:
            order = [0,1,2,3]
            x, y = storage_flip if isinstance(storage_flip, tuple) else [1, 2]
            assert x != 3 and y != 3, 'Cannot storage_flip last dimension. Last dimension must be continuous'
            order[x], order[y] = order[y], order[x]
            i, j, k, l = order
            qdims = (qdims[i], qdims[j], qdims[k], qdims[l])
            kdims = (kdims[i], kdims[j], kdims[k], kdims[l])
            vdims = (vdims[i], vdims[j], vdims[k], vdims[l])
            bdims = (bdims[i], bdims[j], bdims[k], bdims[l])
        g = torch.Generator(device=self.device)
        g.manual_seed(prng_seed)
        def rng(dims):
            return torch.rand(*dims, generator=g, dtype=dtype)
        q = rng(qdims)
        k = rng(kdims)
        v = rng(vdims)
        b = None
        if bias_type == 'matrix' or bias_type == 1:
            b = rng(bdims)
            b = b[:, :, :, :seqlen_k]
        if storage_flip:
            x, y = storage_flip if isinstance(storage_flip, tuple) else [1, 2]
            def tt(t):
                return torch.transpose(t, x, y) if t is not None else None
            q = tt(q)
            k = tt(k)
            v = tt(v)
            b = tt(b)
        if sm_scale == 'l1':
            sm_scale = 1.0 / D_HEAD
        elif sm_scale == 'l2':
            sm_scale = 1.0 / math.sqrt(D_HEAD)
        L = torch.empty((q.shape[0], q.shape[1], q.shape[2]), dtype=torch.float32)
        out = torch.empty(odims, dtype=q.dtype)
        dout = rng(odims)
        philox_null = torch.empty([0], dtype=torch.uint64)
        philox_seed = philox_null
        philox_offset1 = philox_null
        philox_offset2 = 0
        philox_seed_output = philox_null
        philox_offset_output = philox_null
        encoded_softmax = None
        if dropout_p > 0.0:
            philox_seed = torch.tensor([DEFAULT_PHILOX_SEED], dtype=torch.uint64)
            philox_offset1 = torch.tensor([DEFAULT_PHILOX_OFFSET_1], dtype=torch.uint64)
            philox_offset2 = DEFAULT_PHILOX_OFFSET_2
            philox_seed_output = torch.tensor([DEFAULT_PHILOX_SEED], dtype=torch.uint64)
            philox_offset_output = torch.tensor([DEFAULT_PHILOX_OFFSET], dtype=torch.uint64)
            encoded_softmax = torch.zeros((q.shape[0], q.shape[1], q.shape[2], k.shape[2]), dtype=q.dtype)

        if causal:
            window_sizes = (WindowValue.BOTTOM_RIGHT_ALIGNED, WindowValue.BOTTOM_RIGHT_ALIGNED)
            atomic = torch.zeros([1], dtype=torch.int32)
        else:
            window_sizes = None
            atomic = torch.empty([0], dtype=torch.int32)

        inputs = SdpaBidiInputs(q=q,
                                k=k,
                                v=v,
                                sm_scale=sm_scale,
                                out=out,
                                logsumexp=L,
                                dout=dout,
                                b=b,
                                dropout_p=dropout_p,
                                seed=philox_seed,
                                offset1=philox_offset1,
                                offset2=philox_offset2,
                                seedout=philox_seed_output,
                                offsetout=philox_offset_output,
                                encoded_softmax=encoded_softmax,
                                window_sizes=window_sizes,
                                atomic=atomic)
        return inputs

    def prepare_directs(self, im, inputs):
        return im, inputs

    def fill_nan_to_outputs(self, direct_inputs):
        pass

    def direct_call(self, direct_inputs, extargs):
        im, inputs = direct_inputs
        assert extargs is None
        q = inputs.q
        k = inputs.k
        v = inputs.v
        b = inputs.b
        hp_dtype = torch.float64 if q.dtype == torch.float32 else torch.float32
        def clone_hp(t):
            if t is None:
                return None
            t.requires_grad_()
            return t.clone().detach().to(dtype=hp_dtype).requires_grad_()
        hpq = clone_hp(q)
        hpk = clone_hp(k)
        hpv = clone_hp(v)
        hpb = clone_hp(b)
        enable_gqa = q.shape[1] != k.shape[1]
        is_causal = inputs.window_sizes is not None
        # print(f"{q.shape=} {k.shape=} {v.shape=} {enable_gqa=}")
        if inputs.dropout_p > 0.0:
            from aotriton_flash import (
                debug_simulate_encoded_softmax,
                hipError_t,
            )
            debug_simulate_encoded_softmax(inputs.encoded_softmax,
                                           inputs.dropout_p,
                                           inputs.seed,
                                           inputs.offset1,
                                           inputs.offset2)
        out, _ = sdpa_math(q,
                           k,
                           v,
                           attn_mask=b,
                           scale=inputs.sm_scale,
                           is_causal=is_causal,
                           dropout_p=inputs.dropout_p,
                           dropout_mask=inputs.encoded_softmax,
                           enable_gqa=enable_gqa)
        hpout, _ = sdpa_math(hpq,
                             hpk,
                             hpv,
                             attn_mask=hpb,
                             scale=inputs.sm_scale,
                             is_causal=is_causal,
                             dropout_p=inputs.dropout_p,
                             dropout_mask=inputs.encoded_softmax,
                             enable_gqa=enable_gqa)
        logsumexp = sdpa_logsumexp(hpq,
                                   hpk,
                                   hpv,
                                   attn_mask=hpb,
                                   scale=inputs.sm_scale,
                                   is_causal=is_causal,
                                   enable_gqa=enable_gqa)
        inputs.out = hpout.to(inputs.q.dtype)
        inputs.logsumexp = logsumexp.to(torch.float32)
        out.backward(inputs.dout)
        hpout.backward(inputs.dout.to(dtype=hpout.dtype))
        outputs = SdpaGoldenOutputs(out=adiff2(hpout, out),
                                    dq=strip_grad_l1(hpq, q),
                                    dk=strip_grad_l1(hpk, k),
                                    dv=strip_grad_l1(hpv, v),
                                    db=strip_grad_l1(hpb, b))
        return inputs, outputs

    def compare(self, outputs, refs) -> list[float]:    # L1 error
        raise RuntimeError("Should not call SdpaReference.compare. Call compare() with attn_fwd/etc. objects")
