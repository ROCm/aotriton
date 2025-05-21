#!/usr/bin/env python
# Copyright ©2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import io
import os
from typing import List, Tuple, Optional
from collections import namedtuple
import numpy as np
import math
import torch
import hashlib

HAS_REDUCED_SDPA = hasattr(torch.backends.cuda, "allow_fp16_bf16_reduction_math_sdp")
# Set this env var when GPU system is not reliable
# In addition to compute reference in CPU. The any other operators like input generations
# are done on CPU as well
AOTRITON_TORCH_ONLY_USE_CPU = bool(int(os.getenv('AOTRITON_TORCH_ONLY_USE_CPU', default='0')))
# Usually we compare with GPU because it is much faster
# Overrides by AOTRITON_TORCH_ONLY_USE_CPU=1
AOTRITON_REF_DEVICE_OPTION = os.getenv('AOTRITON_REF_DEVICE_OPTION', default='default')

def fmt_hdim(val):
    return f'hdim{val}'

def calc_checksums(tensors):
    def checksum(t):
        if t is None:
            return None
        tensor_bytes = io.BytesIO()
        torch.save(t, tensor_bytes)
        return hashlib.blake2s(tensor_bytes.getvalue()).hexdigest()
    ret = [ checksum(t) for t in tensors ]
    # print(f'{ret=}')
    return ret

def allow_fp16_bf16_reduction_math_sdp(v : bool):
    if HAS_REDUCED_SDPA:
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(v)

def sdpa_math(query, key, value, attn_mask=None, dropout_p=0.0, dropout_mask=None, is_causal=False, scale=None, enable_gqa=False):
    if str(torch.__version__) >= '2.5.0':
        allow_fp16_bf16_reduction_math_sdp(True)
        retv = torch.ops.aten._scaled_dot_product_attention_math(query, key, value,
                                                                 dropout_p=dropout_p,
                                                                 is_causal=is_causal,
                                                                 attn_mask=attn_mask,
                                                                 scale=scale,
                                                                 dropout_mask=dropout_mask,
                                                                 enable_gqa=enable_gqa)
        allow_fp16_bf16_reduction_math_sdp(False)
        return retv
    else:
        return torch.ops.aten._scaled_dot_product_attention_math(query, key, value,
                                                                 dropout_p=dropout_p,
                                                                 is_causal=is_causal,
                                                                 attn_mask=attn_mask,
                                                                 scale=scale,
                                                                 dropout_mask=dropout_mask)


def _reference_scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    # Efficient implementation equivalent to the following:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    SPARSE_HEAD_SINCE = 5
    SPARSE_SEQ_SINCE = 5
    # attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    if dropout_p > 0.0:
        if dropout_mask is not None:
            attn_weight.masked_fill_(dropout_mask.logical_not(), float("0.0"))
            value = value / (1 - dropout_p)
        else:
            # assert False, "TESTING dropout_mask code path"
            attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    else:
        # assert False, "TESTING dropout_mask code path"
        pass
    av = attn_weight @ value
    return av, attn_weight

default_atol = {torch.float16: 1e-3, torch.bfloat16: 1e-3, torch.float32: 1e-5}
default_rtol = {torch.float16: 1e-3, torch.bfloat16: 1.6e-2, torch.float32: 1.3e-6}

# def get_rtol(true_value: torch.Tensor, computed_value: torch.Tensor) -> float:
#     deviation = true_value - computed_value
#     deviation = torch.abs(deviation / true_value)
#     # Fill in the nans with the default rtol
#     torch.nan_to_num_(deviation, nan=default_rtol[computed_value.dtype])
#     return deviation.max().item()
#
# def get_atol(true_value: torch.Tensor, computed_value: torch.Tensor) -> float:
#     # Low precision may yield NAN due to numerical instability
#     # See https://github.com/pytorch/pytorch/issues/116176 for a real-world example.
#     # Section 3 in https://arxiv.org/abs/2112.05682v3 explains how accelerated
#     # SDPA does not suffer from it.
#     deviation = torch.nan_to_num(true_value - computed_value)
#     atol = torch.abs(deviation).max().item()
#     return atol
#
# def get_tolerances(
#     true_value: torch.Tensor,
#     computed_value: torch.Tensor,
#     fudge_factor: Optional[float] = None,
# ) -> Tuple[float, float]:
#     """Returns the absolute and relative tolerances for comparing two tensors."""
#     fudge_factor = fudge_factor if fudge_factor is not None else 1.0
#     raw_atol = get_atol(true_value, computed_value)
#     raw_rtol = get_rtol(true_value, computed_value)
#
#     atol = fudge_factor * max(raw_atol, default_atol[computed_value.dtype])
#     rtol = fudge_factor * max(raw_rtol, default_rtol[computed_value.dtype])
#     # torch.isclose() has weird behavior around see:
#     # https://github.com/pytorch/pytorch/issues/102400
#     if rtol > 1e30:
#         rtol = default_rtol[computed_value.dtype]
#     return atol, rtol, raw_atol, raw_rtol

SdpaParams = namedtuple('SdpaParams', ['causal', 'sm_scale', 'dropout_p', 'dropout_mask'])

class SdpaContext(object):
    TENSOR_NAMES = ('q', 'k', 'v', 'b')

    def __init__(self, BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, dtype,
                 bias_type=None, storage_flip=None, device='cuda', fillnan=False,
                 prng_seed=0x9be9_98d4_cf17_5339,
                 with_backward=True,
                 ):
        real_device = 'cpu' if AOTRITON_TORCH_ONLY_USE_CPU else device
        self._real_device = real_device
        self._prng_seed = prng_seed
        self._target_device = device
        self._input_shapes = (BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, dtype, bias_type, storage_flip, device, fillnan, prng_seed, with_backward)
        self._create_inputs()
        # Maximal value from tune_flash.py and table_tool.py --fudge_factor_tolerance 5.0
        # Note: Navi 3x is experimental and YMMV
        self.OUT_FUDGE_FACTOR = 3.0
        if dtype == torch.float32:
            self.OUT_FUDGE_FACTOR = 14.0
        if torch.version.hip:
            if 'gfx90a' in torch.cuda.get_device_properties(0).gcnArchName:
                self.OUT_FUDGE_FACTOR = 12.0
        if AOTRITON_TORCH_ONLY_USE_CPU:
            self.OUT_FUDGE_FACTOR = 12.0


    def _create_inputs(self):
        BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, dtype, bias_type, storage_flip, device, fillnan, prng_seed, with_backward = self._input_shapes
        if isinstance(N_HEADS, int):
            Q_HEADS = K_HEADS = N_HEADS
        else:
            Q_HEADS, K_HEADS = N_HEADS
        qdims = (BATCH, Q_HEADS, seqlen_q, D_HEAD)
        kdims = (BATCH, K_HEADS, seqlen_k, D_HEAD)
        vdims = (BATCH, K_HEADS, seqlen_k, D_HEAD)
        bdims = (BATCH, Q_HEADS, seqlen_q, seqlen_k)
        if storage_flip is not None:
            order = [0,1,2,3]
            x, y = storage_flip
            order[x], order[y] = order[y], order[x]
            i, j, k, l = order
            qdims = (qdims[i], qdims[j], qdims[k], qdims[l])
            kdims = (kdims[i], kdims[j], kdims[k], kdims[l])
            vdims = (vdims[i], vdims[j], vdims[k], vdims[l])
            bdims = (bdims[i], bdims[j], bdims[k], bdims[l])
        # q = torch.empty(qdims, dtype=dtype, device=device).normal_(mean=0., std=0.5)
        # k = torch.empty(kdims, dtype=dtype, device=device).normal_(mean=0., std=0.5)
        # v = torch.empty(vdims, dtype=dtype, device=device).normal_(mean=0., std=0.5)
        g = torch.Generator(device=self._real_device)
        g.manual_seed(self._prng_seed)
        def rng(dims):
            return torch.rand(*dims, generator=g, dtype=dtype, device=self._real_device)
        q = rng(qdims)
        k = rng(kdims)
        v = rng(vdims)
        if bias_type is None or bias_type == 0:
            b = None
        elif bias_type == 'matrix' or bias_type == 1:
            # b = torch.empty(bdims, dtype=dtype, device="cuda").normal_(mean=0., std=0.5)
            b = rng(bdims)
            # b = b.expand(BATCH, Q_HEADS, b.shape[0], b.shape[1])
        else:
            assert False, f'Unsupported bias_type {bias_type}'
        if storage_flip is not None:
            x, y = storage_flip
            q = torch.transpose(q, x, y)
            k = torch.transpose(k, x, y)
            v = torch.transpose(v, x, y)
            if b is not None:
                b = torch.transpose(b, x, y)
        dout = rng(q.shape) if with_backward else None
        self.dev_tensors = ( q, k, v, b )
        self.ddev_tensors = tuple([dout])

    '''
    Create Tensors that will be kept b/w forward and backward pass
    '''
    def create_ctx_tensors(self):
        q, k, v, b = self.dev_tensors
        o = torch.empty_like(q)
        M = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        self.ctx_tensors = (o, M)

    def create_bwd_tensors(self):
        q, k, v, b = self.dev_tensors
        o, L = self.ctx_tensors
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        db = torch.empty_like(b) if b is not None else None
        delta = torch.empty_like(L)
        self.bwd_tensors = (dq, dk, dv, db, delta)

    @staticmethod
    def fillnan(tensors):
        for t in tensors:
            if t is None:
                continue
            t.fill_(float('nan'))

    @property
    def dtype(self):
        return self.dev_tensors[0].dtype

    @property
    def hdim(self):
        return self.dev_tensors[0].shape[-1]

    @property
    def is_hdim_NPOT_optimized(self):
        def is_power_of_two(n: int) -> bool:
            return (n & (n - 1) == 0) and n != 0
        hdim = self.hdim
        return not is_power_of_two(hdim)

    @property
    def seqlen_q(self):
        q, k, v, b = self.dev_tensors
        seqlen_q = q.shape[2]
        return seqlen_q

    @property
    def seqlen_k(self):
        q, k, v, b = self.dev_tensors
        seqlen_k = k.shape[2]
        return seqlen_k

    @property
    def ref_device(self):
        return self.ref_tensors[0].device

    @staticmethod
    def clone_tensor(t, dtype, device=None):
        if t is None:
            return None
        return t.clone().detach().to(dtype=dtype, device=device).requires_grad_(t.requires_grad)

    @staticmethod
    def clone_tensor_tuple(in_tensors, dtype, device=None):
        return tuple([SdpaContext.clone_tensor(t, dtype=dtype, device=device) for t in in_tensors])

    def create_ref_inputs(self, target_gpu_device='cuda'):
        if AOTRITON_TORCH_ONLY_USE_CPU:
            ref_device_option = 'cpu'
        else:
            ref_device_option = AOTRITON_REF_DEVICE_OPTION
        if ref_device_option == 'default':
            ref_device = target_gpu_device
            seqlen_k = self.seqlen_k
            hdim = self.hdim
            '''
            test_gqa[False-1.2-dtype0-0.5-False-579-2048-203-N_HEADS1-4]
            triggers GPU segfault when testing with -k 'test_gqa[False-1.2-'
            (Cannot be reproduced independently)
            '''
            if seqlen_k == 579:
                ref_device = 'cpu'
            '''
            Shader _ZN2at6native12_GLOBAL__N_119cunn_SoftMaxForwardILi2EdddNS1_22SoftMaxForwardEpilogueEEEvPT2_PKT0_i causes Segfault
            for Case test_op_bwd[False-0.0-dtype2-0.0-False-587-64-8-4-4], but cannot be reproduced by running this individual UT.
            Avoiding running it on GPU for now
            '''
            if self.dtype == torch.float32:
                if seqlen_k == 587 or hdim % 16 != 0:
                    ref_device = 'cpu'
        elif ref_device_option == 'cuda':
            ref_device = target_gpu_device
        elif ref_device_option == 'cpu':
            ref_device = 'cpu'
        else:
            assert False, f'Unknown ref_device_option value {ref_device_option}. Allowed choices "default" "cpu" "cuda"'
        self.create_ref_inputs_with_device(ref_device)

    def create_ref_inputs_with_device(self, ref_device):
        dtype = self.dtype
        hp_dtype = torch.float64 if dtype == torch.float32 else torch.float32
        self.ref_tensors = self.clone_tensor_tuple(self.dev_tensors, dtype=hp_dtype, device=ref_device)
        self.lp_ref_tensors = self.clone_tensor_tuple(self.dev_tensors, dtype=dtype, device=ref_device)

    @staticmethod
    def _require_grads(tensors, skip_dq=False, skip_dk_dv=False, skip_db=False):
        q, k, v, b = tensors
        if not skip_dq:
            q.requires_grad_()
        if not skip_dk_dv:
            k.requires_grad_()
            v.requires_grad_()
        if not skip_db:
            assert b is not None
            b.requires_grad_()

    def set_require_grads(self, skip_dq=False, skip_dk_dv=False, skip_db=False):
        self._require_grads(self.dev_tensors, skip_dq=skip_dq, skip_dk_dv=skip_dk_dv, skip_db=skip_db)
        self._require_grads(self.ref_tensors, skip_dq=skip_dq, skip_dk_dv=skip_dk_dv, skip_db=skip_db)
        self._require_grads(self.lp_ref_tensors, skip_dq=skip_dq, skip_dk_dv=skip_dk_dv, skip_db=skip_db)

    def _compute_fudge_factors(self, p : SdpaParams):
        ref_q, ref_k, ref_v, ref_b = self.ref_tensors
        dtype = self.dtype
        seqlen_k = self.seqlen_k
        seqlen_q = self.seqlen_q

        # seqlen_k_fudge_factor = 1.0 if seqlen_k < 1024 else 2.0
        # seqlen_k_fudge_factor = seqlen_k_fudge_factor if seqlen_k < 8192 else 4.0
        # dropout_fudge_factor = 1.0 if p.dropout_p == 0.0 else 2.0
        # query_fudge_factor = 8 * dropout_fudge_factor * seqlen_k_fudge_factor # TODO: Investigate why grad_q needs larger tolerances
        # key_fudge_factor = 8 * dropout_fudge_factor
        # value_fudge_factor = 7
        # bias_fudge_factor = 12

        # Maximal value from tune_flash.py and table_tool.py --fudge_factor_tolerance 5.0
        # Note: Navi 3x is experimental and YMMV
        query_fudge_factor = 148.0  # NPOT
        key_fudge_factor = 48.0
        # value_fudge_factor = 16.0
        value_fudge_factor = 36.0
        bias_fudge_factor = 17.0
        # print(f'{torch.cuda.get_device_properties(0).gcnArchName=}')
        if torch.version.hip:
            if 'gfx90a' in torch.cuda.get_device_properties(0).gcnArchName:
                query_fudge_factor = max(query_fudge_factor, 130.0 if isinstance(self, VarlenSdpaContext) else 80.0)
                key_fudge_factor = 500.0 if self.is_hdim_NPOT_optimized else 340.0
                bias_fudge_factor = 45.0 if self.is_hdim_NPOT_optimized else 36.0
        if AOTRITON_TORCH_ONLY_USE_CPU:
            query_fudge_factor = 128.0
            key_fudge_factor = 330.0
            bias_fudge_factor = 36.0
            # value_fudge_factor = 36.0
        if dtype == torch.float32:
            key_fudge_factor = 180.0
            value_fudge_factor = 50.0
            bias_fudge_factor = 24.0
        return (query_fudge_factor, key_fudge_factor, value_fudge_factor, bias_fudge_factor)

    @staticmethod
    def _compute_ref_forward(ref_tensors, p : SdpaParams):
        ref_q, ref_k, ref_v, ref_b = ref_tensors
        num_head_q = ref_q.shape[1]
        num_head_k = ref_k.shape[1]
        num_head_v = ref_v.shape[1]
        assert num_head_k == num_head_v
        assert num_head_q % num_head_k == 0
        enable_gqa = num_head_q != num_head_k
        dropout_mask = p.dropout_mask if p.dropout_mask is None else p.dropout_mask.to(device=ref_q.device)
        # _scaled_dot_product_attention_math seems also working for nested tensor
        ref_out, ref_mask = sdpa_math(ref_q, ref_k, ref_v,
                                      dropout_p=p.dropout_p,
                                      is_causal=p.causal,
                                      attn_mask=ref_b,
                                      scale=p.sm_scale,
                                      dropout_mask=dropout_mask,
                                      enable_gqa=enable_gqa)
        return (ref_out, ref_mask)

    def compute_ref_forward(self, p : SdpaParams):
        self.fudge_factors = self._compute_fudge_factors(p)
        self.refout_tensors = self._compute_ref_forward(self.ref_tensors, p)
        self.lp_refout_tensors = self._compute_ref_forward(self.lp_ref_tensors, p)
        return self.lp_refout_tensors

    @staticmethod
    def _compute_backward(in_tensors, out, dout):
        q, k, v, b = in_tensors
        out.backward(dout.to(device=out.device, dtype=out.dtype))
        dq, q.grad = None if not q.requires_grad else q.grad.clone(), None
        dk, k.grad = None if not k.requires_grad else k.grad.clone(), None
        dv, v.grad = None if not v.requires_grad else v.grad.clone(), None
        if b is None or not b.requires_grad:
            db = None
        else:
            db, b.grad = b.grad.clone(), None
        return (dq, dk, dv, db)

    # Note: this follows pytorch's testing approach and expects low precision dout
    def compute_backward(self, out, _in_dout, *, ref_only=False):
        dout = _in_dout if _in_dout is not None else self.ddev_tensors[0]
        self.dref_tensors = self._compute_backward(self.ref_tensors, self.refout_tensors[0], dout)
        self.lp_dref_tensors = self._compute_backward(self.lp_ref_tensors, self.lp_refout_tensors[0], dout)
        if not ref_only:
            self.dout_tensors = self._compute_backward(self.dev_tensors, out, dout)

    @staticmethod
    def _validate(out, ref, lp_ref, fudge_factor, tname,
                  *,
                  return_target_fudge_factors=False):
        if out is None and ref is None:
            return True, 0.0, 1.0
        # atol, rtol, raw_atol, raw_rtol = get_tolerances(ref, lp_ref, fudge_factor)
        assert out is not None, f'd{tname} is none'
        assert ref is not None, f'd{tname}_ref is none'
        # print(f'{out=}')
        # print(f'{ref=}')
        def lmax(x) -> float:
            return x.abs().max().item()
        max_adiff = test_error = lmax(ref - out.to(device=ref.device))
        ref_error = lmax(ref - lp_ref)
        if math.isnan(test_error) and not math.isnan(ref_error):
            # TODO: More detailed feedback
            reason = f"Tensor {tname} has NaN output but not NaN reference"
            # print(f'{max_adiff=} {test_error=} {tname=}')
            return False, max_adiff, None
        atol = default_atol[torch.float32]
        threshold = max(atol, ref_error * fudge_factor)
        valid = test_error <= threshold
        # tft = test_error / ref_error if ref_error * fudge_factor > atol else 1.0
        tft = test_error / ref_error if not valid else 1.0
        if not valid:
            pass
            # print(f'For {tname}, Consider bump fudge_factor to {tft} = {test_error=} / {ref_error=}. So that {test_error=} < {threshold=} = max({atol=}, {ref_error=} * {tft=})')
        if return_target_fudge_factors:
            return valid, max_adiff, tft
        else:
            return valid, max_adiff, None

    def validate_with_reference(self, out, grads,
                                *,
                                no_forward=False,
                                no_backward=False,
                                return_target_fudge_factors=False):
        if no_forward:
            out_allclose, out_adiff, tft = True, None, None
        else:
            out_allclose, out_adiff, tft = self._validate(out,
                                                          self.refout_tensors[0],
                                                          self.lp_refout_tensors[0],
                                                          self.OUT_FUDGE_FACTOR,
                                                          'out',
                                                          return_target_fudge_factors=return_target_fudge_factors)
        target_fudge_factors = {'out' : tft}
        if no_backward:
            if return_target_fudge_factors:
                return out_allclose, out_adiff, [], [], target_fudge_factors
            else:
                return out_allclose, out_adiff, [], []
        grads_allclose = []
        grads_adiff = []
        # print(f'using {self.fudge_factors=}')
        for grad, ref, lp_ref, fudge_factor, tname in zip(grads, self.dref_tensors, self.lp_dref_tensors, self.fudge_factors, self.TENSOR_NAMES):
            allclose, adiff, tft = self._validate(grad,
                                                  ref,
                                                  lp_ref,
                                                  fudge_factor,
                                                  tname,
                                                  return_target_fudge_factors=return_target_fudge_factors)
            grads_allclose.append(allclose)
            grads_adiff.append(adiff)
            # if math.isnan(adiff):
            #     print(f'{adiff=} {grads_adiff=} {tname=}')
            target_fudge_factors[tname] = tft
        if return_target_fudge_factors:
            return out_allclose, out_adiff, grads_allclose, grads_adiff, target_fudge_factors
        else:
            return out_allclose, out_adiff, grads_allclose, grads_adiff

    def display_validation_results(self, tri_out, is_allclose, adiff, grads_allclose, grads_adiff):
        q, k, v, b = self.dev_tensors
        def TO(ref_tensor):
            return ref_tensor.to(device=q.device, dtype=dtype)
        dtype = q.dtype
        SPARSE_HEAD_SINCE = 1
        SPARSE_SEQ_SINCE = 1
        ref_out = self.lp_refout_tensors[0]
        if not is_allclose:
            err_idx = np.unravel_index(torch.argmax(torch.abs(TO(ref_out) - tri_out)).cpu().numpy(), ref_out.shape)
            print(f'{err_idx=}')
            print(f'{tri_out[err_idx]=}')
            print(f'{ref_out[err_idx]=}')
            print(f'{tri_out[0, 0, :4, :16]=}')
            print(f'{ref_out[0, 0, :4, :16]=}')
        dq_allclose, dk_allclose, dv_allclose, db_allclose = grads_allclose
        tri_dq, tri_dk, tri_dv, tri_db = self.dout_tensors
        ref_dq, ref_dk, ref_dv, ref_db = self.dref_tensors
        if not dv_allclose:
            err_idx = np.unravel_index(torch.argmax(torch.abs(TO(ref_dv) - tri_dv)).cpu().numpy(), ref_dv.shape)
            print(f'{q.shape=} {q.stride()=} {q.dtype=}')
            print(f'{k.shape=} {k.stride()=} {k.dtype=}')
            print(f'{v.shape=} {v.stride()=} {v.dtype=}')
            # print(f'{q[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
            # print(f'{k[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
            # print(f'{v[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
            # print(f'{dropout_mask[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
            # print(f'{dropout_mask.shape=}')
            print(f'{err_idx=}')
            print(f'{tri_dv[err_idx]=}')
            print(f'{ref_dv[err_idx]=}')
            print(f'{torch.isnan(ref_dv).any()=}')
            '''
            any_nan = torch.isnan(ref_dv).any()
            if any_nan:
                torch.set_printoptions(linewidth=200)
                print(f'{q=}')
                print(f'{k=}')
                print(f'{v=}')
                print(f'{dropout_p=}')
                print(f'{causal=}')
                print(f'{sm_scale=}')
            if seqlen_q <= 16:
                # torch.set_printoptions(linewidth=200, threshold=4096)
                print(f'{tri_dk[0,0]=}')
                print(f'{ref_dk[0,0]=}')
                print(f'{tri_dv[0,0]=}')
                print(f'{ref_dv[0,0]=}')
                # print(f'{tri_dq[0,0]=}')
                # print(f'{ref_dq[0,0]=}')
            '''

        if dv_allclose and not dk_allclose:
            print(f'{tri_out[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
            print(f'{ref_out[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
            print(f'{tri_dk[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
            print(f'{ref_dk[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
            err_idx = np.unravel_index(torch.argmax(torch.abs(TO(ref_dk) - tri_dk)).cpu().numpy(), ref_dk.shape)
            print(f'{err_idx=}')
            print(f'{tri_dk[err_idx]=} {ref_dk[err_idx]=} error = {torch.abs(tri_dk[err_idx] - ref_dk[err_idx])}')
            # print(f'{tri_dk[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]/ref_dk[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
            # print(f'{dropout_mask[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')

        if dk_allclose and dv_allclose and not dq_allclose:
            err_idx = np.unravel_index(torch.argmax(torch.abs(TO(ref_dq) - tri_dq)).cpu().numpy(), ref_dq.shape)
            print(f'{err_idx=}')
            print(f'{tri_dq[err_idx]=} {ref_dq[err_idx]=} error = {torch.abs(tri_dq[err_idx] - ref_dq[err_idx])}')

        if dk_allclose and dv_allclose and dq_allclose and not db_allclose:
            err_idx = np.unravel_index(torch.argmax(torch.abs(TO(ref_db) - tri_db)).cpu().numpy(), ref_db.shape)
            print(f'{err_idx=}')
            print(f'{tri_db[err_idx]=} {ref_db[err_idx]=} error = {torch.abs(tri_db[err_idx] - ref_db[err_idx])}')


    def save_integrity_checksum(self):
        self._tensor_checksums = {}
        self._tensor_checksums['dev'] = calc_checksums(self.dev_tensors)
        self._tensor_checksums['ref'] = calc_checksums(self.ref_tensors)
        self._tensor_checksums['lp_ref'] = calc_checksums(self.lp_ref_tensors)
        if hasattr(self, 'dref_tensors'):
            self._tensor_checksums['dref'] = calc_checksums(self.dref_tensors)
            self._tensor_checksums['lp_dref'] = calc_checksums(self.lp_dref_tensors)

    def check_integrity(self):
        for key in ['dev', 'ref', 'lp_ref', 'dref', 'lp_dref']:
            prop_key = f'{key}_tensors'
            if not hasattr(self, prop_key):
                continue
            tensors = getattr(self, prop_key)
            saved_checksums = self._tensor_checksums[key]
            current_checksums = calc_checksums(tensors)
            if saved_checksums != current_checksums:
                # print(f'who={key} {current_checksums=} != {saved_checksums=}')
                return False, key
        return True, None

    def restore_integrity(self, who, sdpa_params):
        if who == 'dev':
            todo = [ 'input', 'fwd_ref', 'bwd_ref' ]
        if who in [ 'ref', 'lp_ref' ]:
            todo = [ 'fwd_ref', 'bwd_ref' ]
        if who in [ 'dref', 'lp_dref' ]:
            todo = [ 'fwd_ref', 'bwd_ref' ]
        if 'input' in todo:
            del self.dev_tensors
            del self.ddev_tensors
            self._create_inputs()
            q, k, v, b = self.dev_tensors
            self.set_require_grads(skip_db=True if b is None else False)
        if 'fwd_ref' in todo:
            del self.ref_tensors
            del self.lp_ref_tensors
            self.create_ref_inputs(target_gpu_device=self._real_device)
            self.compute_ref_forward(sdpa_params)
        if hasattr(self, 'dref') and 'bwd_ref' in todo:
            del self.dref_tensors
            del self.lp_dref_tensors
            self.compute_backward(None, None, sdpa_params)
            # print('self.compute_backward')

class VarlenSdpaContext(SdpaContext):
    TENSOR_NAMES = ('q', 'k', 'v', 'b')

    @staticmethod
    def _rng_varlen_tensor(num_heads, seqlens, head_dim, dtype, device, packed=False):
        # Note: do NOT use nested tensor here
        #       PyTorch's UT can use nested tensor because PyTorch preprocessed
        #       the input nested tensors before sending them to its SDPA
        #       backends (See sdpa_nested_preprocessing in
        #       aten/src/ATen/native/nested/cuda/NestedTensorTransformerUtils.cpp)
        #
        #       AOTriton works in the same level as PyTorch's SDPA backends and
        #       its UT should generate the tensor in the preprocessed format directly.
        dims = (1, np.sum(seqlens), num_heads, head_dim)
        return torch.rand(*dims, dtype=dtype, device=device).transpose(1, 2)
        '''
        def _size(seqlen):
            return (seqlen, num_heads, head_dim) if not packed else (seq_len[i], 3 * num_heads * head_dim)

        return torch.nested.nested_tensor([
            torch.rand(_size(seqlen), device=device, dtype=dtype, requires_grad=True)
            for seqlen in seqlens])
        '''

    def __init__(self, N_HEADS, D_HEAD, seqlens_q, seqlens_k, dtype, device='cuda'):
        q  = self._rng_varlen_tensor(N_HEADS, seqlens_q, D_HEAD, dtype, device)
        k  = self._rng_varlen_tensor(N_HEADS, seqlens_k, D_HEAD, dtype, device)
        v  = self._rng_varlen_tensor(N_HEADS, seqlens_k, D_HEAD, dtype, device)
        b = None
        self.dev_tensors = (q, k, v, b)
        self.OUT_FUDGE_FACTOR = 3
        if dtype == torch.float32:
            self.OUT_FUDGE_FACTOR = 14.0
        if torch.version.hip:
            if 'gfx90a' in torch.cuda.get_device_properties(0).gcnArchName:
                self.OUT_FUDGE_FACTOR = 12.0
        if AOTRITON_TORCH_ONLY_USE_CPU:
            self.OUT_FUDGE_FACTOR = 12.0
        self._seqlens_q = np.array(seqlens_q)
        self._seqlens_k = np.array(seqlens_k)

    # Not perfect but fits our needs.
    @property
    def seqlen_k(self):
        return np.max(self._seqlens_q)

    @staticmethod
    def _compute_ref_forward_varlen(ref_tensors, seqlens_q, seqlens_k, p : SdpaParams):
        packed_ref_q, packed_ref_k, packed_ref_v, _ = ref_tensors
        packed_dropout_mask = p.dropout_mask if p.dropout_mask is None else p.dropout_mask.to(device=packed_ref_q.device)
        ref_out_array = []
        ref_mask_array = []
        seqlen_q_start = 0
        seqlen_k_start = 0
        for i, (seqlen_q, seqlen_k) in enumerate(zip(seqlens_q, seqlens_k)):
            ref_q = packed_ref_q[0, :, seqlen_q_start:seqlen_q_start+seqlen_q, :]
            ref_k = packed_ref_k[0, :, seqlen_k_start:seqlen_k_start+seqlen_k, :]
            ref_v = packed_ref_v[0, :, seqlen_k_start:seqlen_k_start+seqlen_k, :]
            dropout_mask = packed_dropout_mask[i, :, :, :] if packed_dropout_mask is not None else None
            print(f'REF {seqlen_q_start=} {seqlen_q_start+seqlen_q=} {ref_q.shape=} {ref_k.shape=} {ref_v.shape=}')
            print(f'REF {ref_q.stride()=}')
            if dropout_mask is not None:
                print(f'REF {packed_dropout_mask.shape=}')
                print(f'REF {dropout_mask.shape=}')
                dropout_mask = dropout_mask[:, :seqlen_q, :seqlen_k]  # Trim to actual seqlen
                print(f'REF CLAMPED {dropout_mask.shape=}')
                # print(f'REF {dropout_mask=}')
            ref_out, ref_mask = torch.ops.aten._scaled_dot_product_attention_math(ref_q, ref_k, ref_v,
                                                                        dropout_p=p.dropout_p,
                                                                        is_causal=p.causal,
                                                                        scale=p.sm_scale,
                                                                        dropout_mask=dropout_mask)
            ref_out_array.append(ref_out)
            ref_mask_array.append(ref_mask)
            seqlen_q_start += seqlen_q
            seqlen_k_start += seqlen_k
        ref_out = torch.cat(ref_out_array, dim=1).unsqueeze(dim=0)
        return ref_out, None

    def compute_ref_forward(self, p : SdpaParams):
        self.fudge_factors = self._compute_fudge_factors(p)
        self.refout_tensors = self._compute_ref_forward_varlen(self.ref_tensors, self._seqlens_q, self._seqlens_k, p)
        self.lp_refout_tensors = self._compute_ref_forward_varlen(self.lp_ref_tensors, self._seqlens_q, self._seqlens_k, p)
        return self.lp_refout_tensors

class SdpaContextFromNPZ(SdpaContext):
    def __init__(self, fn, dtype, device='cuda'):
        d = np.load(fn)
        def real_dtype():
            if d['is_fp16']:
                return torch.float16
            if d['is_bf16']:
                return torch.bfloat16
            if d['is_fp32']:
                return torch.float32
        if dtype is not None:
            assert dtype == real_dtype()
        else:
            dtype = real_dtype()
        def load(n, *, cast_to=dtype):
            return torch.tensor(d[n], dtype=cast_to, device=device)
        def load_qkv(*, prefix='', suffix='', keep_dtype=False):
            cast_to = None if keep_dtype else dtype
            # return tuple([torch.tensor(d[f'{prefix}{n}{suffix}'], dtype=cast_to, device=device) for n in 'qkvo'])
            return tuple([load(f'{prefix}{n}{suffix}', cast_to=cast_to) for n in 'qkv'])
        q, k, v = load_qkv()
        b = None
        self.dev_tensors = (q, k, v, b)
        self.OUT_FUDGE_FACTOR = 3

        # TODO: load dropout_mask

        sm_scale = float(d['scale'])
        assert not np.isnan(sm_scale), 'FIXME: suppport scale=None when capturing torch.nn.functional.scaled_dot_product_attention'
        self.sdpa_params = SdpaParams(causal=bool(d['is_causal']),
                                      sm_scale=sm_scale,
                                      dropout_p=float(d['dropout_p']),
                                      dropout_mask=None)

        self.dout = load('upstream_grad')
        self.refout_tensors = (load('o_ref'), None)
        self.lp_refout_tensors = (load('o_lp_ref'), None)

        dq, dk, dv = load_qkv(prefix='grads_', suffix='_ref', keep_dtype=True)
        self.dref_tensors = (dq, dk, dv, None)
        dq, dk, dv = load_qkv(prefix='grads_', suffix='_ref_lp', keep_dtype=True)
        self.lp_dref_tensors = (dq, dk, dv, None)

    def compute_ref_forward(self, p : SdpaParams):
        self.fudge_factors = self._compute_fudge_factors(p)
        pass

    def compute_backward(self, out, dout, *, ref_only=False):
        assert ref_only == False, 'SdpaContextFromNPZ.compute_backward is incompatible with ref_out=True'
        self.dout_tensors = self._compute_backward(self.dev_tensors, out, dout)

    def democode_save_tensors(self):
        import numpy as np
        np.savez('dump.npz',
                 is_fp16=int(query.dtype == torch.float16),
                 is_bf16=int(query.dtype == torch.bfloat16),
                 is_fp32=int(query.dtype == torch.float32),
                 q=query.float().numpy(force=True),
                 k=key.float().numpy(force=True),
                 v=value.float().numpy(force=True),
                 o=out.float().numpy(force=True),
                 o_ref=out_ref.float().numpy(force=True),
                 o_lp_ref=out_lp_ref.float().numpy(force=True),
                 upstream_grad=upstream_grad.float().numpy(force=True),
                 dropout_p=dropout_p,
                 is_causal=int(is_causal),
                 scale=float('nan') if scale is None else float(scale),
                 enable_gqa=bool(enable_gqa),
                 grads_q=grads[0].float().numpy(force=True),
                 grads_k=grads[1].float().numpy(force=True),
                 grads_v=grads[2].float().numpy(force=True),
                 grads_q_ref_lp=grads_ref_lp[0].float().numpy(force=True),
                 grads_k_ref_lp=grads_ref_lp[1].float().numpy(force=True),
                 grads_v_ref_lp=grads_ref_lp[2].float().numpy(force=True),
                 grads_q_ref=grads_ref[0].float().numpy(force=True),
                 grads_k_ref=grads_ref[1].float().numpy(force=True),
                 grads_v_ref=grads_ref[2].float().numpy(force=True),
                 )
