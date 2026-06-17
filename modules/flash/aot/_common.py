# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os

from aotriton.gpu_targets import AOTRITON_ARCH_WARPSIZE
from aotriton.utils import log


# Shared across the flash kernel descriptions (attn_fwd / bwd_kernel_* / bwd_preprocess*).
MAIN_DTYPES = ['*fp16:16', '*bf16:16', '*fp32:16']


def block_dmodel_values():
    """The BLOCK_DMODEL axis values, overridable via AOTRITON_FLASH_BLOCK_DMODEL."""
    env = os.getenv('AOTRITON_FLASH_BLOCK_DMODEL',
                    default='16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 512')
    return [int(d) for d in env.split(',')]


def flash_disabled(f, *, gfx950_bad_hdims=()):
    """True if functional `f` must be excluded for compiler/numerical correctness.

    The single functional-disable predicate shared by the fwd and bwd ATI
    descriptions. `gfx950_bad_hdims` is the per-kernel set of BLOCK_DMODEL values
    the gfx950 compiler has a known numerical error on (fwd: {16}; bwd: {48, 80});
    everything else (causal+matrix-bias unsupported, gfx11 hdim>256) is common."""
    causal = f.choices.CAUSAL_TYPE
    hdim = f.choices.BLOCK_DMODEL
    bias_type = f.choices.BIAS_TYPE
    if causal != 0 and bias_type != 0:
        return True
    if f.arch.startswith('gfx11') and hdim > 256:
        return True
    if f.arch == 'gfx950' and hdim in gfx950_bad_hdims:
        return True
    return False


def _empty_generator():
    return
    yield  # makes this a generator function


def check_value(functional, repr_name):
    if not isinstance(repr_name, list):
        repr_name = [repr_name]
    tc = functional.compact_choices
    for aname in repr_name:
        if aname in tc:
            return tc[aname].triton_compile_signature
    assert False, f'Cannot find {repr_name=} in {functional=}'

class FlashKernel:
    """Flash-family LUT sancheck + missing-entry diagnostic, called by the ATI
    kdesc (FlashKernel.method(self=kdesc, ...) via family_aot). A plain holder — no
    description base; it relies only on the duck-typed kdesc surface (check_value,
    gen_autotune_configs presence)."""
    FAMILY = 'flash'
    LUT_FULL_SEQLEN_Q = [16,32,64,128,256,512,1024,2048,4096,8192]
    LUT_FULL_SEQLEN_K = [16,32,64,128,256,512,1024,2048,4096,8192]
    LUT_FULL_SEQLEN_NAVI = [16,32,64,128,256,512,1024,2048]

    def is_functional_disabled(self, functional):
        if not hasattr(self, 'gen_autotune_configs'):  # only check acutal FA kernels
            return False
        is_causal = check_value(functional, ['CAUSAL', 'CAUSAL_TYPE'])
        bias_type = check_value(functional, 'BIAS_TYPE')
        # print(f'Functional {functional.godel_number=} {is_causal=} {bias_type=}')
        if is_causal and bias_type != 0:
            return True
        if functional.arch.startswith('gfx11'):
            hdim = check_value(functional, 'BLOCK_DMODEL')
            if hdim > 256:
                return True
        return False

    def sancheck_lut_tensor(self,
                            functional : 'Functional',
                            lut_tensor):
        # Only kernels that provide gen_autotune_configs may have entries in
        # tuning database
        if not hasattr(self, 'gen_autotune_configs'):
            return True, [], _empty_generator()
        arch = functional.arch
        if self.is_functional_disabled(functional):
            return True, [], _empty_generator()
        MI = (AOTRITON_ARCH_WARPSIZE[arch] == 64)
        Navi = (AOTRITON_ARCH_WARPSIZE[arch] == 32)
        LUT_TENSOR_SIZE = (len(self.LUT_FULL_SEQLEN_Q), len(self.LUT_FULL_SEQLEN_K))
        LUT_TENSOR_SIZE_NAVI = (len(self.LUT_FULL_SEQLEN_NAVI), len(self.LUT_FULL_SEQLEN_NAVI))
        log(lambda : f'{lut_tensor.shape=} ==? {LUT_TENSOR_SIZE=}')
        all_pos = (lut_tensor >= 0).all()
        shape = lut_tensor.shape[1:]
        if MI:
            shape_match = shape == LUT_TENSOR_SIZE
        elif Navi:
            shape_match = (shape == LUT_TENSOR_SIZE or shape == LUT_TENSOR_SIZE_NAVI)
        else:
            assert False, f"Unknown {arch}"
        ok = all_pos and shape_match
        if ok:
            return ok, [], _empty_generator()
        errors = []
        if not all_pos:
            errors.append("certain entries are empty (-1)")
        if not shape_match:
            if Navi:
                errors.append(f"Unexpected {shape=}, Expecting {LUT_TENSOR_SIZE} or {LUT_TENSOR_SIZE_NAVI}")
            else:
                errors.append(f"Unexpected {shape=}, Expecting {LUT_TENSOR_SIZE}")
        # Pick the seqlen lists that match the actual lut_tensor shape for this arch.
        if Navi and lut_tensor.shape[1:] == LUT_TENSOR_SIZE_NAVI:
            lut_full_seqlen_q = self.LUT_FULL_SEQLEN_NAVI
            lut_full_seqlen_k = self.LUT_FULL_SEQLEN_NAVI
            expected_size = LUT_TENSOR_SIZE_NAVI
        else:
            lut_full_seqlen_q = self.LUT_FULL_SEQLEN_Q
            lut_full_seqlen_k = self.LUT_FULL_SEQLEN_K
            expected_size = LUT_TENSOR_SIZE
        missing_entries = self._gen_missing_entries(functional, lut_tensor,
                                                    arch, lut_full_seqlen_q,
                                                    lut_full_seqlen_k, expected_size)
        return ok, errors, missing_entries

    def _gen_missing_entries(self, functional, lut_tensor,
                             arch, lut_full_seqlen_q, lut_full_seqlen_k, expected_size):
        import numpy as np
        from .flash_entry import FlashEntry
        causal_raw = check_value(functional, 'CAUSAL_TYPE')
        hdim = check_value(functional, 'BLOCK_DMODEL')
        dropout_p = 0.5 if check_value(functional, 'ENABLE_DROPOUT') else 0.0
        q_ptr = check_value(functional, 'Q')
        if q_ptr.startswith('*fp16'):
            dtype = 'float16'
        elif q_ptr.startswith('*bf16'):
            dtype = 'bfloat16'
        else:
            dtype = 'float32'
        bias_type = check_value(functional, 'BIAS_TYPE')
        causal = bool(causal_raw)  # 0 → False, non-zero → True
        def make_entry(seqlen_q, seqlen_k) -> str:
            entry = FlashEntry(
                dtype=dtype,
                hdim=hdim,
                seqlen_q=seqlen_q,
                seqlen_k=seqlen_k,
                causal=causal,
                dropout_p=dropout_p,
                bias_type=bias_type,
            )
            return f'arch={arch} {entry.as_text()}'
        if lut_tensor.size == 1:
            for seqlen_q in lut_full_seqlen_q:
                for seqlen_k in lut_full_seqlen_k:
                    yield make_entry(seqlen_q, seqlen_k)
        else:
            # TODO: support non-mod0
            if lut_tensor.shape[1:] == expected_size:
                _, M_idxs, N_idxs = np.where(lut_tensor < 0)
            else:
                fake_lut = np.full(expected_size, -1, dtype=np.int32)
                M_idxs, N_idxs = np.where(fake_lut < 0)
            for M_id, N_id in zip(M_idxs, N_idxs):
                yield make_entry(lut_full_seqlen_q[M_id], lut_full_seqlen_k[N_id])
