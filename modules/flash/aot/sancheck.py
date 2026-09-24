# Copyright © 2023-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Flash-family LUT sancheck + missing-entry diagnostic, called by the ATI kdesc
(LutSancheck.method(self=kdesc, ...) via load_family_aot) -- see
kdesc.py's sancheck_lut_tensor/_gen_missing_entries.

Loaded as aot.sancheck.LutSancheck alongside the rest of modules/flash/aot/:
no back-edge into the tuning package. `check_value`/`_empty_generator` come
from ._common, the same helpers the other aot/*.py files use.

Torch-free: safe to import outside a GPU container.
"""

from aotriton.gpu_targets import AOTRITON_ARCH_WARPSIZE
from aotriton.utils import log

from ._common import check_value, _empty_generator


# Flash's LUT axes. These live here, not on the ATI kdesc: the kdesc is the
# generic IR node and must not carry family-shaped values. The methods below
# are called unbound with `self` bound to that kdesc, so they must reference
# these directly rather than through `self`.
LUT_FULL_SEQLEN_Q = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
LUT_FULL_SEQLEN_K = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
LUT_FULL_SEQLEN_NAVI = [16, 32, 64, 128, 256, 512, 1024, 2048]
# gfx1250's tech-preview database is tuned on three seqlen entries only.
LUT_FULL_SEQLEN_TP = [64, 256, 2048]


class LutSancheck:
    """Flash-family LUT sancheck + missing-entry diagnostic, called by the ATI
    kdesc (LutSancheck.method(self=kdesc, ...) via family_aot). A plain holder — no
    description base; it relies only on the duck-typed kdesc surface (check_value,
    gen_autotune_configs presence)."""
    FAMILY = 'flash'

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
        # Mirrors _common.flash_disabled's gfx1250 rules: that one reads
        # f.choices, this one check_value, so neither can call the other.
        if functional.arch == 'gfx1250':
            hdim = check_value(functional, 'BLOCK_DMODEL')
            if hdim > 256:
                return True
            if hdim & (hdim - 1) != 0:
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
        TECH_PREVIEW = (arch == 'gfx1250')
        LUT_TENSOR_SIZE = (len(LUT_FULL_SEQLEN_Q), len(LUT_FULL_SEQLEN_K))
        LUT_TENSOR_SIZE_NAVI = (len(LUT_FULL_SEQLEN_NAVI), len(LUT_FULL_SEQLEN_NAVI))
        LUT_TENSOR_SIZE_TP = (len(LUT_FULL_SEQLEN_TP), len(LUT_FULL_SEQLEN_TP))
        log(lambda : f'{lut_tensor.shape=} ==? {LUT_TENSOR_SIZE=}')
        all_pos = (lut_tensor >= 0).all()
        shape = lut_tensor.shape[1:]
        # Tested before Navi: gfx1250 is wave32, and the Navi branch alone would
        # reject its three-entry tech-preview table.
        if TECH_PREVIEW:
            shape_match = (shape == LUT_TENSOR_SIZE or shape == LUT_TENSOR_SIZE_NAVI
                           or shape == LUT_TENSOR_SIZE_TP)
        elif MI:
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
            if TECH_PREVIEW:
                errors.append(f"Unexpected {shape=}, Expecting {LUT_TENSOR_SIZE}, "
                              f"{LUT_TENSOR_SIZE_NAVI} or {LUT_TENSOR_SIZE_TP}")
            elif Navi:
                errors.append(f"Unexpected {shape=}, Expecting {LUT_TENSOR_SIZE} or {LUT_TENSOR_SIZE_NAVI}")
            else:
                errors.append(f"Unexpected {shape=}, Expecting {LUT_TENSOR_SIZE}")
        # Pick the seqlen lists that match the actual lut_tensor shape for this arch.
        if TECH_PREVIEW and lut_tensor.shape[1:] == LUT_TENSOR_SIZE_TP:
            lut_full_seqlen_q = LUT_FULL_SEQLEN_TP
            lut_full_seqlen_k = LUT_FULL_SEQLEN_TP
            expected_size = LUT_TENSOR_SIZE_TP
        elif Navi and lut_tensor.shape[1:] == LUT_TENSOR_SIZE_NAVI:
            lut_full_seqlen_q = LUT_FULL_SEQLEN_NAVI
            lut_full_seqlen_k = LUT_FULL_SEQLEN_NAVI
            expected_size = LUT_TENSOR_SIZE_NAVI
        else:
            lut_full_seqlen_q = LUT_FULL_SEQLEN_Q
            lut_full_seqlen_k = LUT_FULL_SEQLEN_K
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
