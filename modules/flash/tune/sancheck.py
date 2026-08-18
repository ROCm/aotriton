# Copyright © 2023-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Flash-family LUT sancheck + missing-entry diagnostic, called by the ATI kdesc
(LutSancheck.method(self=kdesc, ...) via aotriton.tune.registry.load_family_tune) --
see python/template_instantiation/ir/kdesc.py's sancheck_lut_tensor/_gen_missing_entries.

Moved out of modules/flash/aot/_common.py (modular-tune.md §3b/step 11) so the
codegen back-edge into modules/flash/tune/ goes through the tune-side registry
instead of aot-side internals. `check_value`/`_empty_generator` are small,
pure, stable helpers duplicated (not imported) from _common.py -- _common.py
must keep its own copies for the other modules/flash/aot/*.py files that use
them (attn_fwd.py, bwd_kernel_dk_dv.py, bwd_kernel_dq.py, bwd_kernel_fuse.py,
bwd_preprocess.py, bwd_preprocess_varlen.py, aiter_fwd.py, aiter_bwd.py), the
same pattern already used for FlashEntry (aot/flash_entry.py vs. entry.py).

Torch-free: safe to import outside a GPU container.
"""

from aotriton.gpu_targets import AOTRITON_ARCH_WARPSIZE
from aotriton.utils import log


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


# Flash's LUT axes. These live here, not on the ATI kdesc: the kdesc is the
# generic IR node and must not carry family-shaped values. The methods below
# are called unbound with `self` bound to that kdesc, so they must reference
# these directly rather than through `self`.
LUT_FULL_SEQLEN_Q = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
LUT_FULL_SEQLEN_K = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
LUT_FULL_SEQLEN_NAVI = [16, 32, 64, 128, 256, 512, 1024, 2048]


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
        LUT_TENSOR_SIZE = (len(LUT_FULL_SEQLEN_Q), len(LUT_FULL_SEQLEN_K))
        LUT_TENSOR_SIZE_NAVI = (len(LUT_FULL_SEQLEN_NAVI), len(LUT_FULL_SEQLEN_NAVI))
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
        from .entry import FlashEntry
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
