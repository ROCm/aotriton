# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os

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

# NOTE: FlashKernel (LUT sancheck + missing-entry diagnostic) moved to
# modules/flash/tune/sancheck.py (modular-tune.md §3b/step 11) -- the codegen
# back-edge in python/template_instantiation/ir/kdesc.py now resolves it via
# aotriton.tune.registry.load_family_tune('flash').sancheck.FlashKernel
# instead of this module. check_value/_empty_generator stay here too (small,
# pure, stable helpers duplicated rather than imported, same pattern as
# aot/flash_entry.py vs. tune/entry.py) since other aot/*.py files
# (aiter_fwd.py, aiter_bwd.py) still depend on check_value from here.
