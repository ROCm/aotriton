# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PADDED_HEAD fallback entries must preserve the meaning of impl_index."""

from pathlib import Path

import pytest

from aotriton.codegen.linker import Linker
from aotriton.template_instantiation.ir.triton.kdesc import KernelDescription


@pytest.mark.parametrize('arch', ['gfx1100', 'gfx1151', 'gfx1201'])
def test_flash_forward_padded_head_candidate_order(arch):
    kernels, _, _, _ = Linker(Path(__file__).resolve().parents[2] / 'modules').link_all_families()
    kernel = next(k for k in kernels if k.FAMILY == 'flash' and k.NAME == 'attn_fwd')
    pairs = {}
    for functional in kernel.gen_functionals({arch: [arch]}):
        if kernel.is_functional_disabled(functional):
            continue
        key = tuple((name, choice)
                    for name, choice in functional.choice.items()
                    if name != 'PADDED_HEAD')
        configs = list(kernel.gen_autotune_configs(functional))
        assert configs, f'{arch}: no candidates for {functional.choice}'
        # Config separates kernel parameters from compiler options. Compare both,
        # in order: fallback shares impl_index, not just the candidate count.
        pairs.setdefault(key, {})[functional.choices.PADDED_HEAD] = [
            (config.perf, config.copts) for config in configs]
    assert pairs, f'{arch}: no enabled forward functionals'
    for key, variants in pairs.items():
        assert set(variants) == {False, True}, f'{arch}: missing PADDED_HEAD variant: {key}'
        assert variants[False] == variants[True], f'{arch}: impl_index mismatch: {key}'


@pytest.mark.parametrize('mutation', ['missing', 'reordered', 'compiler_option', 'empty'])
def test_candidate_order_detects_corruption(monkeypatch, mutation):
    generate = KernelDescription.gen_autotune_configs

    def corrupted(self, functional):
        configs = list(generate(self, functional))
        if self.NAME == 'attn_fwd' and functional.choices.PADDED_HEAD:
            if mutation == 'missing':
                configs.pop()
            elif mutation == 'reordered':
                configs.reverse()
            elif mutation == 'compiler_option':
                configs[0].num_stages += 1
            else:
                configs.clear()
        return iter(configs)

    monkeypatch.setattr(KernelDescription, 'gen_autotune_configs', corrupted)
    with pytest.raises(AssertionError, match='impl_index mismatch|no candidates'):
        test_flash_forward_padded_head_candidate_order('gfx1100')
