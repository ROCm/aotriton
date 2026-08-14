# Copyright © 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from aotriton.tune.tdesc import TuningDescription
from aotriton.tune.utils import asdict_shallow, sanitize_value
from .entry import FlashEntry, FlashInputMetadata
from dataclasses import asdict
import dataclasses
from pathlib import Path
import gc

'''
CAVEAT about imports
FlashTune (and everything it imports at module scope: entry.py, tdesc.py,
utils.py, dataclasses/pathlib/gc) is a dual purpose class, which may
not have torch/pyaotriton/dacite packages installed in the environment.
`dacite`/`from_dict` are imported lazily inside run_single_test() /
run_single_benchmark() below, alongside their own lazy `import torch` --
those are the only two methods that need dacite, and both already require
a live GPU anyway.

dispatch_tasks.py instantiates FlashTune() for EVERY registered tuning module
at CLI startup (to build argparse subparsers from get_entry_choices()), so
this file -- and its LEVELS resolution below -- must stay torch-free at
import time and at construction time regardless of which `level` is
requested. Only the level actually used (level_kernel.py / level_op.py, both
torch/pyaotriton-heavy) is imported, and only once TuningDescription.__init__
actually looks it up (see _LazyLevelMap below) or, more commonly, once
get_impl()/probe_backends()/benchmark() is actually called.
'''


class _LazyLevelMap:
    """dict[str, type[TuningLevel]]-like mapping that imports level_kernel.py
    / level_op.py lazily, one at a time, only for the key actually looked up
    -- so constructing `FlashTune(level='op')` never imports level_kernel.py
    (tuning-lib/torch-heavy) and vice versa (modular-tune.md §4.5)."""

    _NAMES = {
        'kernel': ('.level_kernel', 'FlashKernelLevel'),
        'op': ('.level_op', 'FlashOpLevel'),
    }

    def __getitem__(self, key):
        modname, clsname = self._NAMES[key]
        import importlib
        mod = importlib.import_module(modname, package=__package__)
        return getattr(mod, clsname)

    def __iter__(self):
        return iter(self._NAMES)

    def __len__(self):
        return len(self._NAMES)

    def __contains__(self, key):
        return key in self._NAMES


class FlashTune(TuningDescription):
    ENTRY_CLASS = FlashEntry
    INPUT_METADATA = FlashInputMetadata
    LEVELS = _LazyLevelMap()

    def get_entry_choices(self):
        return FlashEntry(
            dtype=['float16', 'bfloat16', 'float32'],
            hdim=[16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 512],
            seqlen_q=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
            seqlen_k=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
            causal=[False, True],
            dropout_p=[0.0, 0.5],
            bias_type=[0, 1]
        )

    def validate_entry(self, entry: FlashEntry) -> bool:
        # Skip combinations where causal=True and bias_type != 0
        if entry.causal and entry.bias_type != 0:
            return False
        return True

    def validate_hw_feature(self, arch: str, entry: FlashEntry) -> tuple[bool, str]:
        # gfx11xx (RDNA 3, 32-lane wavesize) lacks LDS/register resources for hdim > 256.
        # The code generator also disables these combinations in _common.py; reject here
        # to avoid dispatching tuning tasks that would produce no compiled kernels.
        if arch.startswith('gfx11') and entry.hdim > 256:
            return False, (f'arch {arch} does not support hdim={entry.hdim} '
                           f'(gfx11xx maximum is 256; larger hdim exceeds LDS/register limits)')
        if arch.startswith('gfx11') and (entry.seqlen_q > 2048 or entry.seqlen_k > 2048):
            return False, (f'Insufficient number of gfx1100 GPUs available for tuning arch {arch}: '
                           f'only seqlen_q/k <= 2048 entries are tuned')
        return True, ''

    def _gen_ref(self, entry: FlashEntry, data_root: Path, extra_ims: list = []):
        import torch
        from aotriton.tune.gpu_utils import device_ctx
        with device_ctx():
            yield from self._do_gen_ref(entry, data_root)
            for idx, im in enumerate(extra_ims):
                tname = f'{6 + idx:02d}_utextra'
                yield self._write_ref_no_clamp(im, data_root, tname)

    def _clamp_memory_usage(self, im: FlashInputMetadata) -> FlashInputMetadata:
        '''
        Clamp batch size and number of heads to avoid OOM.
        Based on clamp_memory_usage from test/tune_flash.py.
        '''
        from aotriton.tune.gpu_utils import get_total_memory_from_amdsmi
        import math

        vram_cap_gb = get_total_memory_from_amdsmi()
        if vram_cap_gb is None:
            # Cannot determine VRAM, return unchanged
            return im

        # Extract values
        batch = im.BATCH if isinstance(im.BATCH, int) else 3
        is_gqa = not isinstance(im.N_HEADS, int)
        n_heads = im.N_HEADS[0] if is_gqa else im.N_HEADS
        d_head = im.hdim if isinstance(im.hdim, int) else im.hdim[0]
        seqlen_q = im.seqlen_q
        seqlen_k = im.seqlen_k
        causal = im.causal
        dropout_p = im.dropout_p
        dtype = im.dtype
        bias_type = im.bias_type

        # Empirical for FWD+BWD (assuming all kernels are tuned)
        # Forward-only would use different formula, but we assume backward is enabled
        def current_cost():
            base_cost = 0.11 * batch * n_heads * d_head * seqlen_q * seqlen_k / (1024 ** 3)
            factor = 1.0
            if dropout_p > 0.0:
                factor += 0.25
            if bias_type != 0:
                factor += 0.33
            if dtype == 'float32':
                factor *= 2.0
            return 2.0 * factor * base_cost  # Mul by 2 to ensure only use 50% of VRAM
        if current_cost() > vram_cap_gb:
            n_heads = min(n_heads, 24)
        if current_cost() > vram_cap_gb:
            n_heads = min(n_heads, 12)
        if current_cost() > vram_cap_gb:
            n_heads = min(n_heads, 6)
        if current_cost() > vram_cap_gb:
            n_heads = min(n_heads, 3)
        if current_cost() > vram_cap_gb:
            n_heads = min(n_heads, 2)
        if current_cost() > vram_cap_gb:
            batch = min(batch, 2)
        if is_gqa:
            if n_heads >= 24:
                n_heads = (24, 8)
            elif n_heads >= 12:
                n_heads = (12, 4)
            elif n_heads >= 6:
                n_heads = (6, 2)
            elif n_heads >= 3:
                n_heads = (3, 1)
            elif n_heads >= 2:
                n_heads = (2, 1)

        # Update im if values changed
        if batch != im.BATCH or n_heads != im.N_HEADS:
            import torch
            gc.collect()
            torch.cuda.empty_cache()
            return dataclasses.replace(im, BATCH=batch, N_HEADS=n_heads)
        return im

    def _do_gen_ref(self, entry: FlashEntry, data_root: Path):
        '''
        Pre-condition: called with device_ctx()
        '''
        im = FlashInputMetadata(**asdict(entry))
        im = self._clamp_memory_usage(im)
        yield self._write_ref(im, data_root, '00_benchmark')

        gqa = dataclasses.replace(im, N_HEADS=(10, 2))
        gqa = self._clamp_memory_usage(gqa)
        yield self._write_ref(gqa, data_root, '01_gqa')

        ihdim = dataclasses.replace(im, hdim=im.hdim - 8)
        yield self._write_ref(ihdim, data_root, '02_irregular_hdim')

        irregular_seqlen = dataclasses.replace(im,
                                               seqlen_q=im.seqlen_q - 7,
                                               seqlen_k=im.seqlen_k - 7)
        yield self._write_ref(irregular_seqlen, data_root, '03_irregular_seqlen')

        irregular_both = dataclasses.replace(ihdim,
                                             seqlen_q=ihdim.seqlen_q - 7,
                                             seqlen_k=ihdim.seqlen_k - 7)
        yield self._write_ref(irregular_both, data_root, '04_irregular_both')

        bshd = dataclasses.replace(irregular_seqlen, storage_flip=(1,2))
        yield self._write_ref(bshd, data_root, '05_bshd')
        # TODO: varlen tests

    def _write_ref(self,
                   im: FlashInputMetadata,
                   root: Path,
                   tname: str):
        '''
        Pre-condition: called with device_ctx()
        '''
        import torch
        if im.qkh > 2048 * 2048 * 64:
            gc.collect()
            torch.cuda.empty_cache()
        # print(f'{tname=} {im=}')
        from .reference import SdpaReference
        ref_kernel = SdpaReference()
        bidi_inputs = ref_kernel.generate_inputs(im)
        bidi_inputs, outputs = ref_kernel(im, bidi_inputs, None)
        d = {
            "bidi_inputs" : asdict_shallow(bidi_inputs),
            "bidi_outputs" : asdict_shallow(outputs),
        }
        pt = (root / tname).with_suffix('.pt')
        torch.save(d, pt)
        return tname, im, pt

    def _write_ref_no_clamp(self,
                            im: FlashInputMetadata,
                            root: Path,
                            tname: str):
        '''Like _write_ref but skips _clamp_memory_usage — extra IMs come from real
        pytest runs so their shapes are known to fit in VRAM.
        Pre-condition: called with device_ctx()
        '''
        import torch
        if im.qkh > 2048 * 2048 * 64:
            gc.collect()
            torch.cuda.empty_cache()
        from .reference import SdpaReference
        ref_kernel = SdpaReference()
        bidi_inputs = ref_kernel.generate_inputs(im)
        bidi_inputs, outputs = ref_kernel(im, bidi_inputs, None)
        d = {
            "bidi_inputs" : asdict_shallow(bidi_inputs),
            "bidi_outputs" : asdict_shallow(outputs),
        }
        pt = (root / tname).with_suffix('.pt')
        torch.save(d, pt)
        return tname, im, pt

    def run_single_test(self,
                        im: FlashInputMetadata,
                        pt: Path,
                        which_impl):
        import torch
        from dacite import from_dict
        from aotriton.tune.utils import dacite_tuple
        from aotriton.tune.gpu_utils import device_ctx, default_device_string
        with device_ctx():
            kernel = self.get_impl(which_impl)
            args = kernel.create_extargs(which_impl=which_impl)
            d = torch.load(pt, map_location=default_device_string(), mmap=True)
            inputs = from_dict(data_class=kernel.PT_INPUT_CLASS, data=d["bidi_inputs"], config=dacite_tuple)
            direct_inputs = kernel.prepare_directs(im, inputs)
            kernel.fill_nan_to_outputs(direct_inputs)
            outputs, err = kernel.direct_call(direct_inputs, args)
            refs = from_dict(data_class=kernel.PT_REF_CLASS, data=d["bidi_outputs"], config=dacite_tuple)
            result = kernel.compare(outputs, refs)
            early = kernel.check_early_reject_results(result, err)
            if early is not None:
                result = early
            if im.qkh > 2048 * 2048 * 64:
                gc.collect()
                torch.cuda.empty_cache()
            return sanitize_value(result)

    def run_single_benchmark(self,
                             im: FlashInputMetadata,
                             pt: Path,
                             which_impl):
        import torch
        from dacite import from_dict
        from aotriton.tune.utils import dacite_tuple
        from aotriton.tune.gpu_utils import do_bench, device_ctx, default_device_string
        with device_ctx():
            kernel = self.get_impl(which_impl)
            args = kernel.create_extargs(which_impl=which_impl, probe=True)
            d = torch.load(pt, map_location=default_device_string(), mmap=True)
            inputs = from_dict(data_class=kernel.PT_INPUT_CLASS, data=d["bidi_inputs"], config=dacite_tuple)
            direct_inputs = kernel.prepare_directs(im, inputs)
            kernel.direct_call(direct_inputs, args)
            impl_desc = self.probe_impl_desc(kernel, args)
            args.disable_probing()
            def fn():
                kernel.direct_call(direct_inputs, args)
            times = do_bench(fn, quantiles=(0.5, 0.2, 0.8))
            if im.qkh > 2048 * 2048 * 64:
                gc.collect()
                torch.cuda.empty_cache()
            return sanitize_value(impl_desc), sanitize_value(times)
