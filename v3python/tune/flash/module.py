# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from ..tdesc import TuningDescription
from ..utils import parse_python, asdict_shallow, safeload, dacite_tuple, sanitize_value
from dataclasses import dataclass, asdict
import dataclasses
from dacite import from_dict
from argparse import Namespace
from pathlib import Path
import json
import itertools
from enum import Enum
import gc

'''
CAVEAT about imports
TestingDescription is a dual purpose class, which may not have torch/pyaotriton
package installed in the environment

Any GPU related imports must be deferred to the related function instead import
at the beginning of the file.
'''

@dataclass
class FlashEntry:
    dtype: str = 'float16'
    hdim: int | tuple[int, int] = 16  # tuple[int, int] for hdim_qk != hdim_v
    seqlen_q: int = 16
    seqlen_k: int = 16
    causal: bool | tuple[int, int] = 0
    dropout_p: float = 0.0
    bias_type: int = 0

    @staticmethod
    def parse_text(line: str) -> "FlashEntry":
        d = parse_python(line)
        return FlashEntry(**d)

    @staticmethod
    def from_dict(d: dict) -> "FlashEntry":
        return from_dict(data_class=FlashEntry, data=d, config=dacite_tuple)

    def as_posix(self) -> str:
        return ','.join([f"{k}={v}" for k, v in asdict(self).items()])

    def as_text(self) -> str:
        def tr(v) -> str:
            if isinstance(v, str):
                return f"'{v}'"
            if isinstance(v, tuple):
                return '(' + ','.join(tr(x) for x in v) + ')'
            if isinstance(v, list):
                return '[' + ','.join(tr(x) for x in v) + ']'
            return str(v)
        return ';'.join([f"{k}={tr(v)}" for k, v in asdict(self).items()])

    @property
    def qkh(self):
        return self.seqlen_q * self.seqlen_k * self.hdim

# Field names match mptune/flash/tuner.py and/or _core_test_backward.py
@dataclass
class FlashInputMetadata(FlashEntry):
    N_HEADS: int | tuple[int, int] = 5
    BATCH: int = 3
    sm_scale: str | float = 'l1'
    storage_flip: bool | tuple[int, int] = False
    prng_seed: int = 0x9be9_98d4_cf17_5339

    @staticmethod
    def parse_text(line: str) -> "FlashEntry":
        d = parse_python(line)
        return FlashInputMetadata(**d)

    @staticmethod
    def from_dict(d: dict) -> "FlashInputMetadata":
        return from_dict(data_class=FlashInputMetadata, data=d, config=dacite_tuple)

@dataclass
class FlashKernelSelector:
    kernel_name: str = ''
    hsaco_index: int = -1
    max_hsaco: int = -1

    @staticmethod
    def parse_text(line: str) -> "FlashKernelSelector":
        kernel_name, hsaco_index = line.split("=")
        return FlashKernelSelector(kernel_name=kernel_name, hsaco_index=int(hsaco_index))

class Flash(TuningDescription):
    ENTRY_CLASS = FlashEntry
    INPUT_METADATA = FlashInputMetadata

    # TODO: Make it configurable?
    def __init__(self):
        pass

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
        if arch == 'gfx1250' and entry.hdim > 256:
            return False, (f'arch {arch} does not support hdim={entry.hdim} '
                           f'(no shipped candidate is numerically accurate at hdim > 256)')
        return True, ''

    def list_impls(self, entry: FlashEntry, arch: str | None = None):
        if False:  # Debugging, fwd only tuning. Keep it for selective tuning
            return ['attn_fwd']
        if entry.hdim > 224:
            return ['attn_fwd', 'bwd_kernel_dk_dv', 'bwd_kernel_dq']
        if arch == 'gfx1250':  # No bwd_kernel_fuse for gfx1250
            return ['attn_fwd', 'bwd_kernel_dk_dv', 'bwd_kernel_dq']
        return ['attn_fwd', 'bwd_kernel_dk_dv', 'bwd_kernel_dq', 'bwd_kernel_fuse']

    def _do_probe_backends(self,
                           entry: FlashEntry,
                           im: FlashInputMetadata,
                           which_impl: str,
                           pt: Path) -> list[dict]:
        import torch
        from ..gpu_utils import device_ctx, default_device_string
        with device_ctx():
            kernel = self.get_impl(which_impl)
            args = kernel.create_extargs(probe=True)
            d = torch.load(pt, map_location=default_device_string(), mmap=True)
            inputs = from_dict(data_class=kernel.PT_INPUT_CLASS, data=d["bidi_inputs"], config=dacite_tuple)
            # print(f'{type(inputs)=}')
            _ = kernel(im, inputs, args)
            total_number_of_kernels = int(args.selected_kernel_total_hsacos)
            def gen():
                for hi in range(total_number_of_kernels):
                    args.set_hsaco(hsaco=hi, probe=True)
                    _ = kernel(im, inputs, args)
                    d = {
                        'psels': safeload(args.selected_hsaco_psels),
                        'copts': safeload(args.selected_hsaco_copts),
                    }
                    yield d
            return list(gen())

    def _gen_ref(self, entry: FlashEntry, data_root: Path, extra_ims: list = []):
        import torch
        from ..gpu_utils import device_ctx
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
        from ..gpu_utils import get_total_memory_from_amdsmi
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

        # # Old empirical algorithm that (mostly) works with bwd
        # new_heads = im.N_HEADS
        # if seqlen_q * seqlen_k * d_head >= 2048 * 2048 * vram_cap_gb:
        #     batch = min(batch, 3)
        #     new_heads = (4, 1) if is_gqa else min(n_heads, 4)
        # if (causal or bias_type != 0) and seqlen_q * seqlen_k * d_head >= 2048 * 2048 * vram_cap_gb:
        #     # Prevent OOM, causal=True needs more memory
        #     batch = min(batch, 2)
        #     new_heads = (2, 1) if is_gqa else min(n_heads, 2)

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
                        which_impl: FlashKernelSelector):
        import torch
        from ..gpu_utils import device_ctx, default_device_string
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

    def probe_impl_desc(self, kernel, args) -> dict:
        return {
            'psels': safeload(args.selected_hsaco_psels),
            'copts': safeload(args.selected_hsaco_copts),
        }

    def run_single_benchmark(self,
                             im: FlashInputMetadata,
                             pt: Path,
                             which_impl: FlashKernelSelector):
        import torch
        from ..gpu_utils import do_bench, device_ctx, default_device_string
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

    KERNEL_DICT = None

    def get_impl(self, name: str | FlashKernelSelector):
        if isinstance(name, FlashKernelSelector):
            name = name.kernel_name
        if self.KERNEL_DICT is None:
            if False:  # Debugging, fwd only tuning. Keep it for selective tuning
                from .kernels import (
                    attn_fwd,
                )
                self.KERNEL_DICT = {
                    'attn_fwd'          : attn_fwd(),
                }
            else:
                from .kernels import (
                    attn_fwd,
                    bwd_kernel_dk_dv,
                    bwd_kernel_dq,
                    bwd_kernel_fuse,
                )
                self.KERNEL_DICT = {
                    'attn_fwd'          : attn_fwd(),
                    'bwd_kernel_dk_dv'  : bwd_kernel_dk_dv(),
                    'bwd_kernel_dq'     : bwd_kernel_dq(),
                    'bwd_kernel_fuse'   : bwd_kernel_fuse(),
                }
        return self.KERNEL_DICT[name]
