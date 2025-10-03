# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from ..tdesc import TuningDescription
from ..utils import parse_python, asdict_shallow, safeload, dacite_tuple
from ..gpu_utils import do_bench
from dataclasses import dataclass, asdict
import dataclasses
from dacite import from_dict
from argparse import Namespace
from pathlib import Path
import json
import itertools
from enum import Enum

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

# Field names match mptune/flash/tuner.py and/or _core_test_backward.py
@dataclass
class FlashInputMetadata(FlashEntry):
    N_HEADS: int | tuple[int, int] = 5
    BATCH: int = 3
    sm_scale: str | float = 'l1'
    storage_flip: bool | tuple[int, int] = False
    prng_seed: int = 0x9be9_98d4_cf17_5339

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

    @property
    def device(self):
        import torch
        return f'cuda:{torch.cuda.current_device()}'

    def generate_entries(self):
        a = Namespace()
        a.dtype = ['float16', 'bfloat16', 'float32']
        a.hdim = [16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 512]
        a.seqlen_q = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        a.seqlen_k = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        a.causal = [False, True]
        a.dropout_p = [0.0, 0.5]
        a.bias_type = [0, 1]
        for tup in itertools.product(a.dtype,
                                     a.hdim,
                                     a.seqlen_q,
                                     a.seqlen_k,
                                     a.causal,
                                     a.dropout_p,
                                     a.bias_type):
            yield FlashEntry(*tup)

    def list_kernels(self, entry: FlashEntry):
        return ['attn_fwd', 'bwd_kernel_dk_dv', 'bwd_kernel_dq']

        if entry.hdim > 224:
            return ['attn_fwd', 'bwd_kernel_dk_dv', 'bwd_kernel_dq']
        return ['attn_fwd', 'bwd_kernel_dk_dv', 'bwd_kernel_dq', 'bwd_kernel_fuse']

    def _do_probe_backends(self,
                           entry: FlashEntry,
                           im: FlashInputMetadata,
                           which_kernel: str,
                           pt: Path) -> list[dict]:
        import torch
        with torch.device(self.device):
            kernel = self.get_kernel(which_kernel)
            args = kernel.create_extargs(peek_kernel_numbers=True)
            d = torch.load(pt, map_location=self.device, mmap=True)
            inputs = from_dict(data_class=kernel.PT_INPUT_CLASS, data=d["bidi_inputs"], config=dacite_tuple)
            # print(f'{type(inputs)=}')
            _ = kernel(im, inputs, args)
            total_number_of_kernels = int(args.total_number_of_kernels)
            def gen():
                for hi in range(total_number_of_kernels):
                    args.force_kernel_index = hi
                    _ = kernel(im, inputs, args)
                    d = {
                        'psels': safeload(args.selected_kernel_psels),
                        'copts': safeload(args.selected_kernel_copts),
                    }
                    yield d
            return list(gen())

    def _gen_ref(self, entry: FlashEntry, data_root: Path):
        import torch
        with torch.device(self.device):
            yield from self._do_gen_ref(entry, data_root)

    def _do_gen_ref(self, entry: FlashEntry, data_root: Path):
        im = FlashInputMetadata(**asdict(entry))
        # TODO: cut BH sizes to fit in VRAM
        yield self._write_ref(im, data_root, '00_benchmark')

        gqa = dataclasses.replace(im, N_HEADS=(10, 2))
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
        import torch
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

    def run_single_test(self,
                        im: FlashInputMetadata,
                        pt: Path,
                        which_kernel: FlashKernelSelector):
        import torch
        with torch.device(self.device):
            kernel = self.get_kernel(which_kernel.kernel_name)
            args = kernel.create_extargs(force_kernel_index=which_kernel.hsaco_index)
            d = torch.load(pt, map_location=self.device, mmap=True)
            inputs = from_dict(data_class=kernel.PT_INPUT_CLASS, data=d["bidi_inputs"], config=dacite_tuple)
            outputs = kernel(im, inputs, args)
            refs = from_dict(data_class=kernel.PT_REF_CLASS, data=d["bidi_outputs"], config=dacite_tuple)
            return kernel.compare(outputs, refs)

    def run_single_benchmark(self,
                             im: FlashInputMetadata,
                             pt: Path,
                             which_kernel: FlashKernelSelector):
        import torch
        with torch.device(self.device):
            kernel = self.get_kernel(which_kernel.kernel_name)
            device = f'cuda:{torch.cuda.current_device()}'
            d = torch.load(pt, map_location=self.device, mmap=True)
            inputs = from_dict(data_class=kernel.PT_INPUT_CLASS, data=d["bidi_inputs"], config=dacite_tuple)
            args = kernel.create_extargs(peek_kernel_numbers=True,
                                         force_kernel_index=which_kernel.hsaco_index)
            direct_inputs = kernel.prepare_directs(im, inputs)
            kernel.direct_call(direct_inputs, args)
            impl_desc = {
                'psels': safeload(args.selected_kernel_psels),
                'copts': safeload(args.selected_kernel_copts),
            }
            args.peek_kernel_numbers = False
            def fn():
                kernel.direct_call(direct_inputs, args)
            return impl_desc, do_bench(fn, quantiles=(0.5, 0.2, 0.8))

    KERNEL_DICT = None

    def get_kernel(self, kernel_name: str):
        if self.KERNEL_DICT is None:
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
                # 'bwd_kernel_fuse'   : bwd_kernel_fuse(),
            }
        return self.KERNEL_DICT.get(kernel_name)
