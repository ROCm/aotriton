# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from ..tkdesc import TestingDescription
from dataclasses import dataclass, asdict
import dataclasses
from dacite import from_dict
from typing import Generator
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

def safeload(s):
    return json.loads(s) if s else None

# Field names match mptune/flash/tuner.py and/or _core_test_backward.py
@dataclass
class FlasnConfig:
    BATCH: int = 3
    N_HEADS: int | tuple[int, int] = 5
    D_HEAD: int | tuple[int, int] = 16  # tuple[int, int] for hdim_qk != hdim_v
    seqlen_q: int = 16
    seqlen_k: int = 16
    causal: int | tuple[int, int] = 0
    sm_scale: str | float = 'l1'
    dropout_p: float = 0.0
    dtype: str = 'float16'
    storage_flip: bool | tuple[int, int] = False
    bias_type: int = 0
    prng_seed: int = 0x9be9_98d4_cf17_5339

class Flash(TestingDescription):
    # TODO: Make it configurable?
    def __init__(self):
        a = Namespace()
        a.BATCH = [3]
        a.N_HEADS = [5]
        a.D_HEAD = [16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 512]
        a.seqlen_q = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        a.seqlen_k = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        a.causal = [0, 1]
        a.sm_scale = ['l1']
        a.dropout_p = [0.0, 0.5]
        a.dtype = ['float16', 'bfloat16', 'float32']
        a.storage_flip = [False]
        a.bias_type = [0, 1]
        self._args = a

    def gen_entry_config(self) -> Generator[FlasnConfig]:
        a = self._args
        for tup in itertools.product(a.BATCH,
                                     a.N_HEADS,
                                     a.D_HEAD,
                                     a.seqlen_q,
                                     a.seqlen_k,
                                     a.causal,
                                     a.sm_scale,
                                     a.dropout_p,
                                     a.dtype,
                                     a.storage_flip,
                                     a.bias_type):
            yield FlasnConfig(*tup)

    def list_kernels(self, entry_config: FlasnConfig):
        if entry_config.D_HEAD > 224:
            return ['attn_fwd', 'bwd_kernel_dk_dv', 'bwd_kernel_dq']
        return ['attn_fwd', 'bwd_kernel_dk_dv', 'bwd_kernel_dq', 'bwd_kernel_fuse']

    def probe_backends(self,
                       confg: 'Config',
                       kernel_name: str) -> list[dict]:
        kernel = self.get_kernel(kernel_name)
        inputs = kernel.generate_inputs(config, dry_run=True)
        args = kernel.create_extargs(peek_kernel_numbers=True)
        _ = kernel(inputs, args)
        total_number_of_kernels = int(args.total_number_of_kernels)
        def gen():
            for hi in range(total_number_of_kernels):
                args.force_kernel_index = hi
                _ = kernel(inputs, args)
                d = {
                    'psels': safeload(args.selected_kernel_psels),
                    'copts': safeload(args.selected_kernel_copts),
                }
                yield d
        return list(gen())

    def gen_ref(self, entry_config: FlasnConfig, data_root: Path, device: str = None) -> Generator[tuple[FlasnConfig, Path]]:
        import torch
        if device is None:
            device = f'cuda:{torch.cuda.current_device()}'
        with torch.device(device):
            yield from self._do_gen_ref(entry_config, data_root)

    def _do_gen_ref(self, entry_config: FlasnConfig, data_root: Path) -> Generator[tuple[FlasnConfig, Path]]:
        yield self.write_ref(entry_config, data_root / '00-regular.pt')
        gqa = dataclasses.replace(entry_config, N_HEADS=(10, 2))
        yield self.write_ref(gqa, data_root / '01-gqa.pt')
        irregular_hdim = dataclasses.replace(entry_config, D_HEAD=entry_config.D_HEAD - 8)
        yield self.write_ref(irregular_hdim, data_root / '02-irregular_hdim.pt')
        irregular_seqlen = dataclasses.replace(entry_config,
                                               seqlen_q=entry_config.seqlen_q - 7,
                                               seqlen_k=entry_config.seqlen_k - 7)
        yield self.write_ref(irregular_seqlen, data_root / '03-irregular_seqlen.pt')
        irregular_both = dataclasses.replace(irregular_hdim,
                                             seqlen_q=irregular_hdim.seqlen_q - 7,
                                             seqlen_k=irregular_hdim.seqlen_k - 7)
        yield self.write_ref(irregular_both, data_root / '04-irregular_both.pt')
        bshd = dataclasses.replace(irregular_seqlen, storage_flip=(1,2))
        yield self.write_ref(bshd, data_root / '05-bshd.pt')
        # TODO: varlen tests

    def write_ref(self, config: 'Config', pt: Path) -> tuple['Config', Path]:
        from .reference import SdpaReference
        ref_kernel = SdpaReference()
        bidi_inputs = ref_kernel.generate_inputs(config)
        bidi_inputs, outputs = ref_kernel(bidi_inputs)
        d = {
            "bidi_inputs" : asdict(bidi_inputs),
            "bidi_outputs" : asdict(outputs),
        }
        torch.save(d, pt)
        yield (config, pt)

    def run_test(self,
                 entry_config: 'Config',
                 pts: list[Path],
                 kernel_name: str,
                 backend_index: int,
                 device: str = None):
        import torch
        device = f'cuda:{torch.cuda.current_device()}' if device is None else device
        kernel = self.get_kernel(kernel_name)
        def gen_test():
            args = kernel.create_extargs(force_kernel_index=backend_index)
            for pt in pts:
                d = torch.load(pt, map_location=device_str, mmap=True)
                inputs = from_dict(data_class=kernel.PT_INPUT_CLASS, data=d["bidi_inputs"])
                outputs = kernel(inputs, args)
                refs = from_dict(data_class=kernel.PT_REF_CLASS, data=d["bidi_outputs"])
                result, continue_test = kernel.compare(outputs, refs)
                yield result
                if not continue_test:
                    return
        with torch.device(device)
            return list(gen_test())

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
                'bwd_kernel_fuse'   : bwd_kernel_fuse(),
            }
        return self.KERNEL_DICT.get(kernel_name)

