# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from pathlib import Path
from ..tdesc import TuningDescription
from ..flash.module import (
    Flash,
    FlashEntry,
    FlashInputMetadata,
)

@dataclass
class FlashOpBackendSelector:
    op_name: str = ''
    backend_index: int = -1

    @staticmethod
    def parse_text(line: str) -> "FlashOpBackendSelector":
        op_name, backend_index = line.split("=")
        return FlashOpBackendSelector(op_name=op_name, backend_index=int(backend_index))

class FlashOp(TuningDescription):
    ENTRY_CLASS = FlashEntry
    INPUT_METADATA = FlashInputMetadata

    KERNEL_DICT = None

    def __init__(self):
        self._flash = Flash()

    def get_entry_choices(self):
        return self._flash.get_entry_choices()

    def validate_entry(self, entry):
        return self._flash.validate_entry(entry)

    def validate_hw_feature(self, arch, entry):
        return self._flash.validate_hw_feature(arch, entry)

    def list_impls(self, entry):
        return ['attn_fwd_op', 'attn_bwd_op']

    def get_impl(self, name: str | FlashOpBackendSelector):
        if isinstance(name, FlashOpBackendSelector):
            name = name.op_name
        if self.KERNEL_DICT is None:
            from .kernels import attn_fwd_op, attn_bwd_op
            self.KERNEL_DICT = {
                'attn_fwd_op': attn_fwd_op(),
                'attn_bwd_op': attn_bwd_op(),
            }
        return self.KERNEL_DICT[name]

    def _do_probe_backends(self, entry, im, which_impl, pt):
        kernel = self.get_impl(which_impl)
        return [{'backend_index': i} for i in range(kernel.BACKEND_COUNT)]

    def _gen_ref(self, entry, data_root: Path, extra_ims: list = []):
        return self._flash._gen_ref(entry, data_root, extra_ims)

    def probe_impl_desc(self, kernel, args) -> dict:
        return {'backend_index': args.backend_index}

    def run_single_test(self, im, pt, which_impl: FlashOpBackendSelector):
        return self._flash.run_single_test(im, pt, which_impl)

    def run_single_benchmark(self, im, pt, which_impl: FlashOpBackendSelector):
        return self._flash.run_single_benchmark(im, pt, which_impl)
