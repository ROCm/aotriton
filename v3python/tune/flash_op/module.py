# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
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

class FlashOp(Flash):
    ENTRY_CLASS = FlashEntry
    INPUT_METADATA = FlashInputMetadata

    KERNEL_DICT = None

    def list_impls(self, entry, arch: str | None = None):
        # TODO: rename attn_fwd_op/attn_bwd_op to match the canonical op names
        # defined in v3python/rules/flash/ (op_attn_fwd, op_attn_bwd) so that
        # export_best_results.py no longer needs OP_NAME_MAP for translation.
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

    def probe_impl_desc(self, kernel, args) -> dict:
        return {'backend_index': args.backend_index}
