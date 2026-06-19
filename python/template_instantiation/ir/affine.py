# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
AffineKernel — the codegen-facing IR for a slim AITER-ASM affine kernel.

Built by the linker from the passive AffineDecl (see specs/finalize.py); the
`ati.affine.*` decorator surface that produces that decl lives in decorators/affine.py.
A slim affine kernel owns no functional space (it has a vendored C++ dispatcher); it
contributes only the operands it SUPPLIES to the operator params-struct union (e.g. the
bwd DQ_ACC) and packages pre-built `.co` files via co_gen.
"""

from pathlib import Path

from aotriton.gpu_targets import AOTRITON_ARCH_TO_PACK
from .cfield import cfield
from .interface import Interface


# DQ_ACC-style supplied operands declare a LazyTensor:*fp32:16 rank-N tensor; map
# the spec to the C struct field type the slim shim's params struct expects.
def _supplied_cfield(spec, index):
    """Build a struct cfield for one @ati.affine.supplies operand. Supports the
    LazyTensor tensors the affine backend adds (e.g. DQ_ACC)."""
    from ..decorators import TensorSpec
    assert isinstance(spec, TensorSpec), \
        f'@ati.affine.supplies currently supports ati.tensor specs only, got {spec!r}'
    aname = spec.arg_names[0]
    dtype = spec.dtype
    rank = spec.rank if spec.rank is not None else 4
    if isinstance(dtype, str) and dtype.startswith('LazyTensor:'):
        ctype = f'LazyTensorInternal<{rank}>*'
    else:
        ctype = f'const TensorView<{rank}>*'
    return cfield(ctype=ctype, aname=aname, index=index, nbits=0)


class AffineKernel(Interface):
    """A slim affine (AITER ASM) kernel built from the @ati.affine.* stacked form.

    ATI-native: subclasses the ATI Interface base (identity surface) and absorbs the
    slim-affine codegen surface directly — co_gen (.co packaging), the empty
    functional space (vendored dispatcher), and the metadata SlimAffineGenerator
    reads (CO_DIR / COOKIE_CLASS / SUPPORTED_ARCH / SHARED_IFACE). It also exposes
    `supplied_operands` (the operands it contributes to the operator params-struct
    union, e.g. the bwd DQ_ACC).

    SHARED_IFACE is filled in by infer_shared_iface (operator -> backend); the
    @ati.affine.shared_operator name is kept for reference."""

    CODEGEN_MODULE = 'affine'
    TUNE_NAME = None
    FILE_PFX = 'affine'
    ENUM_PREFIX = 'kSlimAffine_'
    AFFINE_KERNEL_ROOT = Path('aiter/hsa')
    is_tunable = False

    def __init__(self, *, name, family, co_dir, cookie, headers, supported_arch,
                 choice_filters, shared_operator_name, supplied_specs, disable,
                 supplies_after=None, supplies_before=None):
        self.NAME = name
        self.FAMILY = family
        self.CO_DIR = co_dir
        self.COOKIE_CLASS = cookie
        self.HEADER_EXTRA_INCLUDES = list(headers)
        self.SUPPORTED_ARCH = list(supported_arch)
        self.CHOICE_FILTERS = dict(choice_filters)
        self.shared_operator_name = shared_operator_name
        self._supplied_specs = list(supplied_specs)
        self._supplies_after = supplies_after
        self._supplies_before = supplies_before
        self._disable = disable      # DisableSpec | None
        # Precompute the cfields this backend supplies to the operator union (DQ_ACC).
        self._supplied_cfields = [_supplied_cfield(s, i)
                                  for i, s in enumerate(self._supplied_specs)]

    @property
    def param_class_name(self):
        # The params struct is the shared operator's (the Operator exposes it).
        return self.SHARED_IFACE.param_class_name

    # --- slim-affine codegen surface (absorbed from SlimAffineKernelDescription) ---

    @property
    def perf_cfields(self):
        return []

    def gen_functionals(self, build_for_target_arch):
        # Slim affine kernels have a vendored C++ dispatcher — no functional space.
        yield from ()

    def list_functional_params(self):
        return []

    def co_gen(self, build_dir: Path, build_for_target_arch):
        """Yield (aks2_path, inarchive_path, kernel_co) for every prebuilt .co under
        the supported arches — the relocate rules that pack the AITER .co files."""
        target_arch = {arch: gpus for arch, gpus in build_for_target_arch.items()
                       if arch in self.SUPPORTED_ARCH}
        archless_package_path = Path(self.FAMILY) / 'affine_kernels' / self.CO_DIR
        for arch in target_arch.keys():
            aks2_path = Path(f'amd-{AOTRITON_ARCH_TO_PACK[arch]}') / archless_package_path
            aiter_arch = build_dir / self.AFFINE_KERNEL_ROOT / arch
            aiter_arch_module = aiter_arch / self.CO_DIR
            for kernel_co in aiter_arch_module.glob('**/*.co'):
                inarchive_path = kernel_co.relative_to(aiter_arch).as_posix()
                yield (aks2_path, inarchive_path, kernel_co)

    # --- operator-union contribution -------------------------------------
    #
    # The operator's params struct (build_merged_struct_cfields) is the union over
    # all backends' func_cfields. A slim affine kernel contributes only its SUPPLIED
    # operands (DQ_ACC); `func_cfields` exposes their cfields, and `union_order`
    # gives the ordering chain (after-anchor, supplied..., before-anchor) so the
    # merge places them at the right struct index.

    @property
    def func_cfields(self):
        return list(self._supplied_cfields)

    @property
    def union_order(self):
        names = [cf.aname for cf in self._supplied_cfields]
        if not names:
            return []
        chain = []
        if self._supplies_after is not None:
            chain.append(self._supplies_after)
        chain += names
        if self._supplies_before is not None:
            chain.append(self._supplies_before)
        return chain

    @property
    def supplied_operands(self):
        """The TensorSpecs this affine backend contributes to the operator union."""
        return list(self._supplied_specs)

    def is_functional_disabled(self, functional):
        if self._disable is None:
            return False
        return self._disable.when(functional)
