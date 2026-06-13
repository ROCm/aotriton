# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
ATI Interface base — the shared codegen-identity surface.

Every ATI description the generator consumes (kernel / operator / metro / affine /
conditional) carries the same identity & C++-naming surface: FAMILY / NAME /
CODEGEN_MODULE, the `unique_path` selective key, and the `<Base>{Params,Context,
Metadata}` / enum-name rules. This base factors that surface so each concrete class
declares only its distinct behavior (functional/tuning surface, launcher methods,
co_gen, branch refs) plus a few class attrs.

NOTE: this is NOT the legacy `aotriton.base.interface.Interface`. That one also
carried the pre-ATI functional-generation machinery (TYPE_CHOICES / FEAT_CHOICES /
ARGUMENTS dicts, `__init__`/`_late_init`) which the ATI enumeration replaced. The ATI
base is identity-only — no functional `__init__`.
"""

from pathlib import Path


class Interface:
    # --- identity / codegen wiring (subclasses set these) -------------------
    FAMILY = None              # e.g. 'flash'
    NAME = None                # e.g. 'attn_fwd' (often per-instance, set in __init__)
    CODEGEN_MODULE = None      # 'triton' | 'op' | 'affine'  (selective-path segment)
    TUNE_NAME = None           # 'autotune' | 'optune' | None
    FILE_PFX = 'iface'         # 'shim' (kernel) | 'iface' (operator) | 'affine'
    ENUM_PREFIX = None         # 'kShim_' | 'kOp_' | 'kMetro_' | 'kSlimAffine_' | 'kConditional_'
    SHARED_IFACE = None        # the operator whose params struct this borrows (or None)
    HEADER_EXTRA_INCLUDES = []
    SOURCE_EXTRA_INCLUDES = []

    # --- C++ name helpers --------------------------------------------------

    @staticmethod
    def _name_to_base(name):
        return ''.join(x.capitalize() for x in name.lower().split('_'))

    @property
    def class_name_base(self):
        return self._name_to_base(self.NAME)

    @property
    def enum_name(self):
        return f'{self.ENUM_PREFIX}{self.class_name_base}'

    @property
    def param_class_name(self):
        # Default: this description owns its params struct. Subclasses that BORROW an
        # operator's struct (the triton kernel, the affine kernel) override this to
        # SHARED_IFACE.param_class_name.
        return self.class_name_base + 'Params'

    @property
    def context_class_name(self):
        return self.class_name_base + 'Context'

    @property
    def metadata_class_name(self):
        return self.class_name_base + 'Metadata'

    # --- selective-build key ----------------------------------------------

    @property
    def unique_path(self) -> Path:
        return Path(self.FAMILY) / self.CODEGEN_MODULE / self.NAME

    # --- tuning fallback (operator-level partial tune; default identity) ---

    @property
    def partially_tuned_functionals(self) -> dict:
        return {}

    def fallback_compact_dict(self, compact_dict):
        fb = self.partially_tuned_functionals
        return {k: fb.get(k, v) for k, v in compact_dict.items()}

    # --- selective execution (overridden by kernels that support it) -------

    def iter_kernel_slot_names(self):
        yield from ()
