# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
ATI passive spec data models (pipeline Stage 2).

The "object files" the decorators produce and the linker consumes: KernelSpec,
OperatorDecl, AffineDecl, MetroPlan, TuneSpec, and the stacked-@ finalize glue. These
carry authoring data only — no building (that is the builder/ stage).
"""
