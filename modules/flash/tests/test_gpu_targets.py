#!/usr/bin/env python
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Regression tests for aotriton.gpu_targets.

Covers the failure mode reported in #169: when the requested arch list is
fully filtered out (e.g. PYTORCH_ROCM_ARCH=gfx900 against the supported set),
``gpu_targets.py`` used to silently print an empty string and exit 0,
which then caused a cryptic argparse error in ``generate.py --target_gpus``.
After the fix, ``gpu_targets.py`` exits 2 with a clear stderr message naming
the requested / filtered / supported archs.

These tests shell out to the modules so they exercise the real CLI surface
that CMake's ``execute_process`` invokes. No GPU or ROCm runtime required.
"""
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(module: str, *args: str) -> "subprocess.CompletedProcess":
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )


@pytest.mark.parametrize("module", ["aotriton.gpu_targets"])
def test_unsupported_arch_fails_loud(module):
    """gfx900 is unsupported on every aotriton release. Filtering it out must
    not pass through silently — exit non-zero with a diagnostic naming the
    request, the unsupported set, and the supported set."""
    result = _run(module, "--target_arch", "gfx900")
    assert result.returncode == 2, (
        f"{module} should exit 2 on empty-result, got {result.returncode}\n"
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )
    assert "no supported target arch" in result.stderr
    assert "gfx900" in result.stderr
    assert "supported:" in result.stderr
    # stdout must not leak an empty list to a downstream consumer
    assert result.stdout.strip() == ""


@pytest.mark.parametrize("module", ["aotriton.gpu_targets"])
def test_supported_arch_still_works(module):
    """Sanity: a supported arch resolves cleanly (no regression on the
    happy path)."""
    result = _run(module, "--target_arch", "gfx942")
    assert result.returncode == 0, f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    assert "gfx942" in result.stdout


@pytest.mark.parametrize("module", ["aotriton.gpu_targets"])
def test_mixed_arch_drops_unsupported_keeps_supported(module):
    """A multi-arch build with at least one supported target must keep the
    supported subset rather than fail. This is the multi-arch release shape
    @ScottTodd flagged in #169 as the dominant case."""
    result = _run(module, "--target_arch", "gfx900", "gfx942")
    assert result.returncode == 0, f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    assert "gfx942" in result.stdout
    assert "gfx900" not in result.stdout


@pytest.mark.parametrize("module", ["aotriton.gpu_targets"])
def test_no_args_fails_loud(module):
    """Bare invocation (no archs at all) must also fail loud rather than
    print an empty string."""
    result = _run(module)
    assert result.returncode == 2
    assert "no supported target arch" in result.stderr


@pytest.mark.parametrize("module", ["aotriton.gpu_targets"])
def test_comma_separator_blob_diagnosed(module):
    """PyTorch's PYTORCH_ROCM_ARCH often leaks a comma-joined blob like
    'gfx942,gfx950' as a single CMake list element. The error path should
    flag the comma rather than just listing supported archs."""
    result = _run(module, "--target_arch", "gfx942,gfx950")
    assert result.returncode == 2
    assert "no supported target arch" in result.stderr
    assert "comma" in result.stderr.lower()
    assert "gfx942,gfx950" in result.stderr
