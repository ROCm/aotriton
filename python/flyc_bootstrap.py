# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Environment setup for cross-compiling FlyDSL kernels with no GPU present.

Two problems, both invisible until you hit them, and both must be solved
*before* ``flydsl`` (or anything that imports it) is imported:

1. ``ROCM_PATH`` has to point at the directory that has ``llvm/bin/ld.lld``
   at exactly that relative path, or lld invocation fails inside flydsl with
   no further diagnostic (``resolve_rocm_path``).
2. flydsl's compiler module imports ``torch`` for a handful of dtype
   constants and a stream type, but the venv this runs in must never get a
   real torch install (``ensure_flydsl_importable``).

``setup(arch)`` does both, plus sets the environment variables that put
flydsl into ahead-of-time, GPU-less, compile-only mode.

**One interim.** The vendored kernels under ``modules/<family>/flyc/`` import
``from kernels.common import ...``, and ``kernels/`` is not part of the
``flydsl`` wheel at all. Until it is, ``setup()`` also puts a FlyDSL source
checkout root — ``$AOTRITON_FLYDSL_KERNEL_ROOT``, which
``third_party/flydsl-kernel.txt`` pins and CMake clones — on ``sys.path`` so
that import resolves verbatim. Delete that part of ``setup()`` the day
``kernels.common`` ships in the wheel; nothing else in this file depends on
it, and no vendored kernel has to change. See ``docs/FlyDSL.md``.
"""

import importlib.util
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path


def resolve_rocm_path() -> str:
    """Return a directory ``D`` where ``D/llvm/bin/ld.lld`` exists, and set
    ``os.environ['ROCM_PATH']`` to it.

    Tried in order: an existing ``ROCM_PATH`` (validated, not trusted),
    ``rocm-sdk path --root`` (only when ``ROCM_PATH`` is unset and the
    ``rocm-sdk`` CLI is available), the ``_rocm_sdk_core`` package's bundled
    lib directory, then ``/opt/rocm``. Raises ``RuntimeError`` naming every
    candidate tried and why each one failed if none validates — the native
    failure mode this avoids is flydsl's ``error: lld invocation failed``,
    which carries no further detail about what was wrong with the toolchain.
    """
    tried = []

    def _check(candidate) -> str | None:
        if not candidate:
            tried.append((str(candidate), 'empty/unset'))
            return None
        candidate = Path(candidate)
        lld = candidate / 'llvm' / 'bin' / 'ld.lld'
        if lld.is_file():
            return str(candidate)
        tried.append((str(candidate), f'{lld} does not exist'))
        return None

    env_candidate = os.environ.get('ROCM_PATH')
    found = _check(env_candidate)

    if found is None and not env_candidate:
        rocm_sdk = shutil.which('rocm-sdk')
        if rocm_sdk is not None:
            try:
                out = subprocess.run([rocm_sdk, 'path', '--root'],
                                      capture_output=True, text=True, check=True)
                found = _check(out.stdout.strip())
            except subprocess.CalledProcessError as e:
                tried.append(('<rocm-sdk path --root>', f'command failed: {e}'))
        else:
            tried.append(('<rocm-sdk CLI>', 'shutil.which returned None'))

    if found is None:
        spec = importlib.util.find_spec('_rocm_sdk_core')
        if spec is not None and spec.origin:
            found = _check(Path(spec.origin).parent / 'lib')
        else:
            tried.append(('<_rocm_sdk_core package>', 'importlib.util.find_spec returned None'))

    if found is None:
        found = _check('/opt/rocm')

    if found is None:
        detail = '\n'.join(f'  - {path}: {reason}' for path, reason in tried)
        raise RuntimeError(
            'resolve_rocm_path: could not find a ROCM_PATH containing '
            "'llvm/bin/ld.lld'. Candidates tried:\n" + detail
        )

    os.environ['ROCM_PATH'] = found
    return found


def _install_torch_stub() -> None:
    """Install the minimal fake ``torch``/``torch.cuda`` modules flydsl needs.

    Exactly the surface ``flydsl.compiler`` touches at import time: a
    distinct, hashable sentinel per dtype name (they are dict keys in
    ``_TORCH_DTYPE_TO_MLIR_BUILDER``), a real empty ``Tensor`` class (used
    as a registry key and in an ``issubclass`` scan), and a real empty
    ``cuda.Stream`` class. Nothing here is meant to behave like torch —
    the build venv must never actually compute with it.
    """

    class _StubDType:
        __slots__ = ('_name',)

        def __init__(self, name: str):
            self._name = name

        def __repr__(self) -> str:
            return f'torch.{self._name}'

    torch_stub = types.ModuleType('torch')
    for name in (
        'float16', 'bfloat16', 'float32', 'float64',
        'bool', 'uint8', 'int8', 'int16', 'int32', 'int64',
        # Optional 8-bit float dtypes; present on real torch builds that have
        # them, harmless sentinels here either way.
        'float8_e5m2', 'float8_e4m3fn', 'float8_e5m2fnuz', 'float8_e4m3fnuz',
    ):
        setattr(torch_stub, name, _StubDType(name))

    class Tensor:
        pass

    torch_stub.Tensor = Tensor
    torch_stub.__flydsl_aot_stub__ = True

    cuda_stub = types.ModuleType('torch.cuda')

    class Stream:
        pass

    cuda_stub.Stream = Stream
    torch_stub.cuda = cuda_stub

    sys.modules['torch'] = torch_stub
    sys.modules['torch.cuda'] = cuda_stub


def ensure_flydsl_importable() -> bool:
    """Make ``import flydsl.compiler`` succeed without a real torch install.

    Tries the import as-is first. If it fails with ``ModuleNotFoundError``
    for ``torch`` specifically, installs the stub from ``_install_torch_stub``
    and retries; any other exception (including a ``ModuleNotFoundError`` for
    something else) is re-raised unchanged. Returns whether the stub was
    installed, so a caller can print a one-line notice — the guard means the
    stub self-disables the day flydsl drops its torch import.
    """
    try:
        import flydsl.compiler  # noqa: F401
        return False
    except ModuleNotFoundError as e:
        if e.name != 'torch':
            raise
    _install_torch_stub()
    import flydsl.compiler  # noqa: F401
    return True


def _find_flydsl_checkout_root() -> Path:
    """Locate the FlyDSL source tree that has ``kernels/common``.

    ``AOTRITON_FLYDSL_KERNEL_ROOT``, which is REQUIRED -- there is no default.
    Raises ``RuntimeError`` if it is unset, or if ``kernels/common`` is not
    found under it.

    FlyDSL enters this build twice, and the two pins are independent:

    * ``third_party/flydsl-compiler.txt`` -- the pip pin for the *compiler*
      (``flydsl==0.3.1``), installed into the build venv.
    * ``third_party/flydsl-kernel.txt`` -- the git ref for the *source tree*,
      shallow-cloned like ``third_party/aiter.txt``, which is what this
      function points at. The wheel does not ship ``kernels/common``, and the
      vendored kernels in ``modules/flash/flyc/`` import it verbatim.

    There used to be a ``<repo>/third_party/flydsl`` fallback here, from when
    FlyDSL was expected to be a submodule. Task 2.5 pinned a wheel instead, so
    that directory is never created and the fallback could only ever fail --
    and it failed *misleadingly*: with the ``aotriton`` package installed
    non-editably into the build venv, as CMake installs it, ``__file__`` sits
    in site-packages, so the message advertised a path like
    ``<site-packages>/third_party/flydsl`` and invited the reader to go
    looking for a repo that is not there.
    """
    env_root = os.environ.get('AOTRITON_FLYDSL_KERNEL_ROOT')
    if not env_root:
        raise RuntimeError(
            "_find_flydsl_checkout_root: AOTRITON_FLYDSL_KERNEL_ROOT is not "
            "set, and there is no default. It must point at the root of a "
            "FlyDSL source tree -- the build clones one per "
            "third_party/flydsl-kernel.txt. The compiler wheel pinned by "
            "third_party/flydsl-compiler.txt does not ship 'kernels/common', "
            "which the vendored kernels import, so the source tree is a "
            "separate requirement from the wheel."
        )

    root = Path(env_root)
    if not (root / 'kernels' / 'common').is_dir():
        raise RuntimeError(
            f"_find_flydsl_checkout_root: 'kernels/common' not found under "
            f"{root} (from AOTRITON_FLYDSL_KERNEL_ROOT). It must be the ROOT "
            f"of a FlyDSL source tree, not its kernels/ subdirectory."
        )
    return root


def setup(arch: str) -> bool:
    """Prepare the process to cross-compile FlyDSL kernels for ``arch``.

    Resolves ``ROCM_PATH``, makes ``flydsl`` importable without torch, puts
    the Task-0-interim checkout root on ``sys.path`` if needed, and sets the
    environment variables that select ahead-of-time, GPU-less, compile-only
    mode: ``ARCH``, ``FLYDSL_GPU_ARCH``, ``COMPILE_ONLY=1``,
    ``FLYDSL_RUNTIME_ENABLE_CACHE=0``.

    Returns whether the torch stub was installed (see
    ``ensure_flydsl_importable``).
    """
    resolve_rocm_path()
    stubbed = ensure_flydsl_importable()

    checkout_root = _find_flydsl_checkout_root()
    checkout_root_str = str(checkout_root)
    if checkout_root_str not in sys.path:
        sys.path.insert(0, checkout_root_str)

    os.environ['ARCH'] = arch
    os.environ['FLYDSL_GPU_ARCH'] = arch
    os.environ['COMPILE_ONLY'] = '1'
    os.environ['FLYDSL_RUNTIME_ENABLE_CACHE'] = '0'

    return stubbed
