# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Engine utilities that every application needs: logging setup, a defensive
line reader for subprocess pipes, and the by-path loader for family plugins.

Nothing here may import an application, torch or triton (see
python/test/pgdef/test_layering.py).
"""

import errno
import fcntl
import hashlib
import importlib.util
import logging
import os
import select
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def configure_logging_with_flush():
    """
    Configure logging with line-buffered output to ensure logs are written immediately.
    Critical for debugging blocking issues where buffered logs might never appear.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )

    # Force line-buffered output for all handlers
    for handler in logging.root.handlers:
        handler.setStream(sys.stderr)
        if hasattr(handler.stream, 'reconfigure'):
            handler.stream.reconfigure(line_buffering=True)


class SafeLineReader:
    """
    Non-blocking line reader for subprocess stdout with timeout support.

    Maintains buffer state across multiple readline calls to preserve
    partial line data.

    Designed for text mode (text=True) subprocesses.
    """

    def __init__(self, process: subprocess.Popen):
        """
        Initialize reader for a subprocess.

        Args:
            process: subprocess.Popen instance with stdout=subprocess.PIPE in binary mode
        """
        self.process = process
        self.buffer = b""  # Bytes buffer for binary mode
        self.fd = None
        self.original_flags = None

    def _setup_nonblocking(self):
        """Set stdout to non-blocking mode."""
        if self.fd is None and self.process.stdout:
            self.fd = self.process.stdout.fileno()
            self.original_flags = fcntl.fcntl(self.fd, fcntl.F_GETFL)
            fcntl.fcntl(self.fd, fcntl.F_SETFL, self.original_flags | os.O_NONBLOCK)

    def _restore_blocking(self):
        """Restore stdout to blocking mode."""
        if self.fd is not None and self.original_flags is not None:
            fcntl.fcntl(self.fd, fcntl.F_SETFL, self.original_flags)

    def readline(self, timeout: float) -> tuple[str|None, int, str|None]:
        """
        Read one line from subprocess stdout with timeout.

        Args:
            timeout: Maximum time to wait for a line (seconds)

        Returns:
            Tuple of (line, exitcode, error_msg) where:
            - line: The output line if successfully read, None otherwise
            - exitcode: Process exitcode if crashed, errno.ETIMEDOUT if timeout, 0 otherwise
            - error_msg: stderr if any problem occurred, None if successful
        """
        if not self.process.stdout:
            raise ValueError("Process stdout is None")

        def check_crash():
            return_code = self.process.poll()
            if return_code is not None and return_code != 0:
                try:
                    error_msg = self.process.stderr.read().decode('utf-8', errors='replace') if self.process.stderr else ""
                except Exception as e:
                    error_msg = f"Could not read stderr: {e}"
                return (None, return_code, error_msg)
            return None

        # First check if process has already crashed
        ret = check_crash()
        if ret is not None:
            return ret

        # Check if we already have a complete line in buffer
        if b'\n' in self.buffer:
            line_end = self.buffer.index(b'\n')
            line = self.buffer[:line_end + 1]
            # Remove line from buffer, keep rest
            self.buffer = self.buffer[line_end + 1:]
            return (line.rstrip(b'\n\r').decode('utf-8', errors='replace'), 0, None)

        # Setup non-blocking I/O
        self._setup_nonblocking()

        try:
            deadline = time.time() + timeout

            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return (None, errno.ETIMEDOUT, f"Timeout after {timeout} seconds waiting for output")

                ready, _, _ = select.select([self.process.stdout], [], [], remaining)

                if not ready:
                    return (None, errno.ETIMEDOUT, f"Timeout after {timeout} seconds waiting for output")

                # Read available data (non-blocking, binary mode)
                try:
                    chunk = self.process.stdout.read(8192)
                    if not chunk:
                        # EOF - return any remaining buffer as final line
                        if self.buffer:
                            line = self.buffer
                            self.buffer = b""
                            return (line.rstrip(b'\n\r').decode('utf-8', errors='replace'), 0, None)
                        return (None, 0, "EOF")

                    # Debug logging: show chunk received (ASCII-safe)
                    chunk_preview = chunk.decode('ascii', errors='replace').replace('\n', '\\n')
                    logger.info(f"SafeLineReader received {len(chunk)} bytes: {chunk_preview}")

                    self.buffer += chunk

                    # Check if we have a complete line
                    if b'\n' in self.buffer:
                        line_end = self.buffer.index(b'\n')
                        line = self.buffer[:line_end + 1]
                        # Remove line from buffer, keep rest
                        self.buffer = self.buffer[line_end + 1:]

                        # Check if process crashed during read
                        ret = check_crash()
                        if ret is not None:
                            return ret

                        return (line.rstrip(b'\n\r').decode('utf-8', errors='replace'), 0, None)

                except BlockingIOError:
                    # No data available right now, continue waiting
                    continue

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            return (None, -1, f"Unexpected error reading from subprocess: {e}\nTraceback:\n{tb}")

        finally:
            # Note: Don't restore blocking mode here as we may call readline multiple times
            pass

    def close(self):
        """Clean up and restore blocking mode."""
        self._restore_blocking()

def safe_readline(
    process: subprocess.Popen,
    timeout: float = 5.0
) -> tuple[str|None, int, str|None]:
    """
    Defensively read one line from subprocess stdout with timeout and crash protection.

    Args:
        process: subprocess.Popen instance with stdout=subprocess.PIPE
        timeout: Maximum time to wait for a line (seconds)

    Returns:
        Tuple of (line, exitcode, stderr) where:
        - line: The output line if successfully read, None otherwise
        - exitcode: Process exitcode if crashed, errno.ETIMEDOUT if timedout, 0 otherwise
        - error_msg: stderr if any problem occurred, None if successful

    Note: Uses SafeLineReader to maintain buffer state across multiple calls.
          Reader instance is cached on the process object.
    """
    assert sys.platform != 'win32'

    # Get or create reader for this process
    if not hasattr(process, '_stdout_reader'):
        process._stdout_reader = SafeLineReader(process)

    return process._stdout_reader.readline(timeout)


def tree_cache_key(modules_dir, family: str, block: str) -> str:
    """Synthetic module name for `<modules_dir>/<family>/<block>`.

    Keyed on a digest of the RESOLVED `modules_dir`, not on the family name
    alone -- exactly as `python/codegen/parser.py`'s `load_family_aot` is, and
    for the same reason. Two trees can supply the same family (the real
    `modules/` and `python/test/fakefamily/` both have a `flash`), and a
    name-only key means whichever tree loaded first serves both. That is
    invisible while the two agree and silently wrong the moment they diverge.
    """
    digest = hashlib.blake2b(str(Path(modules_dir).resolve()).encode(),
                             digest_size=6).hexdigest()
    return f'_aotriton_modules_{digest}_{family}_{block}'


def load_family_block(family: str, block: str, modules_dir):
    """Import `<modules_dir>/<family>/<block>/__init__.py` by path.

    The package gets a synthetic unique name (see `tree_cache_key`), so
    `modules/<family>` stays a plain directory rather than a package -- the
    same rationale as `python/codegen/parser.py`'s `load_family_aot`.
    `block` is the application's own subdirectory name (`tune`, `visperf`, ...); the engine does not know any of them. Cached in
    `sys.modules` under a tree-keyed name.

    Raises ImportError if the block does not exist.
    """
    modname = tree_cache_key(modules_dir, family, block)
    cached = sys.modules.get(modname)
    if cached is not None:
        return cached
    block_dir = Path(modules_dir) / family / block
    init_path = block_dir / '__init__.py'
    if not init_path.is_file():
        raise ImportError(
            f"No {block} block for family '{family}': {init_path} not found "
            f"(modules_dir={modules_dir}). Set AOTRITON_MODULES_DIR or run "
            f"from the repo root.")
    spec = importlib.util.spec_from_file_location(
        modname, init_path, submodule_search_locations=[str(block_dir)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


def default_modules_dir() -> Path:
    """Resolve the repo's `modules/` directory.

    `modules/` is DATA beside the package source (like codegen's --root_dir),
    never shipped inside the installed `aotriton` package (see setup.py), so
    it cannot be reliably derived from this file's location alone -- a
    non-editable install copies `python/` out of the checkout but never
    ships `modules/` alongside it.

    Resolution order:
      1. `AOTRITON_MODULES_DIR` env var, if set -- the explicit override.
      2. `<cwd>/modules` -- all `.tune/bin/*` / `.tune/remote/*` wrappers `cd`
         into `$AOTRITON_ROOT` before invoking `python3 -m aotriton.<app>.*`
         (see e.g. `.tune/bin/dispatch`), so this covers every production
         call site.
      3. `<repo root inferred from this file>/modules` -- last-resort
         fallback for ad-hoc invocation (e.g. running pytest from the repo
         root with an editable install); only correct when `aotriton` is
         installed editable from this exact checkout.
    """
    env = os.environ.get('AOTRITON_MODULES_DIR')
    if env:
        return Path(env)
    cwd_candidate = Path.cwd() / 'modules'
    if cwd_candidate.is_dir():
        return cwd_candidate
    return Path(__file__).resolve().parent.parent.parent / 'modules'
