# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import sys
import os
from dataclasses import fields
import json
import math
import dacite
import subprocess
import select
import errno
import time
import fcntl
import logging
from pathlib import Path
from typing import Dict, Any

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


def safeload(s):
    return json.loads(s) if s else None

def parse_python(line: str) -> dict:
    args = line.split(';')
    d = {}
    for assignment in args:
        # print(f'{assignment=}', file=sys.stderr)
        k, v = assignment.split('=', maxsplit=1)
        d[k] = eval(v)
    return d

def asdict_shallow(obj) -> dict:
    return {field.name: getattr(obj, field.name) for field in fields(obj)}

dacite_tuple = dacite.Config(cast=[tuple])

def sanitize_float(value):
    """
    Sanitize float values for PostgreSQL JSONB compatibility.
    - NaN -> None (becomes JSON null)
    - +Inf -> max float32
    - -Inf -> min float32
    """
    if isinstance(value, float):
        if math.isnan(value):
            return None
        elif math.isinf(value):
            if value > 0:
                return float(sys.float_info.max)
            else:
                return float(-sys.float_info.max)
    return value

def sanitize_value(obj):
    """Recursively sanitize an object to replace NaN/Inf values."""
    if isinstance(obj, dict):
        return {k: sanitize_value(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        result = [sanitize_value(item) for item in obj]
        return tuple(result) if isinstance(obj, tuple) else result
    elif isinstance(obj, float):
        return sanitize_float(obj)
    else:
        return obj


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


def _read_line_unix(process: subprocess.Popen, timeout: float) -> str|None:
    """
    Read one line with timeout on Unix systems using select.

    Uses select in a loop to accumulate characters until newline or timeout.
    This avoids blocking on readline() when subprocess hasn't written a newline.
    """
    if not process.stdout:
        raise ValueError("Process stdout is None")

    import time
    import fcntl

    # Set stdout to non-blocking mode
    fd = process.stdout.fileno()
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    try:
        buffer = bytearray()
        deadline = time.time() + timeout

        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                return None  # Timeout

            ready, _, _ = select.select([process.stdout], [], [], remaining)

            if not ready:
                return None  # Timeout

            # Read available data (non-blocking)
            try:
                chunk = process.stdout.read(8192)
                if not chunk:
                    # EOF
                    return buffer.decode('utf-8', errors='replace') if buffer else None

                buffer.extend(chunk)

                # Check if we have a complete line
                if b'\n' in buffer:
                    line_end = buffer.index(b'\n')
                    line = buffer[:line_end + 1].decode('utf-8', errors='replace')
                    # Keep remaining data for next read (though we don't use it here)
                    return line

            except BlockingIOError:
                # No data available right now, continue waiting
                continue

    finally:
        # Restore blocking mode
        fcntl.fcntl(fd, fcntl.F_SETFL, flags)


def get_db_connection_params(workdir: Path) -> Dict[str, Any]:
    """
    Get PostgreSQL connection parameters from workdir config.

    Args:
        workdir: Path to workdir containing config.rc

    Returns:
        Connection parameters dictionary with keys: host, port, user, password, dbname
    """
    # Source config.rc and extract environment variables
    config_rc = workdir / 'config.rc'
    if not config_rc.exists():
        raise FileNotFoundError(f"Config file not found: {config_rc}")

    # Source the config file and print specific variables
    # Note: config.rc sets variables but doesn't export them, so we can't use 'env'
    result = subprocess.run(
        f'. {config_rc} && echo "$CELERY_SERVICE_HOST" && echo "$POSTGRES_PORT" && echo "$POSTGRES_USER" && echo "$POSTGRES_PASSWORD"',
        shell=True,
        capture_output=True,
        text=True,
        executable='/bin/bash'
    )

    if result.returncode != 0:
        raise RuntimeError(f"Failed to source config.rc: {result.stderr}")

    # Parse output lines
    lines = result.stdout.strip().split('\n')
    if len(lines) < 4:
        raise RuntimeError(f"Incomplete config.rc output: {result.stdout}")

    return {
        'host': lines[0] or 'localhost',
        'port': int(lines[1]) if lines[1] else 5432,
        'user': lines[2] or 'aotriton',
        'password': lines[3] or None
    }


