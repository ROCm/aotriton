# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
A defensive line reader for a subprocess's stdout: readline with a timeout
that also notices the process dying, keeping its buffer across calls.

Not specific to any one user; pgdef's exaid proxy reads its runners with it.
"""

import errno
import fcntl
import logging
import os
import select
import subprocess
import sys
import time

logger = logging.getLogger(__name__)


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
