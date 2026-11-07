# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Per-action command execution tracking with real-time output capture
"""

import subprocess
import threading
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path

# Maximum number of output lines retained in memory per action.  Older lines
# are dropped from the head of the ring buffer; the full output always remains
# in the on-disk .stdout/.stderr logs.
OUTPUT_BUFFER_LINES = 5000


class ActionTracker:
    """Tracks single command execution with real-time output capture"""

    def __init__(self, action_id, command, description, cwd, log_dir):
        self.action_id = action_id
        # Command can be string or list
        self.command = command if isinstance(command, list) else command
        self.command_str = ' '.join(str(c) for c in command) if isinstance(command, list) else command
        self.description = description
        self.cwd = cwd
        self.log_dir = Path(log_dir)

        # Process state
        self.process = None
        self.status = 'queued'  # queued → running → completed/failed
        self.returncode = None

        # Single merged output buffer holding (seq, stream, line) tuples in
        # arrival order.  stdout and stderr MUST share one buffer and one
        # sequence counter: with two independent buffers there is no stable
        # index that a client can use as a resume cursor, because a new stdout
        # line inserts itself *before* every stderr line already delivered.
        # `seq` is monotonic and never reused, so it stays a valid cursor even
        # after the ring buffer rotates.
        self._output = deque(maxlen=OUTPUT_BUFFER_LINES)
        self._next_seq = 0

        # Timing
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None

        # Thread safety
        self._lock = threading.Lock()
        self._capture_threads = []

        # Log file paths
        self.stdout_log = self.log_dir / f'{action_id}.stdout'
        self.stderr_log = self.log_dir / f'{action_id}.stderr'

    def start(self):
        """Spawn subprocess and start output capture threads"""
        with self._lock:
            if self.status != 'queued':
                return

            self.status = 'running'
            self.started_at = datetime.now()

            # Ensure log directory exists
            self.log_dir.mkdir(parents=True, exist_ok=True)

            # Spawn subprocess
            # If command is a list, use it directly; if string, use shell=True
            # Redirect stdin from /dev/null to prevent interactive prompts
            if isinstance(self.command, list):
                self.process = subprocess.Popen(
                    self.command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,  # Line-buffered
                    cwd=self.cwd
                )
            else:
                self.process = subprocess.Popen(
                    self.command,
                    shell=True,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,  # Line-buffered
                    cwd=self.cwd
                )

            # Spawn capture threads
            stdout_thread = threading.Thread(
                target=self._capture_stream,
                args=(self.process.stdout, 'stdout', self.stdout_log),
                daemon=True
            )
            stderr_thread = threading.Thread(
                target=self._capture_stream,
                args=(self.process.stderr, 'stderr', self.stderr_log),
                daemon=True
            )

            stdout_thread.start()
            stderr_thread.start()

            self._capture_threads = [stdout_thread, stderr_thread]

            # Monitor process completion in background
            monitor_thread = threading.Thread(
                target=self._monitor_completion,
                daemon=True
            )
            monitor_thread.start()

    def _capture_stream(self, stream, stream_name, log_file):
        """Capture stream line-by-line to the merged buffer and the log file"""
        with open(log_file, 'w') as f:
            for line in iter(stream.readline, ''):
                if not line:
                    break
                with self._lock:
                    # Sequence numbers are handed out under the shared lock, so
                    # the two capture threads interleave in arrival order and
                    # never collide on a seq.
                    self._output.append((self._next_seq, stream_name, line.rstrip('\n')))
                    self._next_seq += 1
                f.write(line)
                f.flush()

    def _monitor_completion(self):
        """Wait for process to complete and update status"""
        self.returncode = self.process.wait()

        # Wait for capture threads to finish
        for thread in self._capture_threads:
            thread.join(timeout=1.0)

        with self._lock:
            self.status = 'completed' if self.returncode == 0 else 'failed'
            self.completed_at = datetime.now()

    def is_running(self):
        """Check if process is still running"""
        with self._lock:
            return self.status == 'running'

    def kill(self):
        """Kill the running process"""
        with self._lock:
            if self.process and self.status == 'running':
                try:
                    self.process.kill()
                    self.status = 'killed'
                    self.completed_at = datetime.now()
                    self.returncode = -9
                    return True
                except Exception:
                    return False
            return False

    def read_output(self, cursor: int = 0):
        """Read merged output starting at sequence number `cursor`.

        Returns (lines, chunk_start, next_cursor, dropped):
          lines        -- interleaved stdout/stderr lines in arrival order
          chunk_start  -- seq of the first returned line; may be *greater* than
                          `cursor` when the ring buffer already dropped the
                          requested lines.  Clients compare this against their
                          own cursor to reject stale/out-of-order responses.
          next_cursor  -- cursor to pass to the next call
          dropped      -- number of lines lost to buffer rotation
        """
        if cursor < 0:
            cursor = 0
        with self._lock:
            snapshot = list(self._output)
            next_seq = self._next_seq

        if not snapshot:
            return [], min(cursor, next_seq), next_seq, 0

        oldest_seq = snapshot[0][0]
        start = min(cursor, next_seq)
        dropped = 0
        if start < oldest_seq:
            dropped = oldest_seq - start
            start = oldest_seq

        lines = [line for seq, _stream, line in snapshot if seq >= start]
        return lines, start, next_seq, dropped

    def get_output(self, from_line=0):
        """Get output lines starting from from_line (per-stream, legacy view)"""
        with self._lock:
            stdout_lines = [line for _s, stream, line in self._output
                            if stream == 'stdout'][from_line:]
            stderr_lines = [line for _s, stream, line in self._output
                            if stream == 'stderr'][from_line:]
            return {
                'stdout': stdout_lines,
                'stderr': stderr_lines,
                'total_stdout': len(stdout_lines),
                'total_stderr': len(stderr_lines),
                'status': self.status,
                'returncode': self.returncode
            }

    def to_dict(self):
        """Serialize tracker state"""
        with self._lock:
            return {
                'action_id': self.action_id,
                'command': self.command_str,  # Always return string representation
                'description': self.description,
                'cwd': self.cwd,
                'status': self.status,
                'returncode': self.returncode,
                'created_at': self.created_at.isoformat(),
                'started_at': self.started_at.isoformat() if self.started_at else None,
                'completed_at': self.completed_at.isoformat() if self.completed_at else None,
                'duration_ms': int((self.completed_at - self.started_at).total_seconds() * 1000)
                               if self.completed_at and self.started_at else None
            }


class TrackerRegistry:
    """Registry for managing multiple action trackers"""

    def __init__(self):
        self._trackers = {}
        self._lock = threading.Lock()

    def create(self, command, description, cwd, log_dir):
        """Create new tracker with unique action_id"""
        action_id = str(uuid.uuid4())
        tracker = ActionTracker(action_id, command, description, cwd, log_dir)

        with self._lock:
            self._trackers[action_id] = tracker

        return tracker

    def get(self, action_id):
        """Get tracker by action_id"""
        with self._lock:
            return self._trackers.get(action_id)

    def get_all(self):
        """Get all active trackers"""
        with self._lock:
            return list(self._trackers.values())

    def remove(self, action_id):
        """Remove tracker from registry"""
        with self._lock:
            self._trackers.pop(action_id, None)

    def clear_all(self):
        """Remove all trackers (user-initiated cleanup)"""
        with self._lock:
            self._trackers.clear()
