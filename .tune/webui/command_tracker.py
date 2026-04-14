# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Command execution tracking and output storage
"""

import subprocess
import threading
from datetime import datetime
from collections import deque


class CommandTracker:
    """Tracks command execution and stores output history"""

    def __init__(self, max_commands=100):
        self.max_commands = max_commands
        self._outputs = deque(maxlen=max_commands)
        self._lock = threading.Lock()
        self._id_counter = 0

    def execute(self, cmd, workdir=None, description=None):
        """
        Execute a command and track its output

        Args:
            cmd: Command string to execute
            workdir: Working directory for command execution
            description: Human-readable description of the command

        Returns:
            dict with stdout, stderr, returncode
        """
        with self._lock:
            self._id_counter += 1
            cmd_id = self._id_counter

        # Create command record
        record = {
            'id': cmd_id,
            'command': cmd,
            'description': description or cmd,
            'workdir': workdir,
            'start_time': datetime.now().isoformat(),
            'status': 'running',
            'stdout': '',
            'stderr': '',
            'returncode': None
        }

        with self._lock:
            self._outputs.append(record)

        # Execute command (currently debug mode)
        # TODO: Uncomment when ready to execute real commands
        # try:
        #     result = subprocess.run(
        #         cmd,
        #         shell=True,
        #         capture_output=True,
        #         text=True,
        #         cwd=workdir,
        #         timeout=300
        #     )
        #     record['stdout'] = result.stdout
        #     record['stderr'] = result.stderr
        #     record['returncode'] = result.returncode
        #     record['status'] = 'completed' if result.returncode == 0 else 'failed'
        # except Exception as e:
        #     record['stderr'] = str(e)
        #     record['returncode'] = 1
        #     record['status'] = 'failed'
        # finally:
        #     record['end_time'] = datetime.now().isoformat()

        # DEBUG mode
        record['stdout'] = f'[DEBUG] Would execute: {cmd}'
        record['stderr'] = ''
        record['returncode'] = 0
        record['status'] = 'completed'
        record['end_time'] = datetime.now().isoformat()

        return {
            'stdout': record['stdout'],
            'stderr': record['stderr'],
            'returncode': record['returncode']
        }

    def get_all(self):
        """Get list of all command outputs (most recent first)"""
        with self._lock:
            return list(reversed(self._outputs))

    def get_by_id(self, cmd_id):
        """Get specific command output by ID"""
        with self._lock:
            for cmd in self._outputs:
                if cmd['id'] == cmd_id:
                    return dict(cmd)  # Return a copy
        return None

    def clear(self):
        """Clear all command outputs"""
        with self._lock:
            self._outputs.clear()
            self._id_counter = 0

    def record_action(self, action_tracker):
        """Record action tracker in global history (in-memory only)"""
        # Convert tracker to execution record
        record = action_tracker.to_dict()
        record['stdout_path'] = str(action_tracker.stdout_log)
        record['stderr_path'] = str(action_tracker.stderr_log)

        with self._lock:
            self._outputs.append(record)


# Global instance
_tracker = CommandTracker()


def get_tracker():
    """Get the global command tracker instance"""
    return _tracker
