# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Buffered socket for non-blocking length-prefixed message I/O.

Handles state machine for reading partial messages from non-blocking sockets.
"""

import socket
import struct
import json
import logging
from enum import Enum
from typing import List, Optional

logger = logging.getLogger(__name__)


class RecvState(Enum):
    """State machine for receiving length-prefixed messages"""
    READING_LENGTH = 1  # Reading 4-byte length prefix
    READING_PAYLOAD = 2  # Reading N-byte JSON payload


class BufferedSocket:
    """
    Wrapper for non-blocking socket with message buffering.

    Handles reading length-prefixed JSON messages from non-blocking sockets.
    Maintains internal buffer and state machine to accumulate partial messages.
    """

    def __init__(self, sock: socket.socket, worker_id: Optional[str] = None):
        """
        Initialize buffered socket.

        Args:
            sock: Non-blocking socket
            worker_id: Optional worker identifier
        """
        self.sock = sock
        self.worker_id = worker_id
        self.recv_buffer = b''
        self.state = RecvState.READING_LENGTH
        self.expected_length = None  # Set after reading length prefix

    @property
    def fileno(self) -> int:
        """Get socket file descriptor"""
        return self.sock.fileno()

    def recv_messages(self) -> Optional[List[dict]]:
        """
        Read available data and parse complete messages.

        Returns:
            List of complete messages, or None if connection closed
        """
        # Read available data from socket
        try:
            chunk = self.sock.recv(65536)  # Read up to 64KB
            if not chunk:
                # Connection closed
                return None
            self.recv_buffer += chunk
        except BlockingIOError:
            # No data available (shouldn't happen after epoll signals EPOLLIN)
            pass

        # Parse complete messages from buffer
        messages = []

        while True:
            if self.state == RecvState.READING_LENGTH:
                # Need 4 bytes for length prefix
                if len(self.recv_buffer) < 4:
                    break  # Not enough data yet

                # Parse length prefix
                self.expected_length = struct.unpack('>I', self.recv_buffer[:4])[0]

                # Sanity check (prevent DoS)
                if self.expected_length > 100 * 1024 * 1024:  # 100MB max
                    raise ValueError(f"Message too large: {self.expected_length} bytes")

                # Remove length prefix from buffer
                self.recv_buffer = self.recv_buffer[4:]
                self.state = RecvState.READING_PAYLOAD

            elif self.state == RecvState.READING_PAYLOAD:
                # Need expected_length bytes for payload
                if len(self.recv_buffer) < self.expected_length:
                    break  # Not enough data yet

                # Extract JSON payload
                json_bytes = self.recv_buffer[:self.expected_length]
                msg = json.loads(json_bytes.decode('utf-8'))
                messages.append(msg)

                logger.debug(f"← RECV ({self.expected_length} bytes): {json.dumps(msg)}")

                # Remove payload from buffer
                self.recv_buffer = self.recv_buffer[self.expected_length:]

                # Reset state for next message
                self.state = RecvState.READING_LENGTH
                self.expected_length = None

        return messages

    def send_message(self, msg: dict):
        """
        Send length-prefixed JSON message.

        Args:
            msg: Message dictionary
        """
        json_bytes = json.dumps(msg).encode('utf-8')
        length = len(json_bytes)

        # Send 4-byte length prefix (big-endian)
        self.sock.sendall(struct.pack('>I', length))

        # Send JSON payload
        self.sock.sendall(json_bytes)

        logger.debug(f"→ SEND ({length} bytes): {json.dumps(msg)}")
