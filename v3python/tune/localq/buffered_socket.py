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


class SendState(Enum):
    """State machine for sending buffered messages"""
    IDLE = 1  # No pending sends
    SENDING = 2  # Currently sending data


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

        # Receive buffering state
        self.recv_buffer = bytearray()  # Current partial message being received
        self.recv_state = RecvState.READING_LENGTH
        self.expected_length = None  # Set after reading length prefix

        # Send buffering state
        self.send_buffer = b''
        self.send_state = SendState.IDLE
        self.send_offset = 0  # How many bytes of send_buffer have been sent

    @property
    def fileno(self) -> int:
        """Get socket file descriptor"""
        return self.sock.fileno()

    @property
    def wants_write(self) -> bool:
        """Check if socket wants to write (has pending data)"""
        return len(self.send_buffer) > 0

    def _recv_length(self) -> Optional[bool]:
        """
        Receive 4-byte length prefix.

        Returns:
            True if complete, False if need more data, None if connection closed
        """
        needed = 4 - len(self.recv_buffer)
        if needed > 0:
            try:
                chunk = self.sock.recv(needed)
                if not chunk:
                    # Connection closed
                    return None
                self.recv_buffer.extend(chunk)
            except BlockingIOError:
                # No more data available
                return False

        # Check if we have complete length prefix
        if len(self.recv_buffer) < 4:
            return False  # Need more data

        # Parse length prefix
        self.expected_length = struct.unpack('>I', self.recv_buffer)[0]

        # Sanity check (prevent DoS)
        if self.expected_length > 100 * 1024 * 1024:  # 100MB max
            raise ValueError(f"Message too large: {self.expected_length} bytes")

        # Pre-allocate buffer for payload
        self.recv_buffer = bytearray(self.expected_length)
        self.recv_state = RecvState.READING_PAYLOAD
        return True

    def _recv_payload(self) -> Optional[dict]:
        """
        Receive payload bytes.

        Returns:
            Message dict if complete, False if need more data, None if connection closed
        """
        received = len(self.recv_buffer)
        needed = self.expected_length - received

        if needed > 0:
            try:
                # Receive into pre-allocated buffer
                view = memoryview(self.recv_buffer)[received:]
                nbytes = self.sock.recv_into(view, needed)
                if nbytes == 0:
                    # Connection closed mid-message
                    return None
                # Buffer is already updated by recv_into
            except BlockingIOError:
                # No more data available
                return False

        # Check if we have complete payload
        if len(self.recv_buffer) < self.expected_length:
            return False  # Need more data

        # Parse complete message
        msg = json.loads(bytes(self.recv_buffer).decode('utf-8'))

        logger.debug(f"← RECV ({self.expected_length} bytes): {json.dumps(msg)}")

        # Reset for next message
        self.recv_buffer = bytearray()
        self.recv_state = RecvState.READING_LENGTH
        self.expected_length = None

        return msg

    def recv_messages(self):
        """
        Generator that yields complete messages as they arrive.

        Yields:
            dict: Complete messages

        Raises:
            ConnectionError: If connection is closed
        """
        while True:
            if self.recv_state == RecvState.READING_LENGTH:
                result = self._recv_length()
                if result is None:
                    # Connection closed
                    raise ConnectionError("Connection closed")
                elif result is False:
                    # Need more data, stop iteration
                    return
                # else: length received, continue to read payload

            elif self.recv_state == RecvState.READING_PAYLOAD:
                result = self._recv_payload()
                if result is None:
                    # Connection closed
                    raise ConnectionError("Connection closed")
                elif result is False:
                    # Need more data, stop iteration
                    return
                else:
                    # Complete message received
                    yield result

    def queue_message(self, msg: dict):
        """
        Queue message for sending (non-blocking).

        For non-blocking sockets, we can't use sendall(). Instead:
        1. Serialize message to bytes
        2. Append to send buffer
        3. Caller must register for EPOLLOUT and call flush_send()

        Args:
            msg: Message dictionary
        """
        json_bytes = json.dumps(msg).encode('utf-8')
        length = len(json_bytes)

        # Create length-prefixed message
        message_bytes = struct.pack('>I', length) + json_bytes

        # Append to send buffer
        self.send_buffer += message_bytes

        logger.debug(f"→ QUEUE ({length} bytes): {json.dumps(msg)}")

    def flush_send(self) -> bool:
        """
        Flush send buffer (call when EPOLLOUT signals socket is ready).

        Returns:
            True if all data sent, False if more data remains
        """
        if not self.send_buffer:
            return True  # Nothing to send

        try:
            # Send as much as possible
            sent = self.sock.send(self.send_buffer[self.send_offset:])
            self.send_offset += sent

            # Check if everything sent
            if self.send_offset >= len(self.send_buffer):
                # All sent, reset buffer
                self.send_buffer = b''
                self.send_offset = 0
                return True
            else:
                # More data remains
                return False

        except BlockingIOError:
            # Socket buffer full, try again later
            return False
