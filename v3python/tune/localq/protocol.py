# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Message protocol for Unix socket communication.

Uses length-prefix framing: [4 bytes length][N bytes JSON payload]
"""

import struct
import json
import socket
import logging

logger = logging.getLogger(__name__)


def send_message(sock: socket.socket, msg: dict):
    """
    Send length-prefixed JSON message.

    Args:
        sock: Socket to send on
        msg: Message dictionary
    """
    json_bytes = json.dumps(msg).encode('utf-8')
    length = len(json_bytes)

    # Send 4-byte length prefix (big-endian)
    sock.sendall(struct.pack('>I', length))

    # Send JSON payload
    sock.sendall(json_bytes)

    logger.debug(f"→ SEND ({length} bytes): {json.dumps(msg)}")


def recv_message(sock: socket.socket) -> dict | None:
    """
    Receive length-prefixed JSON message.

    Args:
        sock: Socket to receive from

    Returns:
        Message dictionary, or None if connection closed
    """
    # Read 4-byte length prefix
    length_bytes = recv_exactly(sock, 4)
    if not length_bytes:
        return None  # Connection closed

    length = struct.unpack('>I', length_bytes)[0]

    # Sanity check (prevent DoS)
    if length > 100 * 1024 * 1024:  # 100MB max
        raise ValueError(f"Message too large: {length} bytes")

    # Read exact JSON payload
    json_bytes = recv_exactly(sock, length)
    msg = json.loads(json_bytes.decode('utf-8'))

    logger.debug(f"← RECV ({length} bytes): {json.dumps(msg)}")
    return msg


def recv_exactly(sock: socket.socket, n: int) -> bytes:
    """
    Receive exactly n bytes from socket.

    Args:
        sock: Socket to receive from
        n: Number of bytes to receive

    Returns:
        Exact n bytes

    Raises:
        ConnectionError: If socket closed before n bytes received
    """
    data = b''
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Socket closed before receiving complete message")
        data += chunk
    return data
