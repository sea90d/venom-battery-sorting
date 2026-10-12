"""Read CartesianInfo from UR's read-only primary port without sending bytes.

Wire layout: https://docs.universal-robots.com/tutorials/communication-protocol-tutorials/primary-secondary-guide.html
The current offset does not establish which offset was used in past calibration.
"""
from __future__ import annotations

import math
import socket
import struct
import time


MAX_PACKET_BYTES = 1_048_576


def parse_cartesian_info(packet: bytes) -> dict | None:
    if len(packet) < 5:
        raise ValueError("Truncated primary packet")
    length, kind = struct.unpack_from("!IB", packet)
    if length != len(packet) or not 5 <= length <= MAX_PACKET_BYTES:
        raise ValueError("Invalid primary packet length")
    if kind != 16:  # ROBOT_STATE
        return None
    cursor = 5
    result = None
    while cursor < length:
        if length - cursor < 5:
            raise ValueError("Truncated state subpackage")
        size, subtype = struct.unpack_from("!IB", packet, cursor)
        if size < 5 or cursor + size > length:
            raise ValueError("Invalid state subpackage length")
        if subtype == 4:  # CARTESIAN_INFO: pose then active TCP offset
            if size < 101 or result is not None:
                raise ValueError("Missing or duplicate CartesianInfo values")
            values = struct.unpack_from("!12d", packet, cursor + 5)
            if not all(math.isfinite(value) for value in values):
                raise ValueError("Non-finite CartesianInfo")
            result = {"tcp_base_m_rad": list(values[:6]),
                      "active_tcp_offset_m_rad": list(values[6:])}
        cursor += size
    return result


def read_tcp_info(host: str, *, timeout_s: float = 5.0) -> dict:
    """Receive a fresh controller frame on 30011. Never create RTDE Control."""
    if not math.isfinite(timeout_s) or not 0 < timeout_s <= 30:
        raise ValueError("timeout_s must be in (0, 30]")
    deadline = time.monotonic() + timeout_s
    with socket.create_connection((host, 30011), timeout=timeout_s) as stream:
        def receive_exact(count):
            data = bytearray()
            while len(data) < count:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out reading CartesianInfo")
                stream.settimeout(remaining)
                part = stream.recv(count - len(data))
                if not part:
                    raise ConnectionError("UR primary stream closed before a full frame")
                data.extend(part)
            return bytes(data)

        while True:
            header = receive_exact(5)
            size = struct.unpack_from("!I", header)[0]
            if not 5 <= size <= MAX_PACKET_BYTES:
                raise ValueError("Invalid primary packet length")
            result = parse_cartesian_info(header + receive_exact(size - 5))
            if result is not None:
                return {**result, "host": host, "port": 30011,
                        "source": "primary_readonly_cartesian_info",
                        "matches_historical_calibration": None}
