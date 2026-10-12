"""DH-compatible Modbus RTU feedback probe; DeepSeek draft, reviewed locally.

Only FC03 reads of 0x0200..0x0202 are permitted. No robot/actuator commands.
Successful feedback is NOT identification of the gripper model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time


REGISTERS = (0x0200, 0x0201, 0x0202)


def crc16(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0xA001 if crc & 1 else 0)
    return crc


def with_crc(data: bytes) -> bytes:
    return data + crc16(data).to_bytes(2, "little")


def request(slave: int, register: int) -> bytes:
    if not 1 <= slave <= 247 or register not in REGISTERS:
        raise ValueError("Only unicast FC03 feedback reads are allowed")
    return with_crc(bytes([slave, 3, register >> 8, register & 255, 0, 1]))


def parse_response(frame: bytes, slave: int) -> dict:
    if len(frame) < 5:
        return {"error": "incomplete" if frame else "no_response"}
    if crc16(frame) != 0:
        return {"error": "crc_mismatch"}
    if frame[0] != slave:
        return {"error": "wrong_slave", "received_slave": frame[0]}
    if frame[1] == 0x83 and len(frame) == 5:
        return {"error": "modbus_exception", "exception_code": frame[2]}
    if frame[1] != 3:
        return {"error": "unexpected_function"}
    if len(frame) != 7 or frame[2] != 2:
        return {"error": "invalid_length_or_byte_count"}
    return {"value": int.from_bytes(frame[3:5], "big")}


def read_exact(port, size: int, deadline: float) -> bytes:
    result = bytearray()
    while len(result) < size:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        port.timeout = min(0.5, remaining)
        part = port.read(size - len(result))
        if not part:
            break
        result.extend(part)
    return bytes(result)


def read_register(port, slave: int, register: int, deadline: float) -> dict:
    tx = request(slave, register)
    port.reset_input_buffer()  # Discard host receive buffer, not a device reset.
    if time.monotonic() >= deadline:
        return {"error": "deadline_expired", "tx": "", "rx": ""}
    port.write_timeout = min(0.5, deadline - time.monotonic())
    if port.write(tx) != len(tx):  # The ONLY serial send: whitelisted FC03.
        return {"error": "partial_send", "tx": tx.hex(" "), "rx": ""}
    header = read_exact(port, 2, deadline)
    tail = b""
    if len(header) == 2 and header[1] in (3, 0x83):
        tail = read_exact(port, 5 if header[1] == 3 else 3, deadline)
    rx = header + tail
    return {"tx": tx.hex(" "), "rx": rx.hex(" "), **parse_response(rx, slave)}


def self_test() -> None:
    # Independent published Modbus check vector, and DH feedback request.
    assert crc16(b"123456789") == 0x4B37
    assert request(1, 0x0200) == bytes.fromhex("01 03 02 00 00 01 85 B2")
    response = with_crc(bytes.fromhex("01 03 02 00 01"))
    assert parse_response(response, 1) == {"value": 1}
    assert parse_response(response, 2)["error"] == "wrong_slave"
    assert parse_response(response[:-1] + bytes([response[-1] ^ 1]), 1)["error"] == "crc_mismatch"
    assert parse_response(with_crc(bytes.fromhex("01 83 02")), 1)["exception_code"] == 2
    assert "error" in parse_response(with_crc(bytes.fromhex("01 03 04 00 01")), 1)
    for slave, reg in ((0, 0x0200), (248, 0x0200), (1, 0x0100)):
        try:
            request(slave, reg)
        except ValueError:
            pass
        else:
            raise AssertionError("Unsafe request accepted")

    class FakePort:
        def __init__(self, payload):
            self.payload = payload
            self.sent = []

        def reset_input_buffer(self):
            pass

        def write(self, data):
            self.sent.append(data)
            return len(data)

        def read(self, count):
            # Fragmented transport, one byte per read.
            part, self.payload = self.payload[:1], self.payload[1:]
            return part

    for payload, expected in ((response, None), (response[:4], "incomplete"), (b"", "no_response")):
        port = FakePort(payload)
        result = read_register(port, 1, 0x0200, time.monotonic() + 0.5)
        assert result.get("error") == expected, result
        assert result["rx"] == payload.hex(" ")
        assert port.sent == [request(1, 0x0200)]
    port = FakePort(response)
    result = read_register(port, 1, 0x0200, time.monotonic() - 1)
    assert result["error"] == "deadline_expired" and not port.sent
    print("PASS: CRC, slave, exception, whitelist, fragmented/partial/empty response, deadline")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("port", nargs="?")
    parser.add_argument("--baud", type=int, default=115200,
                        choices=[115200, 57600, 38400, 19200, 9600, 4800])
    parser.add_argument("--slave", type=int, default=1, choices=range(1, 248), metavar="1..247")
    parser.add_argument("--out", type=Path, help="New JSON report path; refuses overwrite")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.port:
        parser.error("port is required unless --self-test")
    if args.out and args.out.exists():
        parser.error("output already exists; choose a new report path")
    report = {"port": args.port, "baud": args.baud, "slave": args.slave,
              "read_only": True, "model_identified": False,
              "timestamp_unix": time.time(), "registers": {}}
    port = None
    failed = False
    try:
        import serial
        port = serial.Serial(port=None, baudrate=args.baud, bytesize=8,
                             parity="N", stopbits=1, timeout=0.5, write_timeout=0.5)
        port.dtr = False
        port.rts = False
        port.port = args.port
        port.open()
        deadline = time.monotonic() + 5.0
        for register in REGISTERS:
            attempts = []
            for _ in range(2):
                result = read_register(port, args.slave, register,
                                       min(deadline, time.monotonic() + 0.5))
                attempts.append(result)
                if "value" in result or time.monotonic() >= deadline:
                    break
                time.sleep(0.05)
            report["registers"][f"0x{register:04X}"] = attempts
            failed |= "value" not in attempts[-1]
            if time.monotonic() < deadline:
                time.sleep(0.01)
    except ImportError:
        report["error"] = "pyserial missing; install requirements-gripper.txt"
        failed = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        failed = True
    finally:
        if port is not None and port.is_open:
            port.close()
    text = json.dumps(report, indent=2, ensure_ascii=True)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as output:
            output.write(text + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
