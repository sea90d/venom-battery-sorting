"""Minimal, strict Modbus-RTU transport for DH grippers."""
from __future__ import annotations

import threading
import time
import math
from typing import Callable

from .gripper import GripperError


def crc16(data: bytes) -> int:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0xA001 if crc & 1 else 0)
    return crc


class DhModbusClient:
    def __init__(self, port: str, baud: int = 115200, slave: int = 1,
                 timeout: float = 0.5, read_retries: int = 1,
                 serial_factory: Callable[..., object] | None = None) -> None:
        if not 1 <= slave <= 247: raise ValueError("slave 必须为 1..247")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive")
        if type(read_retries) is not int or not 0 <= read_retries <= 3:
            raise ValueError("read_retries must be 0..3")
        self.port_name, self.baud, self.slave = port, baud, slave
        self.timeout, self.read_retries = timeout, read_retries
        self._factory = serial_factory
        self._serial = None
        self._lock = threading.Lock()

    @property
    def connected(self) -> bool: return self._serial is not None

    def connect(self) -> None:
        if self._serial is not None: return
        factory = self._factory
        if factory is None:
            try:
                import serial
                factory = serial.Serial
            except ImportError as exc:
                raise GripperError("未安装 pyserial") from exc
        try:
            ser = factory(port=self.port_name, baudrate=self.baud, bytesize=8,
                          parity="N", stopbits=1, timeout=self.timeout,
                          write_timeout=self.timeout)
            ser.dtr = False; ser.rts = False
            if hasattr(ser, "open") and getattr(ser, "is_open", True) is False: ser.open()
            self._serial = ser
        except Exception as exc:
            raise GripperError(f"打开夹爪串口失败: {exc}") from exc

    def close(self) -> None:
        ser, self._serial = self._serial, None
        if ser is not None:
            try: ser.close()
            except Exception as exc: raise GripperError(f"关闭夹爪串口失败: {exc}") from exc

    def _frame(self, payload: bytes) -> bytes: return payload + crc16(payload).to_bytes(2, "little")

    def _read(self, register: int) -> int:
        return self._transaction(3, register, None)

    def read_register(self, register: int) -> int:
        first_error = None
        for attempt in range(self.read_retries + 1):
            try:
                return self._read(register)
            except GripperError as exc:
                first_error = first_error or exc
                if attempt == self.read_retries:
                    raise first_error
                time.sleep(min(0.05, self.timeout))
        raise AssertionError("unreachable")

    def write_register(self, register: int, value: int) -> None:
        self._transaction(6, register, value)

    def _transaction(self, function: int, register: int, value: int | None) -> int:
        if self._serial is None: raise GripperError("夹爪未连接")
        payload = bytes([self.slave, function, register >> 8, register & 255])
        payload += (bytes([value >> 8, value & 255]) if value is not None else b"\x00\x01")
        tx = self._frame(payload)
        with self._lock:
            ser = self._serial
            try: ser.reset_input_buffer()
            except Exception: pass
            deadline = time.monotonic() + self.timeout
            try:
                sent = ser.write(tx)
                if sent != len(tx): raise GripperError("夹爪串口部分写入")
                rx = bytearray()
                while len(rx) < 2 and time.monotonic() < deadline:
                    part = ser.read(2 - len(rx))
                    if part: rx.extend(part)
                if len(rx) != 2: raise GripperError("夹爪响应超时或不完整")
                need = 5 if rx[1] & 0x80 else (8 if function == 6 else 7)
                while len(rx) < need and time.monotonic() < deadline:
                    part = ser.read(need - len(rx))
                    if part: rx.extend(part)
                if len(rx) != need: raise GripperError("夹爪响应超时或不完整")
            except GripperError: raise
            except Exception as exc: raise GripperError(f"夹爪串口事务失败: {exc}") from exc
        raw = bytes(rx)
        if crc16(raw) != 0: raise GripperError("夹爪响应 CRC 错误")
        if raw[0] != self.slave: raise GripperError("夹爪响应从站地址错误")
        if raw[1] == (function | 0x80): raise GripperError(f"夹爪 Modbus 异常码 {raw[2]}")
        if raw[1] != function: raise GripperError("夹爪响应功能码错误")
        if function == 6:
            if raw[2:6] != payload[2:6]: raise GripperError("夹爪写入回显错误")
            return 0
        if raw[2] != 2: raise GripperError("夹爪响应字节数错误")
        return int.from_bytes(raw[3:5], "big")
