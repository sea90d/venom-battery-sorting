from __future__ import annotations

from collections import deque

import pytest

from battery_sorting.dh_modbus import DhModbusClient, crc16
from battery_sorting.gripper import AG105145Gripper, GripperError
from battery_sorting.config import GripperConfig


def frame(payload: bytes) -> bytes:
    return payload + crc16(payload).to_bytes(2, "little")


class FakeSerial:
    def __init__(self, responses: list[bytes] | None = None, *, fragmented: bool = False):
        self.responses = deque(responses or [])
        self.fragmented = fragmented
        self.sent: list[bytes] = []
        self.rx = bytearray()
        self.is_open = True
        self.dtr = True
        self.rts = True

    def reset_input_buffer(self):
        self.rx.clear()

    def write(self, data: bytes) -> int:
        self.sent.append(bytes(data))
        if self.responses:
            self.rx.extend(self.responses.popleft())
        return len(data)

    def read(self, count: int) -> bytes:
        if not self.rx:
            return b""
        take = 1 if self.fragmented else min(count, len(self.rx))
        result = bytes(self.rx[:take])
        del self.rx[:take]
        return result

    def close(self):
        self.is_open = False


def factory_for(port: FakeSerial):
    def factory(**_kwargs):
        return port
    return factory


def read_response(value: int) -> bytes:
    return frame(bytes.fromhex("01 03 02") + value.to_bytes(2, "big"))


def write_response(register: int, value: int) -> bytes:
    return frame(bytes([1, 6, register >> 8, register & 255, value >> 8, value & 255]))


def test_crc_and_fragmented_fc03():
    assert crc16(b"123456789") == 0x4B37
    port = FakeSerial([read_response(123)], fragmented=True)
    client = DhModbusClient("FAKE", serial_factory=factory_for(port), timeout=0.1)
    client.connect()
    assert client.read_register(0x0202) == 123
    assert port.sent[0] == bytes.fromhex("01 03 02 02 00 01 24 72")


def test_connect_is_read_only_and_disconnect_is_idempotent():
    port = FakeSerial([read_response(0), read_response(1), read_response(1000)])
    client = DhModbusClient("FAKE", serial_factory=factory_for(port))
    client.connect()
    assert port.sent == []
    client.close()
    client.close()
    assert not port.is_open


@pytest.mark.parametrize(
    "response, message",
    [
        (b"", "超时"),
        (b"\x01\x03\x02\x00", "不完整"),
        (bytes.fromhex("02 03 02 00 01"), "从站"),
        (bytes.fromhex("01 04 02 00 01"), "功能码"),
        (bytes.fromhex("01 03 01 00 01"), "字节数"),
        (frame(bytes.fromhex("01 83 02")), "异常码"),
    ],
)
def test_bad_read_response_fails_closed(response: bytes, message: str):
    if len(response) >= 5 and response[1] != 0x83:
        response = frame(response)
    port = FakeSerial([response])
    client = DhModbusClient("FAKE", serial_factory=factory_for(port), timeout=0.01)
    client.connect()
    with pytest.raises(GripperError, match=message):
        client.read_register(0x0200)


def test_fc06_echo_and_partial_write():
    port = FakeSerial([write_response(0x0100, 1)])
    client = DhModbusClient("FAKE", serial_factory=factory_for(port))
    client.connect()
    client.write_register(0x0100, 1)
    assert port.sent[-1] == frame(bytes.fromhex("01 06 01 00 00 01"))

    class Partial(FakeSerial):
        def write(self, data):
            self.sent.append(bytes(data))
            return len(data) - 1

    bad = Partial()
    client = DhModbusClient("FAKE", serial_factory=factory_for(bad))
    client.connect()
    with pytest.raises(GripperError, match="部分写入"):
        client.write_register(0x0100, 1)


def make_gripper(responses: list[bytes]) -> tuple[AG105145Gripper, FakeSerial]:
    port = FakeSerial(responses)
    config = GripperConfig(driver="ag105145_modbus", command_timeout=0.1, serial_timeout=0.01)
    gripper = AG105145Gripper(config)
    gripper.client = DhModbusClient(config.port, config.baud, config.slave_id,
                                    config.serial_timeout, config.read_retries,
                                    serial_factory=factory_for(port))
    return gripper, port


def test_initialize_open_close_and_object_feedback():
    # connect reads init/grip/position; initialize reads init; open and close poll state/position.
    responses = [
        read_response(0), read_response(1), read_response(1000),
        read_response(0), write_response(0x0100, 1), read_response(1),
        write_response(0x0104, 50), write_response(0x0101, 60), write_response(0x0103, 1000),
        read_response(1), read_response(1000),
        write_response(0x0104, 50), write_response(0x0101, 60), write_response(0x0103, 0),
        read_response(2), read_response(239), read_response(2), read_response(239),
        read_response(2),
    ]
    gripper, port = make_gripper(responses)
    gripper.connect()
    assert gripper.initialize() is True
    assert gripper.open() is True
    assert gripper.close() is True
    assert gripper.has_object() is True
    assert [packet[1] for packet in port.sent] == [3, 3, 3, 3, 6, 3, 6, 6, 6, 3, 3, 6, 6, 6, 3, 3, 3, 3, 3]


def test_retry_reads_but_never_retry_writes():
    port = FakeSerial([b"", read_response(239), b"", write_response(0x0103, 0)])
    client = DhModbusClient("FAKE", timeout=0.001, read_retries=1, serial_factory=factory_for(port))
    client.connect()
    assert client.read_register(0x0202) == 239
    with pytest.raises(GripperError):
        client.write_register(0x0103, 0)
    assert [p[1] for p in port.sent] == [3, 3, 6]


def test_initialized_gripper_is_not_reinitialized():
    gripper, port = make_gripper([read_response(1), read_response(1), read_response(1000), read_response(1)])
    gripper.connect()
    assert gripper.initialize()
    assert all(p[1] == 3 for p in port.sent)


def test_explicit_recalibration_and_stop_never_writes():
    responses = [
        read_response(1), read_response(1), read_response(0),
        write_response(0x0100, 0xA5), read_response(1), read_response(1),
    ]
    gripper, port = make_gripper(responses)
    gripper.connect()
    before = len(port.sent)
    assert gripper.initialize(recalibrate=True) is True
    after_init = len(port.sent)
    assert gripper.stop() is True
    assert len([packet for packet in port.sent if packet[1] == 6]) == 1
    assert port.sent[before][4:6] == bytes.fromhex("00 a5")


def test_dropped_and_moving_feedback():
    gripper, port = make_gripper([
        read_response(1), read_response(1), read_response(0),
        read_response(3), read_response(0),
    ])
    gripper.connect()
    assert gripper.has_object() is False
    assert gripper.last_grip_state == 3
    with pytest.raises(GripperError, match="正在运动"):
        gripper.stop()
