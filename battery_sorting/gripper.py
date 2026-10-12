"""夹爪接口及 DH Robotics AG-105-145 Modbus 驱动。"""

from __future__ import annotations

from abc import ABC, abstractmethod
import time


class GripperError(RuntimeError):
    """夹爪命令失败。"""


class GripperBase(ABC):
    """连接只管通信；connect 不得回零/开合，stop/disconnect 不得释放工件。"""

    @property
    @abstractmethod
    def is_configured(self) -> bool: ...

    @property
    @abstractmethod
    def is_mock(self) -> bool: ...

    @property
    @abstractmethod
    def has_object_feedback(self) -> bool: ...

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    def initialize(self, recalibrate: bool = False) -> bool:
        """Initialize at a verified safe robot pose before any gripper motion."""
        raise GripperError("当前夹爪驱动不支持初始化")

    @abstractmethod
    def open(self) -> bool: ...

    @abstractmethod
    def close(self) -> bool: ...

    @abstractmethod
    def stop(self) -> bool: ...

    def has_object(self) -> bool | None:
        return None

    def position(self) -> int:
        raise GripperError("夹爪未提供实际位置反馈")


class MockGripper(GripperBase):
    """仅供离线规划和测试，绝不能通过真机门禁。"""

    def __init__(self, object_after_close: bool | None = None) -> None:
        self.actions: list[str] = []
        self._closed = False
        self._object_after_close = object_after_close
        self.connected = False

    @property
    def is_configured(self) -> bool:
        return True

    @property
    def is_mock(self) -> bool:
        return True

    @property
    def has_object_feedback(self) -> bool:
        return self._object_after_close is not None

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def initialize(self, recalibrate: bool = False) -> bool:
        self.actions.append("initialize_recalibrate" if recalibrate else "initialize")
        return True

    def open(self) -> bool:
        self.actions.append("open")
        self._closed = False
        return True

    def close(self) -> bool:
        self.actions.append("close")
        self._closed = True
        return True

    def stop(self) -> bool:
        self.actions.append("stop")
        return True

    def has_object(self) -> bool | None:
        return self._object_after_close if self._closed else False


class NotConfiguredGripper(GripperBase):
    @property
    def is_configured(self) -> bool:
        return False

    @property
    def is_mock(self) -> bool:
        return False

    @property
    def has_object_feedback(self) -> bool:
        return False

    def connect(self) -> None:
        self._fail()

    def disconnect(self) -> None:
        return None

    def _fail(self) -> bool:
        raise GripperError("尚未配置真实夹爪，命令被拒绝")

    def initialize(self, recalibrate: bool = False) -> bool:
        return self._fail()

    def open(self) -> bool:
        return self._fail()

    def close(self) -> bool:
        return self._fail()

    def stop(self) -> bool:
        return self._fail()


class AG105145Gripper(GripperBase):
    """AG-105-145 标准 Modbus RTU；connect 只读，动作必须显式调用。"""
    def __init__(self, config) -> None:
        self.config = config
        self.client = None
        self._initialized = False
        self.last_grip_state: int | None = None

    @property
    def is_configured(self) -> bool: return True
    @property
    def is_mock(self) -> bool: return False
    @property
    def has_object_feedback(self) -> bool: return True

    def connect(self) -> None:
        from .dh_modbus import DhModbusClient
        if self.client is None:
            self.client = DhModbusClient(self.config.port, self.config.baud,
                self.config.slave_id, self.config.serial_timeout,
                self.config.read_retries)
        self.client.connect()
        self._initialized = self.client.read_register(0x0200) == 1
        self.last_grip_state = self.client.read_register(0x0201)
        self.client.read_register(0x0202)

    def disconnect(self) -> None:
        if self.client is not None: self.client.close()
        self._initialized = False

    def initialize(self, recalibrate: bool = False) -> bool:
        if self.client is None or not self.client.connected:
            raise GripperError("夹爪未连接")
        if not recalibrate and self.client.read_register(0x0200) == 1:
            self._initialized = True
            return True
        self.client.write_register(0x0100, 0xA5 if recalibrate else 1)
        deadline = time.monotonic() + self.config.command_timeout
        while time.monotonic() < deadline:
            if self.client.read_register(0x0200) == 1:
                self._initialized = True; return True
            time.sleep(self.config.poll_interval)
        raise GripperError("夹爪初始化超时")

    def _require_connected(self) -> None:
        if self.client is None or not self.client.connected: raise GripperError("夹爪未连接")
        if not self._initialized: raise GripperError("夹爪尚未初始化")

    def _move(self, position: int, opening: bool) -> bool:
        self._require_connected()
        self.client.write_register(0x0104, self.config.speed_percent)
        self.client.write_register(0x0101, self.config.force_percent)
        self.client.write_register(0x0103, position)
        deadline = time.monotonic() + self.config.command_timeout
        consecutive_grips = 0
        while time.monotonic() < deadline:
            state = self.client.read_register(0x0201); self.last_grip_state = state
            current = self.client.read_register(0x0202)
            if state == 3: raise GripperError("夹爪报告工件掉落")
            if not opening and state == 2:
                consecutive_grips += 1
                if consecutive_grips >= 2:
                    return True
            else:
                consecutive_grips = 0
            if state == 1:
                if abs(current - position) <= self.config.position_tolerance: return True
            if state not in (0, 1, 2): raise GripperError(f"未知夹爪状态 {state}")
            time.sleep(self.config.poll_interval)
        raise GripperError("夹爪动作超时")

    def open(self) -> bool: return self._move(self.config.open_position, True)
    def close(self) -> bool: return self._move(self.config.close_position, False)

    def reset_empty_open(self, *, empty_and_clear_confirmed: bool = False) -> dict:
        """Explicit startup recovery only; never call as an error recovery.

        The caller confirms no object and clearance for the fingers to open.
        State 3 may persist briefly after the opening write, but only repeated
        state 1 AND measured opening can complete this operation. No UR motion.
        """
        if empty_and_clear_confirmed is not True:
            raise GripperError('Confirm empty gripper and opening clearance before reset')
        if self.client is None or not self.client.connected:
            raise GripperError('夹爪未连接')
        initial = self.client.read_register(0x0201)
        self.last_grip_state = initial
        if initial not in (1, 3):
            raise GripperError(f'Reset requires stationary empty gripper; state={initial}')
        if not self.initialize(recalibrate=False):
            raise GripperError('Startup initialization failed')
        self._require_connected()
        # One opening command; no repeated initialization or write retries.
        self.client.write_register(0x0104, self.config.speed_percent)
        self.client.write_register(0x0101, self.config.force_percent)
        self.client.write_register(0x0103, self.config.open_position)
        deadline = time.monotonic() + self.config.command_timeout
        stable = 0
        previous = None
        while time.monotonic() < deadline:
            state = self.client.read_register(0x0201)
            self.last_grip_state = state
            position = self.position()
            if state not in (0, 1, 3):
                raise GripperError(f'Unexpected object/state during empty reset: {state}')
            at_target = abs(position-self.config.open_position) <= self.config.position_tolerance
            if state == 1 and at_target:
                stable = stable+1 if previous is not None and abs(position-previous)<=self.config.position_tolerance else 1
            else:
                stable = 0
            previous = position
            if stable >= 3:
                return {'initial_state': initial, 'final_state': state,
                        'position': position, 'open_position': self.config.open_position,
                        'stable_samples': stable}
            time.sleep(self.config.poll_interval)
        raise GripperError('Empty reset timed out: opening/state not confirmed; do not start robot')

    def has_object(self) -> bool | None:
        self._require_connected()
        state = self.client.read_register(0x0201); self.last_grip_state = state
        if state == 2: return True
        if state in (1, 3): return False
        if state == 0: return None
        raise GripperError(f"未知夹爪状态 {state}")

    def position(self) -> int:
        self._require_connected()
        value = self.client.read_register(0x0202)
        if not 0 <= value <= 1000:
            raise GripperError("夹爪实际位置超出 0..1000")
        return value

    def stop(self) -> bool:
        self._require_connected()
        state = self.client.read_register(0x0201); self.last_grip_state = state
        if state == 0: raise GripperError("夹爪正在运动，协议无独立停止命令")
        if state in (1, 2, 3): return True
        raise GripperError(f"未知夹爪状态 {state}")


def create_gripper(driver) -> GripperBase:
    if hasattr(driver, "driver"):
        if driver.driver == "ag105145_modbus": return AG105145Gripper(driver)
        driver = driver.driver
    if driver == "mock":
        return MockGripper()
    if driver == "not_configured":
        return NotConfiguredGripper()
    raise GripperError(f"未知夹爪驱动: {driver}")
