from __future__ import annotations

from dataclasses import replace

import pytest

from battery_sorting.config import BatterySortingConfig
from battery_sorting.robot import (
    RealUR5,
    RobotMotionError,
    Waypoint,
)


class FakeControl:
    def __init__(self) -> None:
        self.stop_count = 0
        self.move_calls = []

    def isPoseWithinSafetyLimits(self, _pose: list[float]) -> bool:
        return True

    def moveL(self, *args) -> bool:
        self.move_calls.append(args)
        return True

    def stopL(self, _acceleration: float) -> None:
        self.stop_count += 1
        return None


class FakeReceiver:
    def __init__(self, target: list[float], force_after_baseline: float = 0.0) -> None:
        self.target = target
        self.force_after_baseline = force_after_baseline
        self.force_reads = 0

    def getActualTCPPose(self) -> list[float]:
        return list(self.target)

    def getActualTCPSpeed(self) -> list[float]:
        return [0.0] * 6

    def getActualTCPForce(self) -> list[float]:
        self.force_reads += 1
        force = 0.0 if self.force_reads == 1 else self.force_after_baseline
        return [force, 0.0, 0.0, 0.0, 0.0, 0.0]

    def getRobotMode(self) -> int:
        return 7

    def getSafetyMode(self) -> int:
        return 1

    def getRuntimeState(self) -> int:
        return 2

    def isEmergencyStopped(self) -> bool:
        return False

    def isProtectiveStopped(self) -> bool:
        return False


def wired_robot(config: BatterySortingConfig, force: float = 0.0) -> tuple[RealUR5, FakeControl]:
    target = config.robot.taught_poses["home"]
    assert target is not None
    robot = RealUR5(replace(config.robot, max_tcp_force_delta_n=5.0))
    control = FakeControl()
    robot._receiver = FakeReceiver(target.as_list(), force)  # type: ignore[attr-defined]
    robot._control = control  # type: ignore[attr-defined]
    return robot, control


def test_stop_l_void_return_is_treated_as_success(valid_config: BatterySortingConfig) -> None:
    robot, control = wired_robot(valid_config)
    assert robot.stop_l(0.2) is True
    assert control.stop_count == 1


def test_force_delta_aborts_async_move(valid_config: BatterySortingConfig) -> None:
    target = valid_config.robot.taught_poses["home"]
    assert target is not None
    robot, control = wired_robot(valid_config, force=10.0)
    with pytest.raises(RobotMotionError, match="TCP 力变化"):
        robot.move_l(Waypoint("force_test", target), 0.05, 0.10)
    assert control.stop_count == 1


def test_explicit_no_force_guard_does_not_read_tcp_force(valid_config: BatterySortingConfig) -> None:
    target = valid_config.robot.taught_poses["home"]
    assert target is not None
    robot, _control = wired_robot(valid_config, force=100.0)
    receiver = robot._receiver
    assert receiver is not None
    assert robot.move_l_without_tcp_force_guard(
        Waypoint("handeye_only", target), 0.01, 0.05
    )
    assert receiver.force_reads == 0


def test_blended_path_is_one_async_controller_command_without_force(valid_config: BatterySortingConfig) -> None:
    target = valid_config.robot.taught_poses['home']
    assert target is not None
    robot,control=wired_robot(valid_config,force=100.)
    first=replace(target,z=target.z+.01)
    path=[(Waypoint('fast',first),.03,.075,.004),(Waypoint('slow',target),.0075,.075,0.)]
    assert robot.move_l_path_without_tcp_force_guard(path)
    assert len(control.move_calls)==1
    command,asynchronous=control.move_calls[0]
    assert asynchronous is True
    assert command==[[*first.as_list(),.03,.075,.004],[*target.as_list(),.0075,.075,0.]]
    assert robot._receiver.force_reads==0


def test_blended_path_requires_zero_final_blend(valid_config: BatterySortingConfig) -> None:
    target=valid_config.robot.taught_poses['home'];assert target is not None
    robot,control=wired_robot(valid_config)
    with pytest.raises(Exception,match='末点 blend'):
        robot.move_l_path_without_tcp_force_guard([(Waypoint('bad',target),.03,.075,.004)])
    assert control.move_calls==[]


def test_live_feedback_failure_stops_continuous_path(valid_config):
    robot,control=wired_robot(valid_config)
    target=valid_config.robot.taught_poses['home']
    calls=[]
    def check():
        calls.append(True)
        if len(calls)==2:raise RuntimeError('held object dropped')
    with pytest.raises(RuntimeError,match='dropped'):
        robot.move_l_path_without_tcp_force_guard([(Waypoint('lift',target),.03,.1,0.)],check=check)
    assert len(control.move_calls)==1 and control.stop_count==1


def test_300mmps_profile_reaches_single_and_path_controller(valid_config):
    from tools.run_smooth_t0 import TRANSIT_SPEED,CONTACT_SPEED,ACCELERATION
    target=valid_config.robot.taught_poses['home']
    robot,control=wired_robot(valid_config)
    assert robot.move_l_without_tcp_force_guard(Waypoint('transfer',target),TRANSIT_SPEED,ACCELERATION)
    assert control.move_calls[-1][1]==pytest.approx(.30)
    assert robot.move_l_path_without_tcp_force_guard([
        (Waypoint('fast',replace(target,z=target.z+.01)),TRANSIT_SPEED,ACCELERATION,.004),
        (Waypoint('contact',target),CONTACT_SPEED,ACCELERATION,0.)])
    assert [point[6] for point in control.move_calls[-1][0]]==pytest.approx([.30,.04])


@pytest.mark.parametrize('speed,acceleration',[(.5,.1),(.3,1.)])
def test_standard_controller_keeps_original_command_limits(valid_config,speed,acceleration):
    robot,control=wired_robot(valid_config)
    target=valid_config.robot.taught_poses['home']
    with pytest.raises(Exception,match='必须在'):
        robot.move_l_without_tcp_force_guard(Waypoint('bad',target),speed,acceleration)
    with pytest.raises(Exception,match='必须在'):
        robot.move_l_path_without_tcp_force_guard([(Waypoint('bad',target),speed,acceleration,0.)])
    assert control.move_calls==[]


def test_t0_trial_controller_receives_fast_single_and_path_commands(valid_config):
    robot=RealUR5(valid_config.robot,motion_profile='t0_timing_trial')
    target=valid_config.robot.taught_poses['home']
    robot._receiver=FakeReceiver(target.as_list())
    robot._control=control=FakeControl()
    assert robot.move_l_without_tcp_force_guard(Waypoint('fast',target),.5,1.)
    assert robot.move_l_path_without_tcp_force_guard([
        (Waypoint('fast',replace(target,z=target.z+.01)),.5,1.,.004),
        (Waypoint('contact',target),.04,.1,0.)])
    assert control.move_calls[0][1:3]==(.5,1.)
    assert robot._receiver.force_reads==0
    with pytest.raises(Exception,match='必须在'):
        robot.move_l_without_tcp_force_guard(Waypoint('too_fast',target),.51,1.)
    assert len(control.move_calls)==2


def test_unknown_motion_profile_rejected(valid_config):
    with pytest.raises(Exception,match='Unknown'):
        RealUR5(valid_config.robot,motion_profile='unlimited')
