from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pytest

from battery_sorting.config import Pose
from battery_sorting.grasp_lift import (
    ExecutionAudit,
    GraspEvidence,
    GraspLiftConfig,
    GraspLiftError,
    build_plan,
    execute_command,
    execute_plan,
    load_evidence,
    validate_evidence,
)
from battery_sorting.gripper import MockGripper
from battery_sorting.robot import FakeRobot


ROOT = Path(__file__).resolve().parents[1]
START = Pose(
    0.13042634040602882,
    -0.2526456392162821,
    0.44785612393060525,
    -2.2448617460523574,
    -2.1864166279658708,
    0.00325035854435066,
)
OBJECT_XYZ = (0.1303638441948183, -0.2526353575548114, 0.025753426596039186)


def config(final_z: float | None = 0.069207) -> GraspLiftConfig:
    loaded = GraspLiftConfig.from_json(ROOT / "config" / "grasp_lift.json")
    return replace(loaded, final_grasp_z_m=final_z, grip_settle_s=0.0)


def evidence() -> GraspEvidence:
    return GraspEvidence("green", START, OBJECT_XYZ)


def reports() -> tuple[dict, dict, dict, dict]:
    calibration = {"passed": True}
    rgbd = {
        "valid": True,
        "detections": [{"color": "green", "valid": True}],
        "slots": [{"color": "green", "valid": True}],
    }
    motion_actual = [
        0.13034043414272192,
        -0.25262009740672225,
        0.4478826012855869,
        -0.04176147376013773,
        -3.13419394993263,
        -0.0064136916166079045,
    ]
    rotation_initial = [
        0.13043552062043504,
        -0.25262289149207307,
        0.447843818062873,
        -0.04137073609236435,
        -3.134215090748755,
        -0.00656629650602544,
    ]
    motion = {
        "passed": True,
        "contact_motion": False,
        "tcp_force_ignored_by_operator_request": True,
        "color": "green",
        "detected_battery_base_xyz_m": list(OBJECT_XYZ),
        "target_tcp_pose": [OBJECT_XYZ[0], OBJECT_XYZ[1], 0.447862658, *motion_actual[3:]],
        "actual_tcp_pose": motion_actual,
    }
    rotation = {
        "passed": True,
        "rotation_axis": "local_tcp_z",
        "rotation_deg": 90.0,
        "z_descent_m": 0.0,
        "tcp_force_ignored_by_operator_request": True,
        "initial_tcp_pose": rotation_initial,
        "target_tcp_pose": list(START.as_list()),
        "actual_tcp_pose": list(START.as_list()),
    }
    return calibration, rgbd, motion, rotation


def execute_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        grasp_config=str(ROOT / "config" / "grasp_lift.json"),
        calibration="must-not-be-read.json",
        rgbd_report="must-not-be-read.json",
        motion_report="must-not-be-read.json",
        rotation_report="must-not-be-read.json",
        output=str(tmp_path / "audit.json"),
        execute=True,
        tcp_payload_confirmed=True,
        emergency_stop_ready=True,
        ignore_tcp_force=True,
    )


def test_null_final_z_blocks_before_hardware_factories(tmp_path: Path) -> None:
    calls: list[str] = []

    null_config = tmp_path / "null_grasp.json"
    raw = json.loads((ROOT / "config" / "grasp_lift.json").read_text(encoding="utf-8"))
    raw["final_grasp_z_m"] = None
    null_config.write_text(json.dumps(raw), encoding="utf-8")

    def robot_factory(_):
        calls.append("robot")
        raise AssertionError("robot factory must not run")

    def gripper_factory(_):
        calls.append("gripper")
        raise AssertionError("gripper factory must not run")

    args = execute_args(tmp_path)
    args.grasp_config = str(null_config)
    result = execute_command(
        args,
        robot_factory=robot_factory,
        gripper_factory=gripper_factory,
    )
    assert result == 2
    assert calls == []
    audit = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert "final_grasp_z_m" in audit["error"]
    assert "null" in audit["error"]


@pytest.mark.parametrize("final_z", [0.039, 0.121])
def test_final_z_outside_site_range_is_rejected(final_z: float) -> None:
    with pytest.raises(GraspLiftError, match="现场夹持高度范围"):
        build_plan(config(final_z), evidence())


def test_total_descent_limit_is_enforced() -> None:
    start = replace(START, z=0.55)
    with pytest.raises(GraspLiftError, match="总下降"):
        build_plan(config(0.12), GraspEvidence("green", start, OBJECT_XYZ))


def test_real_reports_form_a_valid_evidence_chain() -> None:
    loaded = load_evidence(
        ROOT / "output" / "handeye" / "session_20260910" / "handeye_result.json",
        ROOT / "output" / "hover_validation" / "capture_20260910_retry" / "report.json",
        ROOT / "output" / "hover_validation" / "green_hover_result.json",
        ROOT / "output" / "hover_validation" / "green_pregrasp_rotation.json",
        config(),
    )
    assert loaded.color == "green"
    assert loaded.start_pose.as_list() == pytest.approx(START.as_list())


def test_report_chain_discontinuity_is_rejected() -> None:
    calibration, rgbd, motion, rotation = reports()
    rotation["initial_tcp_pose"][0] += 0.01
    with pytest.raises(GraspLiftError, match="不连续"):
        validate_evidence(calibration, rgbd, motion, rotation, config=config())


def test_plan_changes_only_z_and_forces_final_ten_mm_slow() -> None:
    cfg = config()
    plan = build_plan(cfg, evidence())
    assert plan.total_descent_m == pytest.approx(START.z - 0.069207)
    assert plan.descent[-1].pose.z == pytest.approx(0.069207)
    assert plan.descent[-1].speed_mps == pytest.approx(0.005)
    assert plan.descent[-1].slow is True
    assert plan.descent[-2].pose.z - plan.descent[-1].pose.z == pytest.approx(0.01)
    for step in (*plan.descent, *plan.lift):
        assert step.pose.as_list()[:2] == pytest.approx(START.as_list()[:2])
        assert step.pose.as_list()[3:] == pytest.approx(START.as_list()[3:])
    assert plan.lift[-1].pose == START


def test_success_order_is_open_descend_close_feedback_lift() -> None:
    cfg = config()
    plan = build_plan(cfg, evidence())
    robot = FakeRobot(START)
    gripper = MockGripper(object_after_close=True)
    audit = execute_plan(plan, cfg, robot, gripper, sleep=lambda _: None)
    actions = [event["action"] for event in audit.events]
    close_index = actions.index("gripper_close")
    feedback_index = actions.index("object_feedback")
    labels = [event.get("label", "") for event in audit.events]
    assert actions[0] == "gripper_open_high"
    assert all(label.startswith("descend_") for label in labels[1:close_index])
    assert feedback_index == close_index + 1
    assert all(label.startswith("lift_") for label in labels[feedback_index + 1 :])
    assert gripper.actions == ["open", "close"]
    assert audit.passed is True


def test_empty_grasp_stops_without_lifting_or_reopening() -> None:
    cfg = config()
    plan = build_plan(cfg, evidence())
    robot = FakeRobot(START)
    gripper = MockGripper(object_after_close=False)
    audit = ExecutionAudit(color="green")
    with pytest.raises(GraspLiftError, match="停止原地"):
        execute_plan(plan, cfg, robot, gripper, sleep=lambda _: None, audit=audit)
    assert robot.snapshot().tcp_pose == pytest.approx(plan.target_pose.as_list())
    assert not any(record.label.startswith("lift_") for record in robot.records)
    assert gripper.actions == ["open", "close"]


class CloseFailGripper(MockGripper):
    def close(self) -> bool:
        self.actions.append("close")
        raise RuntimeError("synthetic close failure")


def test_close_exception_stops_without_recovery_or_reopening() -> None:
    cfg = config()
    plan = build_plan(cfg, evidence())
    robot = FakeRobot(START)
    gripper = CloseFailGripper(object_after_close=True)
    audit = ExecutionAudit(color="green")
    with pytest.raises(RuntimeError, match="synthetic close failure"):
        execute_plan(plan, cfg, robot, gripper, sleep=lambda _: None, audit=audit)
    assert robot.snapshot().tcp_pose == pytest.approx(plan.target_pose.as_list())
    assert not any(event["action"] == "recovery_lift" for event in audit.events)
    assert gripper.actions == ["open", "close"]


class FakeRealGripper(MockGripper):
    @property
    def is_mock(self) -> bool:
        return False

    def initialize(self, recalibrate: bool = False) -> bool:
        self.actions.append("initialize_recalibrate" if recalibrate else "initialize")
        return True


class DriftOnControlRobot(FakeRobot):
    def connect_control(self) -> None:
        super().connect_control()
        self._pose = self._pose.with_z(self._pose.z - 0.01)


def test_control_connection_pose_drift_blocks_before_open(tmp_path: Path) -> None:
    raw = json.loads((ROOT / "config" / "grasp_lift.json").read_text(encoding="utf-8"))
    raw["final_grasp_z_m"] = 0.069207
    config_path = tmp_path / "ready.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    args = execute_args(tmp_path)
    args.grasp_config = str(config_path)
    args.calibration = str(ROOT / "output" / "handeye" / "session_20260910" / "handeye_result.json")
    args.rgbd_report = str(ROOT / "output" / "hover_validation" / "capture_20260910_retry" / "report.json")
    args.motion_report = str(ROOT / "output" / "hover_validation" / "green_hover_result.json")
    args.rotation_report = str(ROOT / "output" / "hover_validation" / "green_pregrasp_rotation.json")
    robot = DriftOnControlRobot(START)
    gripper = FakeRealGripper(object_after_close=True)

    result = execute_command(
        args,
        robot_factory=lambda _: robot,
        gripper_factory=lambda _: gripper,
        input_fn=lambda prompt: prompt.splitlines()[1],
    )

    assert result == 2
    assert robot.records == []
    assert gripper.actions == []
    audit = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert "建立控制连接后" in audit["error"]


def test_verified_high_pose_initializes_before_open_and_descent(tmp_path: Path) -> None:
    raw = json.loads((ROOT / "config" / "grasp_lift.json").read_text(encoding="utf-8"))
    raw["grip_settle_s"] = 0.0
    config_path = tmp_path / "ready.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    args = execute_args(tmp_path)
    args.grasp_config = str(config_path)
    args.calibration = str(ROOT / "output" / "handeye" / "session_20260910" / "handeye_result.json")
    args.rgbd_report = str(ROOT / "output" / "hover_validation" / "capture_20260910_retry" / "report.json")
    args.motion_report = str(ROOT / "output" / "hover_validation" / "green_hover_result.json")
    args.rotation_report = str(ROOT / "output" / "hover_validation" / "green_pregrasp_rotation.json")
    robot = FakeRobot(START)
    gripper = FakeRealGripper(object_after_close=True)

    result = execute_command(
        args,
        robot_factory=lambda _: robot,
        gripper_factory=lambda _: gripper,
        input_fn=lambda prompt: prompt.splitlines()[1],
    )

    assert result == 0
    assert gripper.actions == ["initialize", "open", "close"]
    audit = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    actions = [event["action"] for event in audit["events"]]
    assert actions.index("gripper_initialize_home") < actions.index("gripper_open_high")
