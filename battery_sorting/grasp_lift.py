"""Continue a validated eye-in-hand hover with a vertical grasp and lift."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from .config import GripperConfig, Pose, RobotConfig, WorkspaceBounds
from .gripper import GripperBase, create_gripper
from .robot import (
    RealUR5,
    RobotBase,
    Waypoint,
    assert_pose_close,
    assert_pose_unchanged,
    rotation_error,
    translation_error,
)


class GraspLiftError(RuntimeError):
    """The grasp configuration, evidence, state, or result is invalid."""


@dataclass(frozen=True)
class GraspLiftConfig:
    robot_host: str
    dashboard_port: int
    rtde_port: int
    gripper: GripperConfig
    final_grasp_z_m: float | None
    allowed_final_z_min_m: float
    allowed_final_z_max_m: float
    max_total_descent_m: float
    max_z_segment_m: float
    final_slow_distance_m: float
    approach_speed_mps: float
    contact_speed_mps: float
    lift_speed_mps: float
    acceleration_mps2: float
    move_timeout_s: float
    grip_settle_s: float
    start_translation_tolerance_m: float
    start_rotation_tolerance_deg: float
    report_chain_translation_tolerance_m: float
    report_chain_rotation_tolerance_deg: float
    confirm_translation_tolerance_m: float
    confirm_rotation_tolerance_deg: float

    @classmethod
    def from_json(cls, path: str | Path) -> "GraspLiftConfig":
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise GraspLiftError(f"无法读取抓取配置 {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise GraspLiftError("抓取配置根节点必须是 JSON 对象")
        expected = set(cls.__dataclass_fields__) - {"gripper"}
        expected.add("gripper")
        _require_exact_keys(raw, expected, "抓取配置")
        gripper_raw = raw["gripper"]
        if not isinstance(gripper_raw, dict):
            raise GraspLiftError("gripper 必须是 JSON 对象")
        _require_exact_keys(gripper_raw, set(GripperConfig.__dataclass_fields__), "gripper")
        final_z = raw["final_grasp_z_m"]
        if final_z is not None:
            final_z = _number(final_z, "final_grasp_z_m")
        config = cls(
            robot_host=_string(raw["robot_host"], "robot_host"),
            dashboard_port=_integer(raw["dashboard_port"], "dashboard_port"),
            rtde_port=_integer(raw["rtde_port"], "rtde_port"),
            gripper=GripperConfig(
                driver=_string(gripper_raw["driver"], "gripper.driver"),
                port=_string(gripper_raw["port"], "gripper.port"),
                baud=_integer(gripper_raw["baud"], "gripper.baud"),
                slave_id=_integer(gripper_raw["slave_id"], "gripper.slave_id"),
                force_percent=_integer(gripper_raw["force_percent"], "gripper.force_percent"),
                speed_percent=_integer(gripper_raw["speed_percent"], "gripper.speed_percent"),
                open_position=_integer(gripper_raw["open_position"], "gripper.open_position"),
                close_position=_integer(gripper_raw["close_position"], "gripper.close_position"),
                position_tolerance=_integer(gripper_raw["position_tolerance"], "gripper.position_tolerance"),
                serial_timeout=_number(gripper_raw["serial_timeout"], "gripper.serial_timeout"),
                command_timeout=_number(gripper_raw["command_timeout"], "gripper.command_timeout"),
                poll_interval=_number(gripper_raw["poll_interval"], "gripper.poll_interval"),
                read_retries=_integer(gripper_raw["read_retries"], "gripper.read_retries"),
            ),
            final_grasp_z_m=final_z,
            **{
                name: _number(raw[name], name)
                for name in expected
                if name not in {"robot_host", "dashboard_port", "rtde_port", "gripper", "final_grasp_z_m"}
            },
        )
        config.require_valid(allow_missing_final_z=True)
        return config

    def require_valid(self, *, allow_missing_final_z: bool) -> None:
        errors: list[str] = []
        if not (1 <= self.dashboard_port <= 65535 and 1 <= self.rtde_port <= 65535):
            errors.append("Dashboard/RTDE 端口必须在 1..65535")
        if self.gripper.driver != "ag105145_modbus":
            errors.append("真机抓取只允许 ag105145_modbus 夹爪")
        if not 1 <= self.gripper.slave_id <= 247:
            errors.append("gripper.slave_id 必须在 1..247")
        if not 0 <= self.gripper.force_percent <= 100:
            errors.append("gripper.force_percent 必须在 0..100")
        if not 0 <= self.gripper.speed_percent <= 100:
            errors.append("gripper.speed_percent 必须在 0..100")
        if not self.allowed_final_z_min_m < self.allowed_final_z_max_m:
            errors.append("允许的最终 Z 下限必须小于上限")
        if not 0.0 < self.max_total_descent_m <= 0.50:
            errors.append("max_total_descent_m 必须在 (0,0.50] m")
        if not 0.0 < self.max_z_segment_m <= 0.05:
            errors.append("max_z_segment_m 必须在 (0,0.05] m")
        if not 0.0 < self.final_slow_distance_m <= self.max_z_segment_m:
            errors.append("final_slow_distance_m 必须在 (0,max_z_segment_m]")
        if not 0.0 < self.contact_speed_mps <= self.approach_speed_mps <= 0.05:
            errors.append("速度必须满足 0 < contact <= approach <= 0.05 m/s")
        if not 0.0 < self.lift_speed_mps <= 0.05:
            errors.append("lift_speed_mps 必须在 (0,0.05] m/s")
        if not 0.0 < self.acceleration_mps2 <= 0.25:
            errors.append("acceleration_mps2 必须在 (0,0.25] m/s^2")
        if not 1.0 <= self.move_timeout_s <= 120.0:
            errors.append("move_timeout_s 必须在 1..120 s")
        if not 0.0 <= self.grip_settle_s <= 10.0:
            errors.append("grip_settle_s 必须在 0..10 s")
        for name, value, maximum in (
            ("start_translation_tolerance_m", self.start_translation_tolerance_m, 0.005),
            ("report_chain_translation_tolerance_m", self.report_chain_translation_tolerance_m, 0.005),
            ("confirm_translation_tolerance_m", self.confirm_translation_tolerance_m, 0.001),
            ("start_rotation_tolerance_deg", self.start_rotation_tolerance_deg, 2.0),
            ("report_chain_rotation_tolerance_deg", self.report_chain_rotation_tolerance_deg, 2.0),
            ("confirm_rotation_tolerance_deg", self.confirm_rotation_tolerance_deg, 0.2),
        ):
            if not 0.0 < value <= maximum:
                errors.append(f"{name} 必须在 (0,{maximum}] 范围")
        if self.final_grasp_z_m is None:
            if not allow_missing_final_z:
                errors.append("final_grasp_z_m 仍为 null，禁止连接机器人或夹爪")
        elif not self.allowed_final_z_min_m <= self.final_grasp_z_m <= self.allowed_final_z_max_m:
            errors.append("final_grasp_z_m 超出允许的现场夹持高度范围")
        if errors:
            raise GraspLiftError("抓取配置无效:\n  - " + "\n  - ".join(errors))

    def robot_config(self, start: Pose) -> RobotConfig:
        return RobotConfig(
            host=self.robot_host,
            dashboard_port=self.dashboard_port,
            rtde_port=self.rtde_port,
            tcp_calibrated=True,
            payload_calibrated=True,
            base_z_up_confirmed=True,
            workspace_bounds=WorkspaceBounds(
                x_min=start.x - 0.005,
                x_max=start.x + 0.005,
                y_min=start.y - 0.005,
                y_max=start.y + 0.005,
                z_min=self.allowed_final_z_min_m,
                z_max=start.z + 0.005,
            ),
            max_vertical_segment_m=self.max_z_segment_m,
            approach_speed=self.approach_speed_mps,
            contact_speed=self.contact_speed_mps,
            move_acceleration=self.acceleration_mps2,
            move_timeout=self.move_timeout_s,
        )


@dataclass(frozen=True)
class GraspEvidence:
    color: str
    start_pose: Pose
    object_base_xyz_m: tuple[float, float, float]


@dataclass(frozen=True)
class VerticalStep:
    label: str
    pose: Pose
    speed_mps: float
    slow: bool = False


@dataclass(frozen=True)
class GraspLiftPlan:
    color: str
    start_pose: Pose
    target_pose: Pose
    descent: tuple[VerticalStep, ...]
    lift: tuple[VerticalStep, ...]

    @property
    def total_descent_m(self) -> float:
        return self.start_pose.z - self.target_pose.z


@dataclass
class ExecutionAudit:
    color: str = ""
    passed: bool = False
    grip_detected: bool | None = None
    error: str = ""
    started_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    events: list[dict[str, Any]] = field(default_factory=list)

    def event(self, action: str, **details: Any) -> None:
        self.events.append({"action": action, **details})


def load_evidence(
    calibration_path: str | Path,
    rgbd_path: str | Path,
    motion_path: str | Path,
    rotation_path: str | Path,
    config: GraspLiftConfig,
) -> GraspEvidence:
    reports: list[dict[str, Any]] = []
    for label, path in (
        ("手眼标定", calibration_path),
        ("RGB-D", rgbd_path),
        ("悬空移动", motion_path),
        ("预抓取旋转", rotation_path),
    ):
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise GraspLiftError(f"无法读取{label}报告 {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise GraspLiftError(f"{label}报告根节点必须是 JSON 对象")
        reports.append(value)
    return validate_evidence(*reports, config=config)


def validate_evidence(
    calibration: Mapping[str, Any],
    rgbd: Mapping[str, Any],
    motion: Mapping[str, Any],
    rotation: Mapping[str, Any],
    *,
    config: GraspLiftConfig,
) -> GraspEvidence:
    if calibration.get("passed") is not True:
        raise GraspLiftError("手眼标定报告未通过")
    if rgbd.get("valid") is not True:
        raise GraspLiftError("RGB-D 报告未通过")
    if motion.get("passed") is not True or motion.get("contact_motion") is not False:
        raise GraspLiftError("悬空移动报告未通过或包含接触运动")
    if motion.get("tcp_force_ignored_by_operator_request") is not True:
        raise GraspLiftError("悬空移动报告与 TCP 力忽略策略不一致")
    if rotation.get("passed") is not True:
        raise GraspLiftError("预抓取旋转报告未通过")
    if rotation.get("rotation_axis") != "local_tcp_z":
        raise GraspLiftError("预抓取旋转不是局部 TCP Z 轴")
    if not 89.0 <= _number(rotation.get("rotation_deg"), "rotation.rotation_deg") <= 91.0:
        raise GraspLiftError("预抓取旋转角不是 90°（允许 89..91°）")
    if abs(_number(rotation.get("z_descent_m"), "rotation.z_descent_m")) > 0.0002:
        raise GraspLiftError("预抓取旋转报告显示发生了 Z 下降")
    if rotation.get("tcp_force_ignored_by_operator_request") is not True:
        raise GraspLiftError("旋转报告与 TCP 力忽略策略不一致")

    color = _string(motion.get("color"), "motion.color")
    if len(_matching_valid_colors(rgbd.get("detections"), color, "detections")) != 1:
        raise GraspLiftError(f"RGB-D 报告中必须恰有一个有效 {color} detection")
    if len(_matching_valid_colors(rgbd.get("slots"), color, "slots")) != 1:
        raise GraspLiftError(f"RGB-D 报告中必须恰有一个有效 {color} slot")

    object_xyz = _vector(motion.get("detected_battery_base_xyz_m"), 3, "motion.detected_battery_base_xyz_m")
    motion_target = _pose(motion.get("target_tcp_pose"), "motion.target_tcp_pose")
    motion_actual = _pose(motion.get("actual_tcp_pose"), "motion.actual_tcp_pose")
    rotation_initial = _pose(rotation.get("initial_tcp_pose"), "rotation.initial_tcp_pose")
    rotation_target = _pose(rotation.get("target_tcp_pose"), "rotation.target_tcp_pose")
    rotation_actual = _pose(rotation.get("actual_tcp_pose"), "rotation.actual_tcp_pose")
    if math.hypot(motion_target.x - object_xyz[0], motion_target.y - object_xyz[1]) > 0.001:
        raise GraspLiftError("悬空目标 XY 与识别到的电池基坐标 XY 不一致")
    _require_pose_chain(motion_actual, rotation_initial, config)
    if translation_error(rotation_target.as_list(), rotation_actual.as_list()) > 0.001:
        raise GraspLiftError("预抓取旋转未到达目标位置")
    if rotation_error(rotation_target.as_list()[3:], rotation_actual.as_list()[3:]) > math.radians(0.5):
        raise GraspLiftError("预抓取旋转未到达目标姿态")
    if math.hypot(rotation_actual.x - object_xyz[0], rotation_actual.y - object_xyz[1]) > 0.001:
        raise GraspLiftError("旋转后的 TCP XY 不再位于所选电池中心")
    if rotation_actual.z - object_xyz[2] < 0.10:
        raise GraspLiftError("旋转后 TCP 与电池的垂直间隙不足 100 mm")
    return GraspEvidence(color, rotation_actual, object_xyz)


def compute_descent_z(
    start_z: float,
    target_z: float,
    max_segment_m: float,
    final_slow_distance_m: float,
) -> list[tuple[float, bool]]:
    if not all(math.isfinite(value) for value in (start_z, target_z, max_segment_m, final_slow_distance_m)):
        raise GraspLiftError("下降参数必须是有限数")
    total = start_z - target_z
    if total <= 0.0 or max_segment_m <= 0.0 or not 0.0 < final_slow_distance_m <= max_segment_m:
        raise GraspLiftError("下降几何参数无效")
    slow_start = target_z + min(total, final_slow_distance_m)
    result: list[tuple[float, bool]] = []
    current = start_z
    while current - slow_start > 1e-12:
        current = max(slow_start, current - max_segment_m)
        result.append((current, False))
    if current - target_z > 1e-12:
        result.append((target_z, True))
    else:
        result[-1] = (result[-1][0], True)
    return result


def build_plan(config: GraspLiftConfig, evidence: GraspEvidence) -> GraspLiftPlan:
    config.require_valid(allow_missing_final_z=False)
    assert config.final_grasp_z_m is not None
    total = evidence.start_pose.z - config.final_grasp_z_m
    if not 0.0 < total <= config.max_total_descent_m:
        raise GraspLiftError(
            f"总下降 {total * 1000:.1f} mm 不在 (0,{config.max_total_descent_m * 1000:.1f}] mm"
        )
    descent = tuple(
        VerticalStep(
            f"descend_{index:02d}",
            evidence.start_pose.with_z(z),
            config.contact_speed_mps if slow else config.approach_speed_mps,
            slow,
        )
        for index, (z, slow) in enumerate(
            compute_descent_z(
                evidence.start_pose.z,
                config.final_grasp_z_m,
                config.max_z_segment_m,
                config.final_slow_distance_m,
            ),
            start=1,
        )
    )
    lift_z = list(reversed([step.pose.z for step in descent[:-1]])) + [evidence.start_pose.z]
    lift = tuple(
        VerticalStep(f"lift_{index:02d}", evidence.start_pose.with_z(z), config.lift_speed_mps)
        for index, z in enumerate(lift_z, start=1)
    )
    plan = GraspLiftPlan(
        evidence.color,
        evidence.start_pose,
        evidence.start_pose.with_z(config.final_grasp_z_m),
        descent,
        lift,
    )
    validate_plan(plan, config)
    return plan


def validate_plan(plan: GraspLiftPlan, config: GraspLiftConfig) -> None:
    prior = plan.start_pose
    for step in (*plan.descent, *plan.lift):
        if not _same_xy_orientation(step.pose, plan.start_pose):
            raise GraspLiftError(f"{step.label} 改变了 XY 或工具姿态")
        if abs(step.pose.z - prior.z) > config.max_z_segment_m + 1e-12:
            raise GraspLiftError(f"{step.label} 的 Z 分段超过上限")
        prior = step.pose
    if not plan.descent or plan.descent[-1].pose != plan.target_pose:
        raise GraspLiftError("下降路径没有以最终抓取位姿结束")
    if not plan.descent[-1].slow or plan.descent[-1].speed_mps != config.contact_speed_mps:
        raise GraspLiftError("最后下降段必须使用接触慢速")
    if not plan.lift or plan.lift[-1].pose != plan.start_pose:
        raise GraspLiftError("提升路径没有返回原高位")


def execute_plan(
    plan: GraspLiftPlan,
    config: GraspLiftConfig,
    robot: RobotBase,
    gripper: GripperBase,
    *,
    sleep: Callable[[float], None] = time.sleep,
    audit: ExecutionAudit | None = None,
) -> ExecutionAudit:
    if robot.is_real:
        raise GraspLiftError("Historical plans are offline-only; use battery_sorting.closed_loop")
    validate_plan(plan, config)
    audit = audit or ExecutionAudit(color=plan.color)
    descent_complete = False
    try:
        if not gripper.open():
            raise GraspLiftError("夹爪张开失败")
        audit.event("gripper_open_high")
        sleep(config.grip_settle_s)
        for step in plan.descent:
            _move_step(robot, step, config)
            audit.event("move", label=step.label, z_m=step.pose.z, speed_mps=step.speed_mps)
        descent_complete = True
        if not gripper.close():
            raise GraspLiftError("夹爪闭合失败")
        audit.event("gripper_close")
        sleep(config.grip_settle_s)
        audit.grip_detected = gripper.has_object()
        audit.event("object_feedback", detected=audit.grip_detected)
        if audit.grip_detected is not True:
            raise GraspLiftError("夹爪未确认抓到电池；停止原地检查")
        for step in plan.lift:
            _move_step(robot, step, config)
            audit.event("move", label=step.label, z_m=step.pose.z, speed_mps=step.speed_mps)
            if gripper.has_object() is not True:
                raise GraspLiftError("提升时夹持反馈丢失")
        sleep(2.0)
        if gripper.has_object() is not True:
            raise GraspLiftError("提升保持时夹持反馈丢失")
        audit.passed = True
        return audit
    except BaseException as exc:
        audit.error = str(exc)
        try:
            robot.stop_l(0.2)
            audit.event("stop_l")
        except Exception as stop_exc:
            audit.event("stop_l_failed", error=str(stop_exc))
        raise


def dynamic_confirmation(plan: GraspLiftPlan) -> str:
    return (
        f"EXECUTE GRASP {plan.color.upper()} "
        f"Z {plan.start_pose.z:.6f}->{plan.target_pose.z:.6f} "
        f"DESCENT {plan.total_descent_m * 1000:.1f}MM"
    )


def plan_command(args: argparse.Namespace) -> int:
    try:
        config = GraspLiftConfig.from_json(args.grasp_config)
        evidence = load_evidence(
            args.calibration, args.rgbd_report, args.motion_report, args.rotation_report, config
        )
        print(f"证据链通过：目标颜色={evidence.color}，预抓取高位 Z={evidence.start_pose.z:.6f} m")
        if config.final_grasp_z_m is None:
            print("BLOCKED AS DESIGNED: final_grasp_z_m=null；不会生成下降路径或连接硬件")
            return 0
        plan = build_plan(config, evidence)
        if getattr(args, "export_simulation", None):
            from .simulation_export import export_grasp_lift_plan, write_plan
            write_plan(export_grasp_lift_plan(plan, config), args.export_simulation)
            print(f"已导出真实 Python 抓取提升计划：{args.export_simulation}；未连接硬件")
        print(
            f"下降 {plan.total_descent_m * 1000:.1f} mm，共 {len(plan.descent)} 段；"
            f"最后 {config.final_slow_distance_m * 1000:.1f} mm 使用 {config.contact_speed_mps:.3f} m/s"
        )
        for step in plan.descent:
            print(f"  {step.label}: Z={step.pose.z:.6f} m, speed={step.speed_mps:.3f} m/s")
        print(f"动态确认口令：{dynamic_confirmation(plan)}")
        return 0
    except (OSError, ValueError, GraspLiftError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2


def execute_command(
    args: argparse.Namespace,
    *,
    robot_factory: Callable[[RobotConfig], RobotBase] = RealUR5,
    gripper_factory: Callable[[GripperConfig], GripperBase] = create_gripper,
    input_fn: Callable[[str], str] = input,
) -> int:
    robot: RobotBase | None = None
    gripper: GripperBase | None = None
    audit = ExecutionAudit()
    output = Path(args.output)
    config: GraspLiftConfig | None = None
    plan: GraspLiftPlan | None = None
    try:
        config = GraspLiftConfig.from_json(args.grasp_config)
        # Intentionally before report reads and before either hardware factory.
        config.require_valid(allow_missing_final_z=False)
        if robot_factory is RealUR5:
            raise GraspLiftError("Historical report execution disabled; use battery_sorting.closed_loop run")
        if not all((args.execute, args.tcp_payload_confirmed, args.emergency_stop_ready, args.ignore_tcp_force)):
            raise GraspLiftError(
                "真机需要 --execute --tcp-payload-confirmed --emergency-stop-ready --ignore-tcp-force"
            )
        evidence = load_evidence(
            args.calibration, args.rgbd_report, args.motion_report, args.rotation_report, config
        )
        plan = build_plan(config, evidence)
        audit.color = plan.color
        robot = robot_factory(config.robot_config(plan.start_pose))
        robot.connect_readonly()
        assert_pose_close(
            robot.snapshot().tcp_pose,
            plan.start_pose,
            config.start_translation_tolerance_m,
            math.radians(config.start_rotation_tolerance_deg),
            "执行前 TCP 必须位于已验证预抓取高位",
        )
        gripper = gripper_factory(config.gripper)
        if not gripper.is_configured or gripper.is_mock or not gripper.has_object_feedback:
            raise GraspLiftError("必须使用已配置、非 mock 且有物体反馈的真实夹爪")
        gripper.connect()
        # connect() is deliberately read-only.  A real gripper must already be
        # initialized before this command starts; open() will fail closed if it
        # is not.  Do not hide calibration/initialization writes inside grasp.
        before = robot.snapshot().tcp_pose
        phrase = dynamic_confirmation(plan)
        if input_fn(f"请输入完整确认口令：\n{phrase}\n> ").strip() != phrase:
            raise GraspLiftError("确认口令不匹配")
        after = robot.snapshot().tcp_pose
        if translation_error(before, after) > config.confirm_translation_tolerance_m:
            raise GraspLiftError("人工确认期间 TCP 平移发生变化")
        if rotation_error(before[3:], after[3:]) > math.radians(config.confirm_rotation_tolerance_deg):
            raise GraspLiftError("人工确认期间 TCP 姿态发生变化")
        assert_pose_unchanged(before, after)
        robot.connect_control()
        assert_pose_close(
            robot.snapshot().tcp_pose,
            plan.start_pose,
            config.start_translation_tolerance_m,
            math.radians(config.start_rotation_tolerance_deg),
            "建立控制连接后 TCP 必须仍位于已验证预抓取高位",
        )
        if not gripper.initialize(recalibrate=False):
            raise GraspLiftError("夹爪在预抓取高位初始化失败")
        audit.event("gripper_initialize_home", recalibrate=False)
        assert_pose_close(
            robot.snapshot().tcp_pose,
            plan.start_pose,
            config.start_translation_tolerance_m,
            math.radians(config.start_rotation_tolerance_deg),
            "夹爪初始化后 TCP 必须仍位于已验证预抓取高位",
        )
        execute_plan(plan, config, robot, gripper, audit=audit)
        _write_audit(output, audit, config, plan, best_effort=False)
        print(f"PASS: {plan.color} 已夹持并返回原高位；审计报告 {output.resolve()}")
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        audit.error = str(exc)
        _write_audit(output, audit, config, plan, best_effort=True)
        print(f"STOP: {exc}", file=sys.stderr)
        return 2
    finally:
        # Neither disconnect method releases a held object.
        if robot is not None:
            try:
                robot.disconnect()
            except Exception:
                pass
        if gripper is not None:
            try:
                gripper.disconnect()
            except Exception:
                pass


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grasp-config", default=str(root / "config" / "grasp_lift.json"))
    parser.add_argument("--calibration", default=str(root / "output" / "handeye" / "session_20260910" / "handeye_result.json"))
    parser.add_argument("--rgbd-report", default=str(root / "output" / "hover_validation" / "capture_20260910_retry" / "report.json"))
    parser.add_argument("--motion-report", default=str(root / "output" / "hover_validation" / "green_hover_result.json"))
    parser.add_argument("--rotation-report", default=str(root / "output" / "hover_validation" / "green_pregrasp_rotation.json"))
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--export-simulation", type=Path, help="导出 JSON 仿真计划，保持只读离线")
    plan_parser.set_defaults(function=plan_command)
    execute_parser = subparsers.add_parser("execute")
    execute_parser.add_argument("--output", default=str(root / "output" / "grasp_lift" / "latest.json"))
    execute_parser.add_argument("--execute", action="store_true")
    execute_parser.add_argument("--tcp-payload-confirmed", action="store_true")
    execute_parser.add_argument("--emergency-stop-ready", action="store_true")
    execute_parser.add_argument("--ignore-tcp-force", action="store_true")
    execute_parser.set_defaults(function=execute_command)
    return parser


def _move_step(robot: RobotBase, step: VerticalStep, config: GraspLiftConfig) -> None:
    accepted = robot.move_l_without_tcp_force_guard(
        Waypoint(step.label, step.pose, "contact" if step.slow else "approach"),
        step.speed_mps,
        config.acceleration_mps2,
    )
    if not accepted:
        raise GraspLiftError(f"UR5 拒绝运动 {step.label}")


def _write_audit(
    path: Path,
    audit: ExecutionAudit,
    config: GraspLiftConfig | None,
    plan: GraspLiftPlan | None,
    *,
    best_effort: bool,
) -> None:
    payload = {
        "passed": audit.passed,
        "started_at_utc": audit.started_at_utc,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "color": audit.color,
        "grip_detected": audit.grip_detected,
        "error": audit.error,
        "events": audit.events,
        "tcp_force_ignored_by_operator_request": True,
        "final_grasp_z_m": None if config is None else config.final_grasp_z_m,
        "start_tcp_pose": None if plan is None else plan.start_pose.as_list(),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except OSError:
        if not best_effort:
            raise


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    missing, unknown = expected - set(value), set(value) - expected
    if missing or unknown:
        raise GraspLiftError(f"{label}字段不匹配，缺少={sorted(missing)}，未知={sorted(unknown)}")


def _matching_valid_colors(value: Any, color: str, label: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise GraspLiftError(f"RGB-D {label} 必须是列表")
    return [
        item for item in value
        if isinstance(item, dict) and item.get("valid") is True and item.get("color") == color
    ]


def _require_pose_chain(first: Pose, second: Pose, config: GraspLiftConfig) -> None:
    trans = translation_error(first.as_list(), second.as_list())
    rot = rotation_error(first.as_list()[3:], second.as_list()[3:])
    if trans > config.report_chain_translation_tolerance_m or rot > math.radians(config.report_chain_rotation_tolerance_deg):
        raise GraspLiftError(f"悬空实际位姿→旋转初始位姿不连续：{trans * 1000:.3f} mm / {math.degrees(rot):.3f}°")


def _same_xy_orientation(first: Pose, second: Pose) -> bool:
    return (
        math.isclose(first.x, second.x, abs_tol=1e-12)
        and math.isclose(first.y, second.y, abs_tol=1e-12)
        and rotation_error(first.as_list()[3:], second.as_list()[3:]) <= 1e-12
    )


def _pose(value: Any, label: str) -> Pose:
    return Pose(*_vector(value, 6, label))


def _vector(value: Any, length: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise GraspLiftError(f"{label} 必须是 {length} 个数值")
    return tuple(_number(item, f"{label}[{index}]") for index, item in enumerate(value))


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GraspLiftError(f"{label} 必须是有限数值")
    result = float(value)
    if not math.isfinite(result):
        raise GraspLiftError(f"{label} 必须是有限数值")
    return result


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GraspLiftError(f"{label} 必须是整数")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GraspLiftError(f"{label} 必须是非空字符串")
    return value.strip()


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.function(args))


if __name__ == "__main__":
    main()
