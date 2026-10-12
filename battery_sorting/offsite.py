"""Pure off-site planning, readiness, replay, and geometry utilities.

Nothing in this module opens a camera, serial port, network socket, or robot
interface.  It exists so the complete competition flow can be tested before
the remaining site measurements are available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


TARGET_NAMES = ("T0", "P1", "P2", "P3")
COLORS = ("red", "yellow", "blue", "green")
VERIFIED_PICK_CONTACT_Z_M = 0.069207
VERIFIED_SAFE_HIGH_Z_M = 0.447856
VERIFIED_FINAL_SLOW_DISTANCE_M = 0.01
VERIFIED_TOOL_ROTATION_VECTOR_RAD = (
    -2.2448617460523574,
    -2.1864166279658708,
    0.00325035854435066,
)
# Levelled reference for the rule requiring the flange plane parallel to the
# table. This is a configured orientation, not a new physical verification.
TABLE_PARALLEL_TOOL_ROTATION_VECTOR_RAD = (2.250560630699371,2.1919355028456944,0.)


class OffsiteError(RuntimeError):
    """An offline configuration, plan, or replay bundle is invalid."""


@dataclass(frozen=True)
class Workspace:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def contains(self, pose: "Pose6") -> bool:
        return (
            self.x_min <= pose.x <= self.x_max
            and self.y_min <= pose.y <= self.y_max
            and self.z_min <= pose.z <= self.z_max
        )


@dataclass(frozen=True)
class Pose6:
    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float

    @classmethod
    def parse(cls, value: Any, where: str) -> "Pose6":
        vector = _finite_vector(value, 6, where)
        return cls(*vector)

    def with_xyz(self, x: float, y: float, z: float) -> "Pose6":
        return Pose6(x, y, z, self.rx, self.ry, self.rz)

    def as_tuple(self) -> tuple[float, ...]:
        return self.x, self.y, self.z, self.rx, self.ry, self.rz


@dataclass(frozen=True)
class Timing:
    grip_settle_s: float
    lifted_hold_s: float
    release_settle_s: float


@dataclass(frozen=True)
class Verification:
    gripper_live_verified: bool
    hover_verified_by_color: Mapping[str, bool]
    full_cycle_live_verified: bool


@dataclass(frozen=True)
class SiteModel:
    schema_version: int
    robot_host: str
    dashboard_port: int
    rtde_port: int
    gripper_driver: str
    gripper_port: str
    handeye_result_path: str
    pick_contact_z_m: float
    safe_high_z_m: float
    final_slow_distance_m: float
    tool_rotation_vector_rad: tuple[float, float, float]
    home_pose_m_rad: Pose6 | None
    placement_contact_z_m: float | None
    workspace_bounds_m: Workspace | None
    target_centers_base_xy_m: Mapping[str, tuple[float, float] | None]
    timing: Timing
    verification: Verification


def _exact_keys(value: Mapping[str, Any], expected: set[str], where: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise OffsiteError(f"{where} 字段不匹配: missing={missing}, extra={extra}")


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OffsiteError(f"{where} 必须是 JSON 对象")
    return value


def _finite(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OffsiteError(f"{where} 必须是数值")
    result = float(value)
    if not math.isfinite(result):
        raise OffsiteError(f"{where} 必须是有限数")
    return result


def _integer(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise OffsiteError(f"{where} 必须是整数")
    return value


def _finite_vector(value: Any, length: int, where: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise OffsiteError(f"{where} 必须包含 {length} 个数")
    return tuple(_finite(item, f"{where}[{index}]") for index, item in enumerate(value))


def _optional_xy(value: Any, where: str) -> tuple[float, float] | None:
    return None if value is None else _finite_vector(value, 2, where)  # type: ignore[return-value]


def load_site(source: str | Path | Mapping[str, Any]) -> SiteModel:
    """Load the strict, standalone competition configuration."""

    if isinstance(source, (str, Path)):
        try:
            root = json.loads(Path(source).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise OffsiteError(f"无法读取离线比赛配置 {source}: {exc}") from exc
    else:
        root = source
    root = _mapping(root, "root")
    _exact_keys(root, {"schema_version", "robot", "gripper", "calibration", "geometry", "timing", "verification"}, "root")

    robot = _mapping(root["robot"], "robot")
    _exact_keys(robot, {"host", "dashboard_port", "rtde_port"}, "robot")
    gripper = _mapping(root["gripper"], "gripper")
    _exact_keys(gripper, {"driver", "port"}, "gripper")
    calibration = _mapping(root["calibration"], "calibration")
    _exact_keys(calibration, {"handeye_result"}, "calibration")
    geometry = _mapping(root["geometry"], "geometry")
    _exact_keys(
        geometry,
        {
            "pick_contact_z_m",
            "safe_high_z_m",
            "final_slow_distance_m",
            "tool_rotation_vector_rad",
            "home_pose_m_rad",
            "placement_contact_z_m",
            "workspace_bounds_m",
            "target_centers_base_xy_m",
        },
        "geometry",
    )
    target_centers = _mapping(geometry["target_centers_base_xy_m"], "geometry.target_centers_base_xy_m")
    _exact_keys(target_centers, set(TARGET_NAMES), "geometry.target_centers_base_xy_m")
    timing_raw = _mapping(root["timing"], "timing")
    _exact_keys(timing_raw, {"grip_settle_s", "lifted_hold_s", "release_settle_s"}, "timing")
    verification_raw = _mapping(root["verification"], "verification")
    _exact_keys(
        verification_raw,
        {"gripper_live_verified", "hover_verified_by_color", "full_cycle_live_verified"},
        "verification",
    )
    hovers = _mapping(verification_raw["hover_verified_by_color"], "verification.hover_verified_by_color")
    _exact_keys(hovers, set(COLORS), "verification.hover_verified_by_color")

    workspace_raw = geometry["workspace_bounds_m"]
    workspace = None
    if workspace_raw is not None:
        values = _finite_vector(workspace_raw, 6, "geometry.workspace_bounds_m")
        workspace = Workspace(*values)
        if not (workspace.x_min < workspace.x_max and workspace.y_min < workspace.y_max and workspace.z_min < workspace.z_max):
            raise OffsiteError("workspace_bounds_m 各下限必须小于上限")
    home_raw = geometry["home_pose_m_rad"]
    home = None if home_raw is None else Pose6.parse(home_raw, "geometry.home_pose_m_rad")
    placement_z = geometry["placement_contact_z_m"]
    placement_z = None if placement_z is None else _finite(placement_z, "geometry.placement_contact_z_m")
    pick_z = _finite(geometry["pick_contact_z_m"], "geometry.pick_contact_z_m")
    safe_z = _finite(geometry["safe_high_z_m"], "geometry.safe_high_z_m")
    slow = _finite(geometry["final_slow_distance_m"], "geometry.final_slow_distance_m")
    if not (pick_z < safe_z and 0.0 < slow < safe_z - pick_z):
        raise OffsiteError("抓取 Z、安全高位和末段慢速距离关系无效")
    timing = Timing(
        _finite(timing_raw["grip_settle_s"], "timing.grip_settle_s"),
        _finite(timing_raw["lifted_hold_s"], "timing.lifted_hold_s"),
        _finite(timing_raw["release_settle_s"], "timing.release_settle_s"),
    )
    if timing.grip_settle_s < 0.0 or timing.lifted_hold_s < 0.0 or timing.release_settle_s < 3.0:
        raise OffsiteError("等待时间必须非负，释放稳定时间必须至少 3 秒")
    for key, value in verification_raw.items():
        if key == "hover_verified_by_color":
            continue
        if not isinstance(value, bool):
            raise OffsiteError(f"verification.{key} 必须是布尔值")
    if any(not isinstance(value, bool) for value in hovers.values()):
        raise OffsiteError("hover_verified_by_color 的值必须是布尔值")

    result = SiteModel(
        schema_version=_integer(root["schema_version"], "schema_version"),
        robot_host=str(robot["host"]),
        dashboard_port=_integer(robot["dashboard_port"], "robot.dashboard_port"),
        rtde_port=_integer(robot["rtde_port"], "robot.rtde_port"),
        gripper_driver=str(gripper["driver"]),
        gripper_port=str(gripper["port"]),
        handeye_result_path=str(calibration["handeye_result"]),
        pick_contact_z_m=pick_z,
        safe_high_z_m=safe_z,
        final_slow_distance_m=slow,
        tool_rotation_vector_rad=_finite_vector(geometry["tool_rotation_vector_rad"], 3, "geometry.tool_rotation_vector_rad"),  # type: ignore[arg-type]
        home_pose_m_rad=home,
        placement_contact_z_m=placement_z,
        workspace_bounds_m=workspace,
        target_centers_base_xy_m={
            name: _optional_xy(target_centers[name], f"target_centers.{name}")
            for name in TARGET_NAMES
        },
        timing=timing,
        verification=Verification(
            bool(verification_raw["gripper_live_verified"]),
            {color: bool(hovers[color]) for color in COLORS},
            bool(verification_raw["full_cycle_live_verified"]),
        ),
    )
    if result.schema_version != 1:
        raise OffsiteError("只支持 schema_version=1")
    if result.robot_host != "192.168.1.10" or result.gripper_port.upper() != "COM11":
        raise OffsiteError("现场配置必须使用已复核的 192.168.1.10 和 COM11")
    if result.gripper_driver != "ag105145_modbus":
        raise OffsiteError("现场夹爪驱动必须是 ag105145_modbus")
    if not math.isclose(result.pick_contact_z_m, VERIFIED_PICK_CONTACT_Z_M, abs_tol=1e-9):
        raise OffsiteError(f"pick_contact_z_m 必须使用已复核值 {VERIFIED_PICK_CONTACT_Z_M}")
    if not math.isclose(result.safe_high_z_m, VERIFIED_SAFE_HIGH_Z_M, abs_tol=1e-9):
        raise OffsiteError(f"safe_high_z_m 必须使用已复核值 {VERIFIED_SAFE_HIGH_Z_M}")
    if not math.isclose(
        result.final_slow_distance_m,
        VERIFIED_FINAL_SLOW_DISTANCE_M,
        abs_tol=1e-9,
    ):
        raise OffsiteError(
            f"final_slow_distance_m 必须使用已复核值 {VERIFIED_FINAL_SLOW_DISTANCE_M}"
        )
    # Keep the archived tilt for replay, and explicitly permit its levelled
    # counterpart selected for the competition's table-parallel requirement.
    allowed_rotations=(VERIFIED_TOOL_ROTATION_VECTOR_RAD,
                       TABLE_PARALLEL_TOOL_ROTATION_VECTOR_RAD)
    if not any(np.allclose(result.tool_rotation_vector_rad,rotation,atol=1e-9,rtol=0.)
               for rotation in allowed_rotations):
        raise OffsiteError("tool_rotation_vector_rad 必须使用历史或桌面平行的限定 UR 旋转向量")
    if not (1 <= result.dashboard_port <= 65535 and 1 <= result.rtde_port <= 65535):
        raise OffsiteError("机器人端口超出 1..65535")
    if home is not None and not math.isclose(home.z, safe_z, abs_tol=1e-6):
        fold_path=Path(__file__).resolve().parents[1]/'config/folded_standby.json'
        fold=json.loads(fold_path.read_text(encoding='utf-8')) if fold_path.exists() else {}
        if not np.allclose(home.as_tuple(),fold.get('home_pose_m_rad',[float('inf')]*6),atol=1e-8,rtol=0):
            raise OffsiteError("home.z 必须等于 safe_high_z_m，或使用明确配置的关节折叠待机位")
    return result


def readiness_report(site: SiteModel, task: str = "all", colors: Sequence[str] = COLORS) -> dict[str, Any]:
    """Return offline blockers separately from later live-verification work."""

    if task not in {"basic", "sequence", "all"}:
        raise OffsiteError("task 必须是 basic/sequence/all")
    blockers: list[str] = []
    if site.home_pose_m_rad is None:
        blockers.append("geometry.home_pose_m_rad 仍为 null")
    if site.placement_contact_z_m is None:
        blockers.append("geometry.placement_contact_z_m 仍为 null")
    if site.workspace_bounds_m is None:
        blockers.append("geometry.workspace_bounds_m 仍为 null")
    required_targets = ("T0",) if task == "basic" else ("P1", "P2", "P3") if task == "sequence" else TARGET_NAMES
    for target in required_targets:
        if site.target_centers_base_xy_m[target] is None:
            blockers.append(f"geometry.target_centers_base_xy_m.{target} 仍为 null")
    calibration_ok = False
    calibration_reason = ""
    path = Path(site.handeye_result_path)
    try:
        calibration = json.loads(path.read_text(encoding="utf-8"))
        calibration_ok = calibration.get("passed") is True
        if not calibration_ok:
            calibration_reason = "手眼标定报告未通过"
    except (OSError, json.JSONDecodeError) as exc:
        calibration_reason = f"无法读取手眼标定报告: {exc}"
    if not calibration_ok:
        blockers.append(calibration_reason)

    requested_colors = tuple(str(color).lower() for color in colors)
    if any(color not in COLORS for color in requested_colors):
        raise OffsiteError("colors 只能包含 red/yellow/blue/green")
    pending_live: list[str] = []
    if not site.verification.gripper_live_verified:
        pending_live.append("COM11 夹爪初始化/开合/物体反馈尚未现场复核")
    for color in requested_colors:
        if not site.verification.hover_verified_by_color[color]:
            pending_live.append(f"{color} 悬空 XY 尚未现场复核")
    if not site.verification.full_cycle_live_verified:
        pending_live.append("完整抓取—放置循环尚未现场复核")
    return {
        "offline_plan_ready": not blockers,
        "live_competition_ready": not blockers and not pending_live,
        "blockers": blockers,
        "pending_live_verification": pending_live,
        "known": {
            "robot_host": site.robot_host,
            "gripper_port": site.gripper_port,
            "pick_contact_z_m": site.pick_contact_z_m,
            "safe_high_z_m": site.safe_high_z_m,
            "handeye_passed": calibration_ok,
        },
    }


class StepKind(str, Enum):
    INITIALIZE = "initialize"
    MOVE = "move"
    OPEN = "open"
    CLOSE = "close"
    CHECK_OBJECT = "check_object"
    DWELL = "dwell"


@dataclass(frozen=True)
class PlanStep:
    kind: StepKind
    label: str
    start_pose: Pose6 | None = None
    end_pose: Pose6 | None = None
    profile: str | None = None
    seconds: float = 0.0


@dataclass(frozen=True)
class OfflinePlan:
    task: str
    colors: tuple[str, ...]
    steps: tuple[PlanStep, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "colors": list(self.colors),
            "steps": [
                {
                    "kind": step.kind.value,
                    "label": step.label,
                    "start_pose": None if step.start_pose is None else list(step.start_pose.as_tuple()),
                    "end_pose": None if step.end_pose is None else list(step.end_pose.as_tuple()),
                    "profile": step.profile,
                    "seconds": step.seconds,
                }
                for step in self.steps
            ],
        }


def _validate_xy_map(values: Mapping[str, Sequence[float]], colors: Sequence[str]) -> dict[str, tuple[float, float]]:
    output: dict[str, tuple[float, float]] = {}
    for color in colors:
        try:
            output[color] = _finite_vector(values[color], 2, f"pick_xy_by_color.{color}")  # type: ignore[assignment]
        except KeyError as exc:
            raise OffsiteError(f"缺少 {color} 的抓取 XY") from exc
    return output


def build_offline_plan(
    site: SiteModel,
    task: str,
    colors: Sequence[str],
    pick_xy_by_color: Mapping[str, Sequence[float]],
) -> OfflinePlan:
    """Build and validate a full basic or three-item sequence cycle."""

    chosen = tuple(str(color).lower() for color in colors)
    if task == "basic" and len(chosen) != 1:
        raise OffsiteError("basic 必须包含一个颜色")
    if task == "sequence" and (len(chosen) != 3 or len(set(chosen)) != 3):
        raise OffsiteError("sequence 必须包含三个互不重复的颜色")
    if task not in {"basic", "sequence"} or any(color not in COLORS for color in chosen):
        raise OffsiteError("任务或颜色无效")
    readiness = readiness_report(site, task, chosen)
    if readiness["blockers"]:
        raise OffsiteError("离线计划仍有阻塞项: " + "; ".join(readiness["blockers"]))
    assert site.home_pose_m_rad is not None
    assert site.placement_contact_z_m is not None
    assert site.workspace_bounds_m is not None
    picks = _validate_xy_map(pick_xy_by_color, chosen)
    target_names = ("T0",) if task == "basic" else ("P1", "P2", "P3")
    rotation = site.tool_rotation_vector_rad
    home = site.home_pose_m_rad
    current = home
    steps: list[PlanStep] = [PlanStep(StepKind.INITIALIZE, "initialize_gripper_at_home")]
    steps.append(PlanStep(StepKind.OPEN, "initial_gripper_open"))

    def pose(x: float, y: float, z: float) -> Pose6:
        return Pose6(x, y, z, *rotation)

    def move(label: str, destination: Pose6, profile: str) -> None:
        nonlocal current
        steps.append(PlanStep(StepKind.MOVE, label, current, destination, profile))
        current = destination

    work_home = pose(home.x, home.y, site.safe_high_z_m)
    if not np.allclose(home.as_tuple(), work_home.as_tuple(), atol=1e-12):
        move("orient_for_work", work_home, "transit")

    for color, target_name in zip(chosen, target_names):
        pick_xy = picks[color]
        target_xy = site.target_centers_base_xy_m[target_name]
        assert target_xy is not None
        pick_safe = pose(*pick_xy, site.safe_high_z_m)
        pick_slow = pose(*pick_xy, site.pick_contact_z_m + site.final_slow_distance_m)
        pick_contact = pose(*pick_xy, site.pick_contact_z_m)
        place_safe = pose(*target_xy, site.safe_high_z_m)
        place_slow = pose(*target_xy, site.placement_contact_z_m + site.final_slow_distance_m)
        place_contact = pose(*target_xy, site.placement_contact_z_m)
        move(f"{color}:pick_transit", pick_safe, "transit")
        move(f"{color}:pick_approach", pick_slow, "approach")
        move(f"{color}:pick_contact", pick_contact, "contact")
        steps.extend(
            (
                PlanStep(StepKind.CLOSE, f"{color}:gripper_close"),
                PlanStep(StepKind.DWELL, f"{color}:grip_settle", seconds=site.timing.grip_settle_s),
                PlanStep(StepKind.CHECK_OBJECT, f"{color}:check_after_close"),
            )
        )
        move(f"{color}:pick_lift", pick_safe, "approach")
        steps.append(PlanStep(StepKind.DWELL, f"{color}:lifted_hold", seconds=site.timing.lifted_hold_s))
        steps.append(PlanStep(StepKind.CHECK_OBJECT, f"{color}:check_after_lift"))
        move(f"{color}:{target_name}_transit", place_safe, "transit")
        steps.append(PlanStep(StepKind.CHECK_OBJECT, f"{color}:check_before_place"))
        move(f"{color}:{target_name}_approach", place_slow, "approach")
        move(f"{color}:{target_name}_contact", place_contact, "contact")
        steps.append(PlanStep(StepKind.OPEN, f"{color}:gripper_open"))
        move(f"{color}:{target_name}_retreat", place_safe, "approach")
        steps.append(PlanStep(StepKind.DWELL, f"{color}:release_stability", seconds=site.timing.release_settle_s))
    if not np.allclose(current.as_tuple()[3:], home.as_tuple()[3:], atol=1e-12):
        move("return_work_home", work_home, "transit")
    move("return_home", home, "transit")
    plan = OfflinePlan(task, chosen, tuple(steps))
    validate_offline_plan(plan, site)
    return plan


def validate_offline_plan(plan: OfflinePlan, site: SiteModel) -> None:
    """Reject discontinuities, low horizontal motion, non-vertical contact, or bounds errors."""

    if not plan.steps or plan.steps[0].kind is not StepKind.INITIALIZE:
        raise OffsiteError("计划必须在 home 初始化夹爪")
    previous: Pose6 | None = site.home_pose_m_rad
    for step in plan.steps:
        if step.kind is StepKind.MOVE:
            if step.start_pose is None or step.end_pose is None or step.profile not in {"transit", "approach", "contact"}:
                raise OffsiteError(f"{step.label} 移动字段不完整")
            if previous is None or not np.allclose(step.start_pose.as_tuple(), previous.as_tuple(), atol=1e-12):
                raise OffsiteError(f"{step.label} 路径不连续")
            if site.workspace_bounds_m is None or not (
                site.workspace_bounds_m.contains(step.start_pose)
                and site.workspace_bounds_m.contains(step.end_pose)
            ):
                raise OffsiteError(f"{step.label} 超出工作区")
            dx = step.end_pose.x - step.start_pose.x
            dy = step.end_pose.y - step.start_pose.y
            dz = step.end_pose.z - step.start_pose.z
            horizontal = math.hypot(dx, dy)
            orientation_changed = not np.allclose(
                step.start_pose.as_tuple()[3:],
                step.end_pose.as_tuple()[3:],
                atol=1e-12,
            )
            if orientation_changed and (
                step.profile != "transit"
                or horizontal > 1e-9
                or abs(dz) > 1e-9
                or not math.isclose(step.start_pose.z, site.safe_high_z_m, abs_tol=1e-9)
            ):
                raise OffsiteError(f"{step.label} 工具旋转必须在安全高位原地进行")
            if horizontal > 1e-9 and (
                not math.isclose(step.start_pose.z, site.safe_high_z_m, abs_tol=1e-9)
                or not math.isclose(step.end_pose.z, site.safe_high_z_m, abs_tol=1e-9)
                or abs(dz) > 1e-9
            ):
                raise OffsiteError(f"{step.label} 水平移动不在安全高位")
            if step.profile == "contact" and (horizontal > 1e-9 or dz >= -1e-9):
                raise OffsiteError(f"{step.label} 接触段必须是向下的非零纯 Z 运动")
            previous = step.end_pose
        elif step.kind is StepKind.DWELL:
            if not math.isfinite(step.seconds) or step.seconds < 0.0:
                raise OffsiteError(f"{step.label} 等待时间无效")
    if site.home_pose_m_rad is None or previous is None or not np.allclose(
        previous.as_tuple(), site.home_pose_m_rad.as_tuple(), atol=1e-12
    ):
        raise OffsiteError("计划没有返回 home")
    releases = [step for step in plan.steps if step.label.endswith(":release_stability")]
    if len(releases) != len(plan.colors) or any(step.seconds < 3.0 for step in releases):
        raise OffsiteError("每块电池释放后必须稳定至少 3 秒")


def simulate_offline_plan(plan: OfflinePlan, site: SiteModel) -> dict[str, Any]:
    """Execute the state transitions without a robot or gripper object."""

    validate_offline_plan(plan, site)
    initialized = False
    gripper_open = False
    holding = False
    events: list[str] = []
    for step in plan.steps:
        if step.kind is StepKind.INITIALIZE:
            initialized = True
        elif step.kind is StepKind.OPEN:
            if not initialized:
                raise OffsiteError("夹爪未初始化")
            gripper_open = True
            holding = False
        elif step.kind is StepKind.CLOSE:
            if not initialized:
                raise OffsiteError("夹爪未初始化")
            gripper_open = False
            holding = True
        elif step.kind is StepKind.CHECK_OBJECT and not holding:
            raise OffsiteError(f"{step.label} 模拟物体反馈失败")
        events.append(step.label)
    if holding or not gripper_open:
        raise OffsiteError("离线循环结束时夹爪状态不正确")
    return {"passed": True, "event_count": len(events), "events": events}


@dataclass(frozen=True)
class ReplayIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class ReplayBundle:
    colour: np.ndarray
    depth_m: np.ndarray
    intrinsics: ReplayIntrinsics
    tcp_pose_before: tuple[float, ...]
    tcp_pose_after: tuple[float, ...]
    handeye_result_path: str
    report: Mapping[str, Any]


def _rotation_distance(first: Sequence[float], second: Sequence[float]) -> float:
    first_matrix, _ = cv2.Rodrigues(np.asarray(first, dtype=np.float64))
    second_matrix, _ = cv2.Rodrigues(np.asarray(second, dtype=np.float64))
    relative = first_matrix.T @ second_matrix
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.acos(cosine)


def validate_replay_bundle(
    bundle: ReplayBundle,
    *,
    max_tcp_translation_m: float = 0.0002,
    max_tcp_rotation_deg: float = 0.1,
) -> None:
    if not isinstance(bundle.colour, np.ndarray) or bundle.colour.dtype != np.uint8 or bundle.colour.ndim != 3 or bundle.colour.shape[2] != 3:
        raise OffsiteError("colour 必须是 uint8 BGR 图像")
    if not isinstance(bundle.depth_m, np.ndarray) or bundle.depth_m.dtype != np.float32 or bundle.depth_m.ndim != 2:
        raise OffsiteError("depth_m 必须是 float32 二维数组")
    if bundle.colour.shape[:2] != bundle.depth_m.shape:
        raise OffsiteError("colour 与 depth_m 尺寸不一致")
    intrinsics = (
        bundle.intrinsics.fx,
        bundle.intrinsics.fy,
        bundle.intrinsics.cx,
        bundle.intrinsics.cy,
    )
    if not all(math.isfinite(value) for value in intrinsics) or bundle.intrinsics.fx <= 0.0 or bundle.intrinsics.fy <= 0.0:
        raise OffsiteError("相机内参无效")
    before = _finite_vector(bundle.tcp_pose_before, 6, "tcp_pose_before")
    after = _finite_vector(bundle.tcp_pose_after, 6, "tcp_pose_after")
    translation = math.dist(before[:3], after[:3])
    rotation_deg = math.degrees(_rotation_distance(before[3:], after[3:]))
    if translation > max_tcp_translation_m or rotation_deg > max_tcp_rotation_deg:
        raise OffsiteError(
            f"采集期间 TCP 漂移过大: {translation * 1000:.3f}mm, {rotation_deg:.3f}deg"
        )
    valid_depth = np.isfinite(bundle.depth_m) & (bundle.depth_m > 0.0)
    if float(np.count_nonzero(valid_depth)) / bundle.depth_m.size < 0.50:
        raise OffsiteError("有效深度覆盖率低于 50%")
    if not bundle.handeye_result_path.strip():
        raise OffsiteError("缺少 handeye_result_path")
    try:
        json.dumps(dict(bundle.report), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise OffsiteError(f"report 不能安全序列化: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_replay_bundle(directory: str | Path, bundle: ReplayBundle) -> Path:
    validate_replay_bundle(bundle)
    destination = Path(directory)
    if destination.exists() and any(destination.iterdir()):
        raise OffsiteError(f"回放目录非空，拒绝覆盖: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", bundle.colour)
    if not ok:
        raise OffsiteError("colour.png 编码失败")
    colour_path = destination / "colour.png"
    colour_path.write_bytes(encoded.tobytes())
    depth_path = destination / "depth_m.npy"
    with depth_path.open("wb") as handle:
        np.save(handle, bundle.depth_m, allow_pickle=False)
    manifest = {
        "schema_version": 1,
        "colour_file": "colour.png",
        "depth_file": "depth_m.npy",
        "shape": list(bundle.depth_m.shape),
        "intrinsics": asdict(bundle.intrinsics),
        "tcp_pose_before": list(bundle.tcp_pose_before),
        "tcp_pose_after": list(bundle.tcp_pose_after),
        "handeye_result_path": bundle.handeye_result_path,
        "report": dict(bundle.report),
        "sha256": {"colour.png": _sha256(colour_path), "depth_m.npy": _sha256(depth_path)},
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


def load_replay_bundle(directory: str | Path) -> ReplayBundle:
    source = Path(directory)
    try:
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OffsiteError(f"无法读取回放 manifest: {exc}") from exc
    expected_keys = {
        "schema_version", "colour_file", "depth_file", "shape", "intrinsics",
        "tcp_pose_before", "tcp_pose_after", "handeye_result_path", "report", "sha256",
    }
    _exact_keys(_mapping(manifest, "manifest"), expected_keys, "manifest")
    if manifest["schema_version"] != 1 or manifest["colour_file"] != "colour.png" or manifest["depth_file"] != "depth_m.npy":
        raise OffsiteError("回放 manifest 版本或固定文件名无效")
    colour_path = source / "colour.png"
    depth_path = source / "depth_m.npy"
    sha256_map = _mapping(manifest["sha256"], "manifest.sha256")
    _exact_keys(sha256_map, {"colour.png", "depth_m.npy"}, "manifest.sha256")
    for name, path in (("colour.png", colour_path), ("depth_m.npy", depth_path)):
        if _sha256(path) != sha256_map[name]:
            raise OffsiteError(f"{name} 完整性校验失败")
    colour = cv2.imdecode(np.fromfile(colour_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if colour is None:
        raise OffsiteError("colour.png 解码失败")
    try:
        with depth_path.open("rb") as handle:
            depth = np.load(handle, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise OffsiteError(f"depth_m.npy 读取失败: {exc}") from exc
    intrinsics_raw = _mapping(manifest["intrinsics"], "manifest.intrinsics")
    _exact_keys(intrinsics_raw, {"fx", "fy", "cx", "cy"}, "manifest.intrinsics")
    bundle = ReplayBundle(
        colour=colour,
        depth_m=depth,
        intrinsics=ReplayIntrinsics(**{name: _finite(intrinsics_raw[name], f"intrinsics.{name}") for name in ("fx", "fy", "cx", "cy")}),
        tcp_pose_before=_finite_vector(manifest["tcp_pose_before"], 6, "tcp_pose_before"),
        tcp_pose_after=_finite_vector(manifest["tcp_pose_after"], 6, "tcp_pose_after"),
        handeye_result_path=str(manifest["handeye_result_path"]),
        report=_mapping(manifest["report"], "manifest.report"),
    )
    if list(bundle.depth_m.shape) != manifest["shape"]:
        raise OffsiteError("manifest shape 与深度文件不一致")
    validate_replay_bundle(bundle)
    return bundle


def geometry_report() -> dict[str, Any]:
    """Return scoring margins without pretending nominal dimensions are tolerance."""

    battery = (70.0, 80.0)
    t0 = (100.0, 200.0)
    corner_radius = math.hypot(*battery)/2
    return {
        "dimensions_mm": {
            "battery": list(battery),
            "T0": list(t0),
            "P_inner_diameter": 80.0,
            "P_outer_diameter": 150.0,
        },
        "nominal_per_side_margin_mm": {
            "T0": [(t0[0] - battery[0]) / 2.0, (t0[1] - battery[1]) / 2.0],
        },
        "radial_corner_margin_mm": {"P_inner":40-corner_radius,"P_outer":75-corner_radius},
        "risk": "P 是圆形。70×80mm 电池对角线约 106.3mm，按完整底面入圈理解无法放入直径80mm内圆；内圆判分须核对裁判口径。",
        "strategy": [
            "T0 基础任务优先放在矩形中心，70mm 轴对准 100mm 边。",
            "P 区居中放置；不要用人为偏置换取余量。圆形包含关系与电池偏航无关。",
            "闭环仅验收完整落入直径150mm外圆，不声称获得内圆分。",
        ],
    }


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="离线比赛准备度与几何检查")
    sub = parser.add_subparsers(dest="command", required=True)
    readiness = sub.add_parser("readiness")
    readiness.add_argument("--config", required=True)
    readiness.add_argument("--task", choices=("basic", "sequence", "all"), default="all")
    sub.add_parser("geometry")
    replay = sub.add_parser("replay-check")
    replay.add_argument("directory")
    args = parser.parse_args(argv)
    if args.command == "readiness":
        payload = readiness_report(load_site(args.config), args.task)
    elif args.command == "geometry":
        payload = geometry_report()
    else:
        bundle = load_replay_bundle(args.directory)
        payload = {"valid": True, "shape": list(bundle.depth_m.shape)}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
