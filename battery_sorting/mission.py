"""把可靠颜色结果映射到示教槽位，并生成/执行固定坐标任务。"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from .config import BatterySortingConfig, ConfigError, Pose, RobotConfig
from .gripper import GripperBase
from .robot import PathPlanner, RobotBase, Waypoint, assert_pose_close
from .task import BasicTask, SequenceTask, Task
from .vision import DetectionError, DetectionResult, HomographyModel

LOGGER = logging.getLogger(__name__)


class MissionError(RuntimeError):
    """任务映射、夹爪或动作执行失败。"""


class ActionKind(str, Enum):
    MOVE = "move"
    OPEN = "open"
    CLOSE = "close"
    CHECK_OBJECT = "check_object"
    DWELL = "dwell"
    VISUAL_ACCEPT = "visual_accept"


@dataclass(frozen=True)
class MissionAction:
    kind: ActionKind
    label: str
    waypoint: Waypoint | None = None
    seconds: float = 0.0


@dataclass(frozen=True)
class MissionPlan:
    task_description: str
    actions: tuple[MissionAction, ...]
    dynamic_pick_poses: tuple[Pose, ...] = ()

    @property
    def waypoints(self) -> tuple[Waypoint, ...]:
        return tuple(
            action.waypoint
            for action in self.actions
            if action.kind is ActionKind.MOVE and action.waypoint is not None
        )


@dataclass(frozen=True)
class PlanMetrics:
    """Nominal translation-only estimate; it never replaces the competition timer."""

    total_translation_m: float
    nominal_motion_seconds: float
    dwell_seconds: float
    nominal_total_seconds: float


def compute_plan_metrics(
    plan: MissionPlan, home: Pose, robot: RobotConfig
) -> PlanMetrics:
    speeds = {
        "transit": robot.transit_speed,
        "approach": robot.approach_speed,
        "contact": robot.contact_speed,
    }
    previous = home
    distance_total = 0.0
    motion_total = 0.0
    dwell_total = 0.0
    for action in plan.actions:
        if action.kind is ActionKind.MOVE and action.waypoint is not None:
            current = action.waypoint.pose
            distance = math.dist(
                (previous.x, previous.y, previous.z),
                (current.x, current.y, current.z),
            )
            try:
                speed = speeds[action.waypoint.profile]
            except KeyError as exc:
                raise MissionError(
                    f"无法估时的未知速度档: {action.waypoint.profile}"
                ) from exc
            if speed <= 0.0 or robot.move_acceleration <= 0.0:
                raise MissionError("速度和加速度必须为正才能估算任务时间")
            distance_total += distance
            if distance > 0.0:
                acceleration_distance = speed**2 / robot.move_acceleration
                if distance >= acceleration_distance:
                    motion_total += distance / speed + speed / robot.move_acceleration
                else:
                    motion_total += 2.0 * math.sqrt(
                        distance / robot.move_acceleration
                    )
            previous = current
        elif action.kind is ActionKind.DWELL:
            if action.seconds < 0.0:
                raise MissionError(f"无法估算负等待时间: {action.label}")
            dwell_total += action.seconds
    return PlanMetrics(
        total_translation_m=distance_total,
        nominal_motion_seconds=motion_total,
        dwell_seconds=dwell_total,
        nominal_total_seconds=motion_total + dwell_total,
    )


class Mission:
    def __init__(
        self,
        config: BatterySortingConfig,
        robot: RobotBase,
        gripper: GripperBase,
        sleep_fn: Callable[[float], None] = time.sleep,
        mapping: HomographyModel | None = None,
        placement_verifier: Callable[[str], bool] | None = None,
    ) -> None:
        self.config = config
        self.robot = robot
        self.gripper = gripper
        self.sleep = sleep_fn
        self.path = PathPlanner(config.robot)
        self.mapping = mapping
        self.placement_verifier = placement_verifier

    def build_plan(
        self,
        task: Task,
        detection: DetectionResult,
        *,
        base_xy_by_color: dict[str, tuple[float, float]] | None = None,
    ) -> MissionPlan:
        if not detection.valid:
            raise MissionError(f"颜色识别无效: {detection.error}")
        try:
            color_to_centroid = detection.color_to_centroid()
        except DetectionError as exc:
            raise MissionError(f"无法取得电池图像位置: {exc}") from exc
        pairs: list[tuple[Pose, Pose, str]] = []
        if isinstance(task, BasicTask):
            pairs.append(
                (
                    self._dynamic_pick_pose(
                        task.target_color.value, color_to_centroid, base_xy_by_color
                    ),
                    self._pose("T0"),
                    task.target_color.value,
                )
            )
        elif isinstance(task, SequenceTask):
            for index, color in enumerate(task.color_order, start=1):
                pairs.append(
                    (
                        self._dynamic_pick_pose(
                            color.value, color_to_centroid, base_xy_by_color
                        ),
                        # The optional map is produced by the same-frame
                        # hand-eye RGB-D runtime mapper.
                        self._pose(f"P{index}"),
                        color.value,
                    )
                )
        else:  # pragma: no cover - protected by Task type
            raise MissionError(f"未知任务类型: {type(task).__name__}")

        dynamic_pick_poses = tuple(pick for pick, _place, _color in pairs)
        self._require_dynamic_pick_separation(dynamic_pick_poses)
        actions: list[MissionAction] = [
            MissionAction(ActionKind.OPEN, "initial_gripper_open"),
            MissionAction(
                ActionKind.DWELL,
                "initial_gripper_settle",
                seconds=self.config.robot.grip_settle_seconds,
            ),
        ]
        for pick, place, color in pairs:
            actions.extend(self._pick_place_actions(pick, place, color))
        home = self._pose("home")
        actions.append(
            MissionAction(ActionKind.MOVE, "return_home", self.path.transit(home, "return_home"))
        )
        self.path.validate_sequence(
            home,
            [action.waypoint for action in actions if action.waypoint is not None],
            dynamic_pick_poses,
        )
        return MissionPlan(task.description, tuple(actions), dynamic_pick_poses)

    def execute(self, plan: MissionPlan) -> None:
        home = self._pose("home")
        self._validate_dynamic_plan(plan)
        self.path.validate_sequence(home, plan.waypoints, plan.dynamic_pick_poses)
        assert_pose_close(
            self.robot.snapshot().tcp_pose,
            home,
            self.config.robot.home_translation_tolerance,
            self.config.robot.home_rotation_tolerance,
            "任务开始前必须位于 home",
        )
        expected_pose = home
        try:
            for action in plan.actions:
                if action.kind is ActionKind.MOVE:
                    assert_pose_close(
                        self.robot.snapshot().tcp_pose,
                        expected_pose,
                        self.config.robot.segment_start_translation_tolerance,
                        self.config.robot.segment_start_rotation_tolerance,
                        f"{action.label} 运动前起点检查",
                    )
                self._execute_action(action)
                if action.kind is ActionKind.MOVE and action.waypoint is not None:
                    expected_pose = action.waypoint.pose
        except (Exception, KeyboardInterrupt):
            try:
                self.robot.stop_l(self.config.robot.stop_deceleration)
            except Exception:  # pragma: no cover - best-effort hardware cleanup
                LOGGER.exception("stopL 失败，请立即使用示教器急停")
            try:
                self.gripper.stop()
            except Exception:
                LOGGER.exception("夹爪停止失败")
            raise

    def _pick_place_actions(self, pick: Pose, place: Pose, color: str) -> list[MissionAction]:
        path = self.path
        grip = self.config.robot.grip_settle_seconds
        lifted = self.config.robot.lifted_hold_seconds
        released = self.config.robot.release_settle_seconds
        actions = [
            MissionAction(ActionKind.MOVE, f"{color}:pick_transit", path.transit(pick, f"{color}:pick_transit")),
            MissionAction(ActionKind.MOVE, f"{color}:pick_approach", path.approach(pick, f"{color}:pick_approach")),
            MissionAction(ActionKind.MOVE, f"{color}:pick_contact", path.contact(pick, f"{color}:pick_contact")),
            MissionAction(ActionKind.CLOSE, f"{color}:gripper_close"),
            MissionAction(ActionKind.DWELL, f"{color}:grip_settle", seconds=grip),
            MissionAction(ActionKind.CHECK_OBJECT, f"{color}:check_after_close"),
            MissionAction(ActionKind.MOVE, f"{color}:lift_approach", path.approach(pick, f"{color}:lift_approach")),
            MissionAction(ActionKind.DWELL, f"{color}:lifted_hold", seconds=lifted),
            MissionAction(ActionKind.CHECK_OBJECT, f"{color}:check_after_lift"),
            MissionAction(ActionKind.MOVE, f"{color}:pick_retreat", path.transit(pick, f"{color}:pick_retreat")),
            MissionAction(ActionKind.MOVE, f"{color}:place_transit", path.transit(place, f"{color}:place_transit")),
            MissionAction(ActionKind.CHECK_OBJECT, f"{color}:check_before_place"),
            MissionAction(ActionKind.MOVE, f"{color}:place_approach", path.approach(place, f"{color}:place_approach")),
            MissionAction(ActionKind.CHECK_OBJECT, f"{color}:check_before_contact"),
            MissionAction(ActionKind.MOVE, f"{color}:place_contact", path.contact(place, f"{color}:place_contact")),
            MissionAction(ActionKind.OPEN, f"{color}:gripper_open"),
            MissionAction(ActionKind.DWELL, f"{color}:open_settle", seconds=grip),
            MissionAction(ActionKind.MOVE, f"{color}:release_approach", path.approach(place, f"{color}:release_approach")),
            MissionAction(ActionKind.MOVE, f"{color}:release_retreat", path.transit(place, f"{color}:release_retreat")),
            MissionAction(ActionKind.DWELL, f"{color}:release_stability", seconds=released),
        ]
        if self.placement_verifier is not None:
            actions.append(MissionAction(ActionKind.VISUAL_ACCEPT, f"{color}:visual_acceptance"))
        return actions

    def _execute_action(self, action: MissionAction) -> None:
        if action.kind is ActionKind.MOVE:
            if action.waypoint is None:
                raise MissionError(f"动作 {action.label} 缺少位姿")
            if not self.robot.move_l_without_tcp_force_guard(
                action.waypoint,
                self._speed_for(action.waypoint.profile),
                self.config.robot.move_acceleration,
            ):
                raise MissionError(f"moveL 返回失败: {action.label}")
        elif action.kind is ActionKind.OPEN:
            if not self.gripper.open():
                raise MissionError(f"夹爪打开失败: {action.label}")
        elif action.kind is ActionKind.CLOSE:
            if not self.gripper.close():
                raise MissionError(f"夹爪闭合失败: {action.label}")
        elif action.kind is ActionKind.CHECK_OBJECT:
            has_object = self.gripper.has_object()
            if has_object is False:
                raise MissionError(f"夹爪确认未抓到电池: {action.label}")
            if has_object is None:
                if self.robot.is_real:
                    raise MissionError(f"真机夹爪未返回物体检测反馈: {action.label}")
                LOGGER.warning("%s: 夹爪无物体检测反馈", action.label)
        elif action.kind is ActionKind.DWELL:
            if action.seconds < 0:
                raise MissionError(f"等待时间为负: {action.label}")
            self.sleep(action.seconds)
        elif action.kind is ActionKind.VISUAL_ACCEPT:
            if self.placement_verifier is None:
                raise MissionError(f"缺少视觉验收回调: {action.label}")
            color = action.label.split(":", 1)[0]
            if not self.placement_verifier(color):
                raise MissionError(f"T0 视觉验收失败: {action.label}")

    def _speed_for(self, profile: str) -> float:
        speeds = {
            "transit": self.config.robot.transit_speed,
            "approach": self.config.robot.approach_speed,
            "contact": self.config.robot.contact_speed,
        }
        try:
            return speeds[profile]
        except KeyError as exc:
            raise MissionError(f"未知运动速度档: {profile}") from exc

    def _pose(self, name: str) -> Pose:
        pose = self.config.robot.taught_poses.get(name)
        if pose is None:
            raise MissionError(f"示教位姿 {name} 仍为 null")
        return pose

    def _dynamic_pick_pose(
        self,
        color: str,
        centroids: dict[str, tuple[float, float]],
        base_xy_by_color: dict[str, tuple[float, float]] | None = None,
    ) -> Pose:
        try:
            u, v = centroids[color]
        except KeyError as exc:
            raise MissionError(f"四槽中未找到 {color} 的可靠质心") from exc
        try:
            if base_xy_by_color is not None:
                x, y = base_xy_by_color[color]
            else:
                if self.mapping is None:
                    self.mapping = HomographyModel(self.config.vision.homography)
                x, y = self.mapping.map_pixel_to_robot(u, v)
            self.config.require_dynamic_pick_xy(x, y)
        except (DetectionError, ConfigError) as exc:
            raise MissionError(f"{color} 图像坐标无法生成安全抓取 XY: {exc}") from exc
        template = self.config.robot.pick_pose_template
        if template is None:
            raise MissionError("robot.pick_pose_template 尚未标定")
        return Pose(x, y, template.z, template.rx, template.ry, template.rz)

    def _validate_dynamic_pick_pose(self, pose: Pose) -> None:
        try:
            self.config.require_dynamic_pick_xy(pose.x, pose.y)
        except ConfigError as exc:
            raise MissionError(f"动态抓取点安全校验失败: {exc}") from exc
        template = self.config.robot.pick_pose_template
        if template is None:
            raise MissionError("robot.pick_pose_template 尚未标定")
        expected = (template.z, template.rx, template.ry, template.rz)
        actual = (pose.z, pose.rx, pose.ry, pose.rz)
        if any(not math.isclose(a, b, abs_tol=1e-12) for a, b in zip(actual, expected)):
            raise MissionError("动态抓取点的 Z 或工具姿态不符合 pick_pose_template")

    def _require_dynamic_pick_separation(self, poses: tuple[Pose, ...]) -> None:
        for index, pose in enumerate(poses):
            self._validate_dynamic_pick_pose(pose)
            for other in poses[index + 1 :]:
                separation = math.hypot(pose.x - other.x, pose.y - other.y)
                if separation < self.config.robot.min_contact_xy_separation:
                    raise MissionError(
                        f"两个图像抓取点仅相距 {separation:.4f} m，拒绝规划"
                    )

    def _validate_dynamic_plan(self, plan: MissionPlan) -> None:
        contacts = tuple(
            action.waypoint.pose
            for action in plan.actions
            if action.kind is ActionKind.MOVE
            and action.waypoint is not None
            and action.waypoint.profile == "contact"
            and action.label.endswith(":pick_contact")
        )
        if contacts != plan.dynamic_pick_poses:
            raise MissionError("任务中的动态抓取点声明与 pick_contact 动作不一致")
        self._require_dynamic_pick_separation(plan.dynamic_pick_poses)
