"""严格配置加载与分阶段安全校验。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

COLORS = frozenset({"red", "blue", "yellow", "green"})
POSE_NAMES = frozenset({"home", "T0", "P1", "P2", "P3"})


class ConfigError(ValueError):
    """配置格式错误或尚不具备运行条件。"""


def _dict(raw: Any, where: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ConfigError(f"{where} 必须是对象，实际为 {type(raw).__name__}")
    return raw


def _keys(raw: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise ConfigError(f"{where} 含未知字段: {sorted(unknown)}")


def _bool(raw: Any, where: str) -> bool:
    if type(raw) is not bool:
        raise ConfigError(f"{where} 必须是 true/false")
    return raw


def _int(raw: Any, where: str) -> int:
    if type(raw) is not int:
        raise ConfigError(f"{where} 必须是整数")
    return raw


def _float(raw: Any, where: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ConfigError(f"{where} 必须是数值")
    value = float(raw)
    if not math.isfinite(value):
        raise ConfigError(f"{where} 必须是有限数值")
    return value


def _optional_float(raw: Any, where: str) -> float | None:
    return None if raw is None else _float(raw, where)


@dataclass(frozen=True)
class Pose:
    """UR 工具位姿，单位依次为 m、rad。"""

    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float

    @classmethod
    def parse(cls, raw: Any, where: str) -> "Pose":
        if not isinstance(raw, list) or len(raw) != 6:
            raise ConfigError(f"{where} 必须是 6 个数的 [x,y,z,rx,ry,rz]")
        values = [_float(value, f"{where}[{index}]") for index, value in enumerate(raw)]
        if any(abs(value) > 2.0 for value in values[:3]):
            raise ConfigError(f"{where} 平移分量超出 ±2 m 保守边界")
        if any(abs(value) > 2.0 * math.pi for value in values[3:]):
            raise ConfigError(f"{where} 旋转向量分量超出 ±2π rad 保守边界")
        return cls(*values)

    def as_list(self) -> list[float]:
        return [self.x, self.y, self.z, self.rx, self.ry, self.rz]

    def with_z(self, z: float) -> "Pose":
        return Pose(self.x, self.y, z, self.rx, self.ry, self.rz)


@dataclass(frozen=True)
class CameraConfig:
    device_id: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    exposure: float | None = None
    gain: float | None = None
    white_balance_temperature: float | None = None
    focus: float | None = None


@dataclass(frozen=True)
class HSVRange:
    h_min: int
    h_max: int
    s_min: int
    s_max: int
    v_min: int
    v_max: int


@dataclass(frozen=True)
class ColorThreshold:
    name: str
    ranges: tuple[HSVRange, ...]


@dataclass(frozen=True)
class SlotROI:
    slot_name: str
    x: int
    y: int
    width: int
    height: int
    expected_center_x: float
    expected_center_y: float
    max_centroid_offset_px: float
    expected_component_area_px: float | None = None
    expected_long_axis_angle_deg: float | None = None
    area_tolerance_fraction: float = 0.30
    max_angle_error_deg: float = 5.0


@dataclass(frozen=True)
class HomographyConfig:
    """Pixel-to-UR-base planar calibration point pairs."""

    pixel_points: tuple[tuple[float, float], ...] = ()
    robot_xy_points_m: tuple[tuple[float, float], ...] = ()
    calibrated: bool = False
    max_reprojection_error_m: float = 0.002


@dataclass(frozen=True)
class VisionConfig:
    rois: tuple[SlotROI, ...] = ()
    colors: tuple[ColorThreshold, ...] = ()
    homography: HomographyConfig = field(default_factory=HomographyConfig)
    min_coverage: float = 0.10
    min_confidence: float = 0.75
    min_margin: float = 0.25
    morph_kernel_size: int = 5
    morph_iterations: int = 1
    sample_frames: int = 5
    required_consistent_frames: int = 4
    calibrated: bool = False


@dataclass(frozen=True)
class WorkspaceBounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float


@dataclass(frozen=True)
class PickXYBounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float


@dataclass(frozen=True)
class PickPoseTemplate:
    """Calibrated pick contact Z and UR axis-angle; XY comes from vision."""

    z: float
    rx: float
    ry: float
    rz: float


@dataclass(frozen=True)
class RobotConfig:
    host: str = "192.168.0.10"
    dashboard_port: int = 29999
    rtde_port: int = 30004
    tcp_calibrated: bool = False
    payload_calibrated: bool = False
    base_z_up_confirmed: bool = False
    safe_z_approach: float = 0.0
    safe_z_transit: float = 0.0
    min_approach_clearance: float = 0.03
    min_transit_clearance: float = 0.05
    max_orientation_delta: float = math.radians(5.0)
    home_translation_tolerance: float = 0.001
    home_rotation_tolerance: float = math.radians(0.5)
    segment_start_translation_tolerance: float = 0.0002
    segment_start_rotation_tolerance: float = math.radians(0.1)
    min_contact_xy_separation: float = 0.01
    workspace_bounds: WorkspaceBounds | None = None
    pick_xy_bounds: PickXYBounds | None = None
    pick_pose_template: PickPoseTemplate | None = None
    max_horizontal_segment_m: float = 0.75
    max_vertical_segment_m: float = 0.25
    transit_speed: float = 0.05
    approach_speed: float = 0.02
    contact_speed: float = 0.005
    move_acceleration: float = 0.10
    stop_deceleration: float = 0.50
    max_stop_distance_m: float = 0.005
    move_timeout: float = 20.0
    monitor_rate: float = 50.0
    max_tcp_force_delta_n: float = 20.0
    grip_settle_seconds: float = 0.5
    lifted_hold_seconds: float = 2.1
    release_settle_seconds: float = 3.1
    taught_poses: dict[str, Pose | None] = field(default_factory=dict)


@dataclass(frozen=True)
class GripperConfig:
    driver: str = "not_configured"
    port: str = "COM11"
    baud: int = 115200
    slave_id: int = 1
    force_percent: int = 60
    speed_percent: int = 50
    open_position: int = 1000
    close_position: int = 0
    position_tolerance: int = 20
    serial_timeout: float = 0.5
    command_timeout: float = 5.0
    poll_interval: float = 0.05
    read_retries: int = 1


@dataclass(frozen=True)
class BatterySortingConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    gripper: GripperConfig = field(default_factory=GripperConfig)
    dry_run: bool = True

    @classmethod
    def from_json(cls, path: str | Path) -> "BatterySortingConfig":
        try:
            with Path(path).open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"无法读取配置 {path}: {exc}") from exc
        top = _dict(raw, "配置根节点")
        _keys(top, {"camera", "vision", "robot", "gripper", "dry_run"}, "配置根节点")
        return cls(
            camera=_parse_camera(_dict(top.get("camera", {}), "camera")),
            vision=_parse_vision(_dict(top.get("vision", {}), "vision")),
            robot=_parse_robot(_dict(top.get("robot", {}), "robot")),
            gripper=_parse_gripper(_dict(top.get("gripper", {}), "gripper")),
            dry_run=_bool(top.get("dry_run", True), "dry_run"),
        )

    def validate_vision(self) -> list[str]:
        errors: list[str] = []
        camera = self.camera
        if camera.device_id < 0:
            errors.append("camera.device_id 不能为负")
        if not (1 <= camera.width <= 7680 and 1 <= camera.height <= 4320):
            errors.append("camera.width/height 超出合理范围")
        if not (1 <= camera.fps <= 240):
            errors.append("camera.fps 必须在 1..240")

        rois = self.vision.rois
        if len(rois) != 4:
            errors.append(f"vision.rois 必须恰好有 4 个，实际 {len(rois)}")
        slot_names = [roi.slot_name for roi in rois]
        expected_slots = {"slot1", "slot2", "slot3", "slot4"}
        if set(slot_names) != expected_slots or len(slot_names) != len(expected_slots):
            errors.append("vision.rois 必须用 slot_name 显式配置 slot1..slot4 各一次")
        for index, roi in enumerate(rois):
            if roi.x < 0 or roi.y < 0 or roi.width <= 0 or roi.height <= 0:
                errors.append(f"vision.rois[{index}] 原点或尺寸无效")
            if roi.x + roi.width > camera.width or roi.y + roi.height > camera.height:
                errors.append(f"vision.rois[{index}] 超出相机画面")
            if not (
                roi.x <= roi.expected_center_x < roi.x + roi.width
                and roi.y <= roi.expected_center_y < roi.y + roi.height
            ):
                errors.append(f"vision.rois[{index}] 的期望质心不在 ROI 内")
            if roi.max_centroid_offset_px <= 0:
                errors.append(f"vision.rois[{index}].max_centroid_offset_px 必须为正")
            if roi.max_centroid_offset_px > math.hypot(roi.width, roi.height) / 2.0:
                errors.append(f"vision.rois[{index}] 的质心偏移阈值过大")
            if roi.expected_component_area_px is not None:
                if roi.expected_component_area_px <= 0:
                    errors.append(
                        f"vision.rois[{index}].expected_component_area_px 必须为正"
                    )
                elif roi.expected_component_area_px > roi.width * roi.height:
                    errors.append(
                        f"vision.rois[{index}].expected_component_area_px 超过 ROI 面积"
                    )
            if roi.expected_long_axis_angle_deg is not None and not (
                -90.0 <= roi.expected_long_axis_angle_deg < 90.0
            ):
                errors.append(
                    f"vision.rois[{index}].expected_long_axis_angle_deg 必须在 [-90,90)"
                )
            if not (0.0 < roi.area_tolerance_fraction <= 0.50):
                errors.append(
                    f"vision.rois[{index}].area_tolerance_fraction 必须在 (0,0.50]"
                )
            if not (0.0 < roi.max_angle_error_deg <= 15.0):
                errors.append(
                    f"vision.rois[{index}].max_angle_error_deg 必须在 (0,15]"
                )
        for first in range(len(rois)):
            for second in range(first + 1, len(rois)):
                if _overlap(rois[first], rois[second]):
                    errors.append(f"vision.rois[{first}] 与 rois[{second}] 重叠")

        homography = self.vision.homography
        if not (0.0 < homography.max_reprojection_error_m <= 0.01):
            errors.append("vision.homography.max_reprojection_error_m 必须在 (0,0.01] m")
        point_count = len(homography.pixel_points)
        if homography.calibrated and point_count < 4:
            errors.append("vision.homography 至少需要 4 个标定点对")
        if point_count != len(homography.robot_xy_points_m):
            errors.append("vision.homography 像素点与机器人 XY 点数量不一致")
        if point_count:
            if len(set(homography.pixel_points)) != point_count:
                errors.append("vision.homography.pixel_points 存在重复点")
            if len(set(homography.robot_xy_points_m)) != len(
                homography.robot_xy_points_m
            ):
                errors.append("vision.homography.robot_xy_points_m 存在重复点")
            for index, (u, v) in enumerate(homography.pixel_points):
                if not (0.0 <= u < camera.width and 0.0 <= v < camera.height):
                    errors.append(
                        f"vision.homography.pixel_points[{index}] 超出相机画面"
                    )
            for index, (x, y) in enumerate(homography.robot_xy_points_m):
                if abs(x) > 2.0 or abs(y) > 2.0:
                    errors.append(
                        f"vision.homography.robot_xy_points_m[{index}] 超出 ±2 m 保守边界"
                    )
            if point_count >= 3:
                if _max_triangle_double_area(homography.pixel_points) < 1.0:
                    errors.append("vision.homography.pixel_points 共线或覆盖面积过小")
                if _max_triangle_double_area(homography.robot_xy_points_m) < 1e-8:
                    errors.append("vision.homography.robot_xy_points_m 共线或覆盖面积过小")

        names = [color.name for color in self.vision.colors]
        if set(names) != COLORS or len(names) != len(COLORS):
            errors.append(f"vision.colors 必须且只能配置 {sorted(COLORS)} 各一次")
        for color in self.vision.colors:
            if not color.ranges:
                errors.append(f"颜色 {color.name} 没有 HSV 范围")
            if color.name == "red" and len(color.ranges) != 2:
                errors.append("red 必须有色环首尾两个 HSV 范围")
            for index, hsv in enumerate(color.ranges):
                values = (hsv.h_min, hsv.h_max, hsv.s_min, hsv.s_max, hsv.v_min, hsv.v_max)
                if not (0 <= hsv.h_min <= hsv.h_max <= 179):
                    errors.append(f"{color.name}.ranges[{index}] 的 H 范围无效")
                if not (0 <= hsv.s_min <= hsv.s_max <= 255):
                    errors.append(f"{color.name}.ranges[{index}] 的 S 范围无效")
                if not (0 <= hsv.v_min <= hsv.v_max <= 255):
                    errors.append(f"{color.name}.ranges[{index}] 的 V 范围无效")
                if any(type(value) is not int for value in values):
                    errors.append(f"{color.name}.ranges[{index}] 必须全为整数")

        if not (0.0 < self.vision.min_coverage <= 1.0):
            errors.append("vision.min_coverage 必须在 (0,1]")
        if not (0.50 <= self.vision.min_confidence <= 1.0):
            errors.append("vision.min_confidence 必须在 [0.50,1]")
        if not (0.0 < self.vision.min_margin <= 1.0):
            errors.append("vision.min_margin 必须在 (0,1]")
        kernel = self.vision.morph_kernel_size
        if kernel < 1 or kernel > 31 or kernel % 2 == 0:
            errors.append("vision.morph_kernel_size 必须是 1..31 的奇数")
        if rois and kernel > min(min(roi.width, roi.height) for roi in rois) // 3:
            errors.append("vision.morph_kernel_size 相对 ROI 尺寸过大")
        if not (1 <= self.vision.morph_iterations <= 10):
            errors.append("vision.morph_iterations 必须在 1..10")
        if not (1 <= self.vision.sample_frames <= 30):
            errors.append("vision.sample_frames 必须在 1..30")
        if not (1 <= self.vision.required_consistent_frames <= self.vision.sample_frames):
            errors.append("required_consistent_frames 必须在 1..sample_frames")
        return errors

    def require_vision_valid(self) -> None:
        _raise_validation(self.validate_vision(), "视觉配置")

    def validate_planning(self) -> list[str]:
        errors = self.validate_vision()
        if not self.vision.homography.calibrated:
            errors.append("vision.homography.calibrated=false，无法由图像生成抓取 XY")
        poses = self.robot.taught_poses
        missing = POSE_NAMES - set(poses)
        unknown = set(poses) - POSE_NAMES
        if missing:
            errors.append(f"缺少示教位姿: {sorted(missing)}")
        if unknown:
            errors.append(f"未知示教位姿名: {sorted(unknown)}")
        for name in POSE_NAMES:
            if poses.get(name) is None:
                errors.append(f"示教位姿 {name} 仍为 null")

        robot = self.robot
        if not (1 <= robot.dashboard_port <= 65535 and 1 <= robot.rtde_port <= 65535):
            errors.append("robot.dashboard_port/rtde_port 必须在 1..65535")
        if robot.rtde_port != 30004:
            errors.append("ur_rtde 1.6.3 固定使用 RTDE 端口 30004，robot.rtde_port 必须为 30004")
        if robot.min_approach_clearance <= 0 or robot.min_transit_clearance <= 0:
            errors.append("robot 的两个安全高度余量必须为正")
        if robot.max_orientation_delta <= 0:
            errors.append("robot.max_orientation_delta 必须为正")
        if robot.min_contact_xy_separation <= 0:
            errors.append("robot.min_contact_xy_separation 必须为正")
        if robot.home_translation_tolerance <= 0 or robot.home_translation_tolerance > 0.005:
            errors.append("home_translation_tolerance 必须在 (0,0.005] m")
        if robot.home_rotation_tolerance <= 0 or robot.home_rotation_tolerance > math.radians(2.0):
            errors.append("home_rotation_tolerance 必须在 (0,2°] rad")
        if not (0.0 < robot.segment_start_translation_tolerance <= 0.0002):
            errors.append("segment_start_translation_tolerance 必须在 (0,0.0002] m")
        if not (0.0 < robot.segment_start_rotation_tolerance <= math.radians(0.1)):
            errors.append("segment_start_rotation_tolerance 必须在 (0,0.1°] rad")
        if not (0.0 < robot.max_horizontal_segment_m <= 1.0):
            errors.append("max_horizontal_segment_m 必须在 (0,1.0] m")
        if not (0.0 < robot.max_vertical_segment_m <= 0.5):
            errors.append("max_vertical_segment_m 必须在 (0,0.5] m")
        if not (0.0 < robot.transit_speed <= 0.10):
            errors.append("robot.transit_speed 必须在 (0,0.10] m/s")
        if not (0.0 < robot.approach_speed <= 0.05):
            errors.append("robot.approach_speed 必须在 (0,0.05] m/s")
        if not (0.0 < robot.contact_speed <= 0.01):
            errors.append("robot.contact_speed 必须在 (0,0.01] m/s")
        if not robot.contact_speed <= robot.approach_speed <= robot.transit_speed:
            errors.append("速度必须满足 contact <= approach <= transit")
        if not (0.0 < robot.move_acceleration <= 0.50):
            errors.append("robot.move_acceleration 必须在 (0,0.50] m/s²")
        if not (0.0 < robot.stop_deceleration <= 1.5):
            errors.append("robot.stop_deceleration 必须在 (0,1.5] m/s²")
        if not (0.0 < robot.max_stop_distance_m <= 0.01):
            errors.append("robot.max_stop_distance_m 必须在 (0,0.01] m")
        if robot.stop_deceleration > 0:
            reaction_distance = robot.transit_speed / robot.monitor_rate if robot.monitor_rate > 0 else math.inf
            estimated_stop = (
                reaction_distance
                + robot.transit_speed**2 / (2.0 * robot.stop_deceleration)
            )
            if estimated_stop > robot.max_stop_distance_m:
                errors.append(
                    f"最坏轮询反应+理论制动距离 {estimated_stop:.4f}m 超过 max_stop_distance_m"
                )
        if not (0.1 <= robot.move_timeout <= 120.0):
            errors.append("robot.move_timeout 必须在 0.1..120 秒")
        if not (50.0 <= robot.monitor_rate <= 200.0):
            errors.append("robot.monitor_rate 必须在 50..200 Hz")
        if not (0.0 < robot.max_tcp_force_delta_n <= 60.0):
            errors.append("robot.max_tcp_force_delta_n 必须在 (0,60] N")
        if not (0.0 <= robot.grip_settle_seconds <= 10.0):
            errors.append("robot.grip_settle_seconds 必须在 0..10 秒")
        if not (2.0 <= robot.lifted_hold_seconds <= 30.0):
            errors.append("robot.lifted_hold_seconds 必须在 2..30 秒")
        if not (3.0 <= robot.release_settle_seconds <= 30.0):
            errors.append("robot.release_settle_seconds 必须在 3..30 秒")

        complete = {name: pose for name, pose in poses.items() if pose is not None}
        template = robot.pick_pose_template
        if template is None:
            errors.append("robot.pick_pose_template 仍为 null")
        elif any(abs(value) > 2.0 * math.pi for value in (template.rx, template.ry, template.rz)):
            errors.append("robot.pick_pose_template 旋转向量分量超出 ±2π rad 保守边界")

        pick_bounds = robot.pick_xy_bounds
        if pick_bounds is None:
            errors.append("robot.pick_xy_bounds 仍为 null")
        elif not (
            pick_bounds.x_min < pick_bounds.x_max
            and pick_bounds.y_min < pick_bounds.y_max
        ):
            errors.append("robot.pick_xy_bounds 的 min 必须小于 max")
        elif any(
            abs(value) > 2.0
            for value in (
                pick_bounds.x_min,
                pick_bounds.x_max,
                pick_bounds.y_min,
                pick_bounds.y_max,
            )
        ):
            errors.append("robot.pick_xy_bounds 超出 ±2 m 保守边界")

        contact_poses = [
            complete[name] for name in POSE_NAMES - {"home"} if name in complete
        ]
        if template is not None:
            contact_poses.append(
                Pose(0.0, 0.0, template.z, template.rx, template.ry, template.rz)
            )
        if contact_poses:
            highest_contact = max(pose.z for pose in contact_poses)
            if robot.safe_z_approach < highest_contact + robot.min_approach_clearance:
                errors.append("safe_z_approach 未高出所有接触位姿所需余量")
            if robot.safe_z_transit < robot.safe_z_approach + robot.min_transit_clearance:
                errors.append("safe_z_transit 未高出 approach 所需余量")
            if any(abs(z) > 2.0 for z in (robot.safe_z_approach, robot.safe_z_transit)):
                errors.append("safe_z_approach/safe_z_transit 超出 ±2 m 保守边界")
            contact_items = [
                (name, complete[name]) for name in sorted(POSE_NAMES - {"home"}) if name in complete
            ]
            for first in range(len(contact_items)):
                for second in range(first + 1, len(contact_items)):
                    first_name, first_pose = contact_items[first]
                    second_name, second_pose = contact_items[second]
                    separation = math.hypot(first_pose.x - second_pose.x, first_pose.y - second_pose.y)
                    if separation < robot.min_contact_xy_separation:
                        errors.append(
                            f"示教点 {first_name}/{second_name} 的 XY 仅相距 {separation:.4f} m"
                        )
        home = complete.get("home")
        if home is not None and not math.isclose(home.z, robot.safe_z_transit, abs_tol=1e-6):
            errors.append("home.z 必须等于 safe_z_transit")
        workspace = robot.workspace_bounds
        if workspace is None:
            errors.append("robot.workspace_bounds 仍为 null，必须按现场工作区测量")
        elif not (
            workspace.x_min < workspace.x_max
            and workspace.y_min < workspace.y_max
            and workspace.z_min < workspace.z_max
        ):
            errors.append("robot.workspace_bounds 的 min 必须小于 max")
        else:
            for name, pose in complete.items():
                if not _pose_in_workspace(pose, workspace):
                    errors.append(f"示教位姿 {name} 超出 workspace_bounds")
            if not (workspace.z_min <= robot.safe_z_approach <= workspace.z_max):
                errors.append("safe_z_approach 超出 workspace_bounds.z")
            if not (workspace.z_min <= robot.safe_z_transit <= workspace.z_max):
                errors.append("safe_z_transit 超出 workspace_bounds.z")
            if template is not None and not (workspace.z_min <= template.z <= workspace.z_max):
                errors.append("robot.pick_pose_template.z 超出 workspace_bounds.z")
            if pick_bounds is not None and not (
                workspace.x_min <= pick_bounds.x_min < pick_bounds.x_max <= workspace.x_max
                and workspace.y_min <= pick_bounds.y_min < pick_bounds.y_max <= workspace.y_max
            ):
                errors.append("robot.pick_xy_bounds 必须完全位于 workspace_bounds 内")
        complete_items = sorted(complete.items())
        for first in range(len(complete_items)):
            for second in range(first + 1, len(complete_items)):
                first_name, first_pose = complete_items[first]
                second_name, second_pose = complete_items[second]
                delta = _rotation_distance(first_pose, second_pose)
                if delta > robot.max_orientation_delta:
                    errors.append(f"{first_name}/{second_name} 工具朝向相差 {math.degrees(delta):.2f}°")
        if template is not None:
            template_pose = Pose(
                0.0, 0.0, template.z, template.rx, template.ry, template.rz
            )
            for name, pose in complete_items:
                delta = _rotation_distance(template_pose, pose)
                if delta > robot.max_orientation_delta:
                    errors.append(
                        f"pick_pose_template/{name} 工具朝向相差 "
                        f"{math.degrees(delta):.2f}°"
                    )
        return errors

    def require_planning_valid(self) -> None:
        _raise_validation(self.validate_planning(), "离线规划配置")

    def require_dynamic_pick_xy(self, x: float, y: float) -> None:
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ConfigError("图像映射得到非有限抓取 XY")
        bounds = self.robot.pick_xy_bounds
        if bounds is None:
            raise ConfigError("robot.pick_xy_bounds 尚未配置")
        if not (
            bounds.x_min <= x <= bounds.x_max
            and bounds.y_min <= y <= bounds.y_max
        ):
            raise ConfigError(
                f"图像映射抓取 XY ({x:.4f}, {y:.4f}) 超出 pick_xy_bounds"
            )
        workspace = self.robot.workspace_bounds
        if workspace is None:
            raise ConfigError("robot.workspace_bounds 尚未配置")
        if not (
            workspace.x_min <= x <= workspace.x_max
            and workspace.y_min <= y <= workspace.y_max
        ):
            raise ConfigError(
                f"图像映射抓取 XY ({x:.4f}, {y:.4f}) 超出 workspace_bounds"
            )

    def validate_execution(
        self, *, gripper_configured: bool, gripper_is_mock: bool,
        gripper_has_object_feedback: bool,
    ) -> list[str]:
        errors = self.validate_planning()
        if self.dry_run:
            errors.append("dry_run=true，禁止真机执行")
        if not self.robot.tcp_calibrated:
            errors.append("robot.tcp_calibrated=false")
        if not self.robot.payload_calibrated:
            errors.append("robot.payload_calibrated=false")
        if not self.robot.base_z_up_confirmed:
            errors.append("robot.base_z_up_confirmed=false")
        if self.camera.exposure is None:
            errors.append("真机执行前必须标定并锁定 camera.exposure")
        if self.camera.white_balance_temperature is None:
            errors.append("真机执行前必须标定并锁定 camera.white_balance_temperature")
        if not self.vision.calibrated:
            errors.append("vision.calibrated=false，尚未确认现场 ROI/HSV/质心阈值")
        for roi in self.vision.rois:
            if roi.expected_component_area_px is None:
                errors.append(
                    f"真机执行要求 {roi.slot_name}.expected_component_area_px 已现场标定"
                )
            if roi.expected_long_axis_angle_deg is None:
                errors.append(
                    f"真机执行要求 {roi.slot_name}.expected_long_axis_angle_deg 已现场标定"
                )
        if self.vision.sample_frames < 3:
            errors.append("真机执行要求 vision.sample_frames >= 3")
        if self.vision.required_consistent_frames <= self.vision.sample_frames // 2:
            errors.append("真机执行要求 required_consistent_frames 为严格多数")
        if not gripper_configured or gripper_is_mock:
            errors.append("未接入并验证真实夹爪驱动")
        if not gripper_has_object_feedback:
            errors.append("真实夹爪必须提供可靠的 has_object 夹持反馈")
        return errors

    def require_execution_valid(
        self, *, gripper_configured: bool, gripper_is_mock: bool,
        gripper_has_object_feedback: bool,
    ) -> None:
        _raise_validation(
            self.validate_execution(
                gripper_configured=gripper_configured,
                gripper_is_mock=gripper_is_mock,
                gripper_has_object_feedback=gripper_has_object_feedback,
            ),
            "真机配置",
        )


def _parse_camera(raw: dict[str, Any]) -> CameraConfig:
    allowed = {
        "device_id", "width", "height", "fps", "exposure", "gain",
        "white_balance_temperature", "focus",
    }
    _keys(raw, allowed, "camera")
    return CameraConfig(
        device_id=_int(raw.get("device_id", 0), "camera.device_id"),
        width=_int(raw.get("width", 1280), "camera.width"),
        height=_int(raw.get("height", 720), "camera.height"),
        fps=_int(raw.get("fps", 30), "camera.fps"),
        exposure=_optional_float(raw.get("exposure"), "camera.exposure"),
        gain=_optional_float(raw.get("gain"), "camera.gain"),
        white_balance_temperature=_optional_float(
            raw.get("white_balance_temperature"), "camera.white_balance_temperature"
        ),
        focus=_optional_float(raw.get("focus"), "camera.focus"),
    )


def _parse_xy_points(raw: Any, where: str) -> tuple[tuple[float, float], ...]:
    if not isinstance(raw, list):
        raise ConfigError(f"{where} 必须是二维点数组")
    points: list[tuple[float, float]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, list) or len(item) != 2:
            raise ConfigError(f"{where}[{index}] 必须是 [x,y]")
        points.append(
            (
                _float(item[0], f"{where}[{index}][0]"),
                _float(item[1], f"{where}[{index}][1]"),
            )
        )
    return tuple(points)


def _parse_vision(raw: dict[str, Any]) -> VisionConfig:
    allowed = {
        "rois", "colors", "min_coverage", "min_confidence", "min_margin",
        "morph_kernel_size", "morph_iterations", "sample_frames",
        "required_consistent_frames", "calibrated", "homography",
    }
    _keys(raw, allowed, "vision")
    homography_raw = _dict(raw.get("homography", {}), "vision.homography")
    _keys(
        homography_raw,
        {
            "pixel_points",
            "robot_xy_points_m",
            "calibrated",
            "max_reprojection_error_m",
        },
        "vision.homography",
    )
    homography = HomographyConfig(
        pixel_points=_parse_xy_points(
            homography_raw.get("pixel_points", []),
            "vision.homography.pixel_points",
        ),
        robot_xy_points_m=_parse_xy_points(
            homography_raw.get("robot_xy_points_m", []),
            "vision.homography.robot_xy_points_m",
        ),
        calibrated=_bool(
            homography_raw.get("calibrated", False),
            "vision.homography.calibrated",
        ),
        max_reprojection_error_m=_float(
            homography_raw.get("max_reprojection_error_m", 0.002),
            "vision.homography.max_reprojection_error_m",
        ),
    )
    rois_raw = raw.get("rois", [])
    if not isinstance(rois_raw, list):
        raise ConfigError("vision.rois 必须是数组")
    rois: list[SlotROI] = []
    for index, item in enumerate(rois_raw):
        item = _dict(item, f"vision.rois[{index}]")
        required = {
            "slot_name", "x", "y", "width", "height", "expected_center_x",
            "expected_center_y", "max_centroid_offset_px",
        }
        optional = {
            "expected_component_area_px", "expected_long_axis_angle_deg",
            "area_tolerance_fraction", "max_angle_error_deg",
        }
        _keys(item, required | optional, f"vision.rois[{index}]")
        missing = required - set(item)
        if missing:
            raise ConfigError(f"vision.rois[{index}] 缺少必填字段: {sorted(missing)}")
        if not isinstance(item["slot_name"], str):
            raise ConfigError(f"vision.rois[{index}].slot_name 必须是字符串")
        rois.append(SlotROI(
            slot_name=item["slot_name"],
            x=_int(item["x"], f"vision.rois[{index}].x"),
            y=_int(item["y"], f"vision.rois[{index}].y"),
            width=_int(item["width"], f"vision.rois[{index}].width"),
            height=_int(item["height"], f"vision.rois[{index}].height"),
            expected_center_x=_float(item["expected_center_x"], f"vision.rois[{index}].expected_center_x"),
            expected_center_y=_float(item["expected_center_y"], f"vision.rois[{index}].expected_center_y"),
            max_centroid_offset_px=_float(item["max_centroid_offset_px"], f"vision.rois[{index}].max_centroid_offset_px"),
            expected_component_area_px=_optional_float(
                item.get("expected_component_area_px"),
                f"vision.rois[{index}].expected_component_area_px",
            ),
            expected_long_axis_angle_deg=_optional_float(
                item.get("expected_long_axis_angle_deg"),
                f"vision.rois[{index}].expected_long_axis_angle_deg",
            ),
            area_tolerance_fraction=_float(
                item.get("area_tolerance_fraction", 0.30),
                f"vision.rois[{index}].area_tolerance_fraction",
            ),
            max_angle_error_deg=_float(
                item.get("max_angle_error_deg", 5.0),
                f"vision.rois[{index}].max_angle_error_deg",
            ),
        ))

    colors_raw = raw.get("colors", [])
    if not isinstance(colors_raw, list):
        raise ConfigError("vision.colors 必须是数组")
    colors: list[ColorThreshold] = []
    hsv_fields = {"h_min", "h_max", "s_min", "s_max", "v_min", "v_max"}
    for color_index, item in enumerate(colors_raw):
        item = _dict(item, f"vision.colors[{color_index}]")
        _keys(item, {"name", "ranges"}, f"vision.colors[{color_index}]")
        if set(item) != {"name", "ranges"} or not isinstance(item["name"], str):
            raise ConfigError(f"vision.colors[{color_index}] 必须含字符串 name 和 ranges")
        ranges_raw = item["ranges"]
        if not isinstance(ranges_raw, list):
            raise ConfigError(f"vision.colors[{color_index}].ranges 必须是数组")
        ranges: list[HSVRange] = []
        for range_index, value in enumerate(ranges_raw):
            value = _dict(value, f"vision.colors[{color_index}].ranges[{range_index}]")
            _keys(value, hsv_fields, f"vision.colors[{color_index}].ranges[{range_index}]")
            if set(value) != hsv_fields:
                raise ConfigError(f"HSV 范围必须包含 {sorted(hsv_fields)}")
            ranges.append(HSVRange(**{
                key: _int(value[key], f"vision.colors[{color_index}].ranges[{range_index}].{key}")
                for key in hsv_fields
            }))
        colors.append(ColorThreshold(item["name"], tuple(ranges)))
    return VisionConfig(
        rois=tuple(rois),
        colors=tuple(colors),
        homography=homography,
        min_coverage=_float(raw.get("min_coverage", 0.10), "vision.min_coverage"),
        min_confidence=_float(raw.get("min_confidence", 0.75), "vision.min_confidence"),
        min_margin=_float(raw.get("min_margin", 0.25), "vision.min_margin"),
        morph_kernel_size=_int(raw.get("morph_kernel_size", 5), "vision.morph_kernel_size"),
        morph_iterations=_int(raw.get("morph_iterations", 1), "vision.morph_iterations"),
        sample_frames=_int(raw.get("sample_frames", 5), "vision.sample_frames"),
        required_consistent_frames=_int(
            raw.get("required_consistent_frames", 4), "vision.required_consistent_frames"
        ),
        calibrated=_bool(raw.get("calibrated", False), "vision.calibrated"),
    )


def _parse_robot(raw: dict[str, Any]) -> RobotConfig:
    defaults = RobotConfig()
    allowed = {
        "host", "dashboard_port", "rtde_port", "tcp_calibrated", "payload_calibrated",
        "base_z_up_confirmed", "safe_z_approach", "safe_z_transit",
        "min_approach_clearance", "min_transit_clearance", "max_orientation_delta",
        "home_translation_tolerance", "home_rotation_tolerance", "min_contact_xy_separation",
        "segment_start_translation_tolerance", "segment_start_rotation_tolerance",
        "workspace_bounds", "pick_xy_bounds", "pick_pose_template",
        "max_horizontal_segment_m", "max_vertical_segment_m",
        "transit_speed", "approach_speed", "contact_speed", "move_acceleration",
        "stop_deceleration", "max_stop_distance_m", "move_timeout", "monitor_rate",
        "max_tcp_force_delta_n",
        "grip_settle_seconds", "lifted_hold_seconds", "release_settle_seconds", "taught_poses",
    }
    _keys(raw, allowed, "robot")
    host = raw.get("host", defaults.host)
    if not isinstance(host, str) or not host.strip():
        raise ConfigError("robot.host 必须是非空字符串")
    poses_raw = _dict(raw.get("taught_poses", {}), "robot.taught_poses")
    _keys(poses_raw, set(POSE_NAMES), "robot.taught_poses")
    poses = {
        name: None if value is None else Pose.parse(value, f"robot.taught_poses.{name}")
        for name, value in poses_raw.items()
    }
    workspace_raw = raw.get("workspace_bounds")
    workspace: WorkspaceBounds | None
    if workspace_raw is None:
        workspace = None
    else:
        workspace_dict = _dict(workspace_raw, "robot.workspace_bounds")
        workspace_fields = {"x_min", "x_max", "y_min", "y_max", "z_min", "z_max"}
        _keys(workspace_dict, workspace_fields, "robot.workspace_bounds")
        if set(workspace_dict) != workspace_fields:
            raise ConfigError(f"robot.workspace_bounds 必须包含 {sorted(workspace_fields)}")
        workspace = WorkspaceBounds(**{
            key: _float(workspace_dict[key], f"robot.workspace_bounds.{key}")
            for key in workspace_fields
        })
    pick_bounds_raw = raw.get("pick_xy_bounds")
    pick_bounds: PickXYBounds | None
    if pick_bounds_raw is None:
        pick_bounds = None
    else:
        pick_bounds_dict = _dict(pick_bounds_raw, "robot.pick_xy_bounds")
        pick_bounds_fields = {"x_min", "x_max", "y_min", "y_max"}
        _keys(pick_bounds_dict, pick_bounds_fields, "robot.pick_xy_bounds")
        if set(pick_bounds_dict) != pick_bounds_fields:
            raise ConfigError(
                f"robot.pick_xy_bounds 必须包含 {sorted(pick_bounds_fields)}"
            )
        pick_bounds = PickXYBounds(**{
            key: _float(pick_bounds_dict[key], f"robot.pick_xy_bounds.{key}")
            for key in pick_bounds_fields
        })
    template_raw = raw.get("pick_pose_template")
    pick_template: PickPoseTemplate | None
    if template_raw is None:
        pick_template = None
    else:
        template_dict = _dict(template_raw, "robot.pick_pose_template")
        template_fields = {"z", "rx", "ry", "rz"}
        _keys(template_dict, template_fields, "robot.pick_pose_template")
        if set(template_dict) != template_fields:
            raise ConfigError(
                f"robot.pick_pose_template 必须包含 {sorted(template_fields)}"
            )
        pick_template = PickPoseTemplate(**{
            key: _float(template_dict[key], f"robot.pick_pose_template.{key}")
            for key in template_fields
        })
    return RobotConfig(
        host=host,
        dashboard_port=_int(raw.get("dashboard_port", defaults.dashboard_port), "robot.dashboard_port"),
        rtde_port=_int(raw.get("rtde_port", defaults.rtde_port), "robot.rtde_port"),
        tcp_calibrated=_bool(raw.get("tcp_calibrated", False), "robot.tcp_calibrated"),
        payload_calibrated=_bool(raw.get("payload_calibrated", False), "robot.payload_calibrated"),
        base_z_up_confirmed=_bool(raw.get("base_z_up_confirmed", False), "robot.base_z_up_confirmed"),
        safe_z_approach=_float(raw.get("safe_z_approach", 0.0), "robot.safe_z_approach"),
        safe_z_transit=_float(raw.get("safe_z_transit", 0.0), "robot.safe_z_transit"),
        min_approach_clearance=_float(raw.get("min_approach_clearance", defaults.min_approach_clearance), "robot.min_approach_clearance"),
        min_transit_clearance=_float(raw.get("min_transit_clearance", defaults.min_transit_clearance), "robot.min_transit_clearance"),
        max_orientation_delta=_float(raw.get("max_orientation_delta", defaults.max_orientation_delta), "robot.max_orientation_delta"),
        home_translation_tolerance=_float(raw.get("home_translation_tolerance", defaults.home_translation_tolerance), "robot.home_translation_tolerance"),
        home_rotation_tolerance=_float(raw.get("home_rotation_tolerance", defaults.home_rotation_tolerance), "robot.home_rotation_tolerance"),
        segment_start_translation_tolerance=_float(raw.get("segment_start_translation_tolerance", defaults.segment_start_translation_tolerance), "robot.segment_start_translation_tolerance"),
        segment_start_rotation_tolerance=_float(raw.get("segment_start_rotation_tolerance", defaults.segment_start_rotation_tolerance), "robot.segment_start_rotation_tolerance"),
        min_contact_xy_separation=_float(raw.get("min_contact_xy_separation", defaults.min_contact_xy_separation), "robot.min_contact_xy_separation"),
        workspace_bounds=workspace,
        pick_xy_bounds=pick_bounds,
        pick_pose_template=pick_template,
        max_horizontal_segment_m=_float(raw.get("max_horizontal_segment_m", defaults.max_horizontal_segment_m), "robot.max_horizontal_segment_m"),
        max_vertical_segment_m=_float(raw.get("max_vertical_segment_m", defaults.max_vertical_segment_m), "robot.max_vertical_segment_m"),
        transit_speed=_float(raw.get("transit_speed", defaults.transit_speed), "robot.transit_speed"),
        approach_speed=_float(raw.get("approach_speed", defaults.approach_speed), "robot.approach_speed"),
        contact_speed=_float(raw.get("contact_speed", defaults.contact_speed), "robot.contact_speed"),
        move_acceleration=_float(raw.get("move_acceleration", defaults.move_acceleration), "robot.move_acceleration"),
        stop_deceleration=_float(raw.get("stop_deceleration", defaults.stop_deceleration), "robot.stop_deceleration"),
        max_stop_distance_m=_float(raw.get("max_stop_distance_m", defaults.max_stop_distance_m), "robot.max_stop_distance_m"),
        move_timeout=_float(raw.get("move_timeout", defaults.move_timeout), "robot.move_timeout"),
        monitor_rate=_float(raw.get("monitor_rate", defaults.monitor_rate), "robot.monitor_rate"),
        max_tcp_force_delta_n=_float(raw.get("max_tcp_force_delta_n", defaults.max_tcp_force_delta_n), "robot.max_tcp_force_delta_n"),
        grip_settle_seconds=_float(raw.get("grip_settle_seconds", defaults.grip_settle_seconds), "robot.grip_settle_seconds"),
        lifted_hold_seconds=_float(raw.get("lifted_hold_seconds", defaults.lifted_hold_seconds), "robot.lifted_hold_seconds"),
        release_settle_seconds=_float(raw.get("release_settle_seconds", defaults.release_settle_seconds), "robot.release_settle_seconds"),
        taught_poses=poses,
    )


def _parse_gripper(raw: dict[str, Any]) -> GripperConfig:
    allowed = {"driver", "port", "baud", "slave_id", "force_percent", "speed_percent",
               "open_position", "close_position", "position_tolerance", "serial_timeout",
               "command_timeout", "poll_interval", "read_retries"}
    _keys(raw, allowed, "gripper")
    driver = raw.get("driver", "not_configured")
    if not isinstance(driver, str) or driver not in {"mock", "not_configured", "ag105145_modbus"}:
        raise ConfigError("gripper.driver 必须是 mock、not_configured 或 ag105145_modbus")
    if driver in {"mock", "not_configured"} and set(raw) - {"driver"}:
        raise ConfigError("mock/not_configured 不得包含真实夹爪字段")
    defaults = GripperConfig()
    port = raw.get("port", defaults.port)
    if not isinstance(port, str) or not port.strip(): raise ConfigError("gripper.port 必须是非空字符串")
    baud = _int(raw.get("baud", defaults.baud), "gripper.baud")
    slave = _int(raw.get("slave_id", defaults.slave_id), "gripper.slave_id")
    force = _int(raw.get("force_percent", defaults.force_percent), "gripper.force_percent")
    speed = _int(raw.get("speed_percent", defaults.speed_percent), "gripper.speed_percent")
    op = _int(raw.get("open_position", defaults.open_position), "gripper.open_position")
    cp = _int(raw.get("close_position", defaults.close_position), "gripper.close_position")
    tol = _int(raw.get("position_tolerance", defaults.position_tolerance), "gripper.position_tolerance")
    serial_timeout = _float(raw.get("serial_timeout", defaults.serial_timeout), "gripper.serial_timeout")
    command_timeout = _float(raw.get("command_timeout", defaults.command_timeout), "gripper.command_timeout")
    poll = _float(raw.get("poll_interval", defaults.poll_interval), "gripper.poll_interval")
    retries = _int(raw.get("read_retries", defaults.read_retries), "gripper.read_retries")
    if not 1 <= slave <= 247 or not 20 <= force <= 100 or not 1 <= speed <= 100 or not 0 <= op <= 1000 or not 0 <= cp <= 1000 or op == cp:
        raise ConfigError("gripper 从站/力/速度/位置参数超出范围")
    if tol < 0 or serial_timeout <= 0 or command_timeout <= 0 or poll <= 0 or poll > command_timeout or not 0 <= retries <= 3:
        raise ConfigError("gripper 超时、轮询或重试参数无效")
    return GripperConfig(driver, port, baud, slave, force, speed, op, cp, tol,
                         serial_timeout, command_timeout, poll, retries)


def _overlap(first: SlotROI, second: SlotROI) -> bool:
    return (
        first.x < second.x + second.width
        and first.x + first.width > second.x
        and first.y < second.y + second.height
        and first.y + first.height > second.y
    )


def _max_triangle_double_area(points: tuple[tuple[float, float], ...]) -> float:
    """Largest absolute 2-D cross product among all point triples."""
    largest = 0.0
    for first in range(len(points)):
        ax, ay = points[first]
        for second in range(first + 1, len(points)):
            bx, by = points[second]
            for third in range(second + 1, len(points)):
                cx, cy = points[third]
                largest = max(
                    largest,
                    abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)),
                )
    return largest


def _pose_in_workspace(pose: Pose, workspace: WorkspaceBounds) -> bool:
    return (
        workspace.x_min <= pose.x <= workspace.x_max
        and workspace.y_min <= pose.y <= workspace.y_max
        and workspace.z_min <= pose.z <= workspace.z_max
    )


def _raise_validation(errors: list[str], label: str) -> None:
    if errors:
        raise ConfigError(f"{label}未通过:\n  - " + "\n  - ".join(errors))


def _rotation_distance(first: Pose, second: Pose) -> float:
    first_q = _rotvec_quaternion((first.rx, first.ry, first.rz))
    second_q = _rotvec_quaternion((second.rx, second.ry, second.rz))
    dot = abs(sum(a * b for a, b in zip(first_q, second_q)))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def _rotvec_quaternion(values: tuple[float, float, float]) -> tuple[float, float, float, float]:
    angle = math.sqrt(sum(value * value for value in values))
    if angle < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    scale = math.sin(angle / 2.0) / angle
    return (math.cos(angle / 2.0), *(scale * value for value in values))
