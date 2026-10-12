"""Convert one same-frame RGB-D observation into UR base coordinates.

This module is deliberately separate from the motion executor.  It consumes a
validated hand-eye result, an RGB-D frame, and the TCP pose measured for that
same frame.  It never connects to or commands the robot.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import validate_rigid_transform
from .handeye import HandEyeError, pose_vector_to_transform
from .rgbd import RGBDFrameResult


class HandEyeRuntimeError(RuntimeError):
    """Same-frame hand-eye input is missing or unsafe to use."""


@dataclass(frozen=True)
class BaseObjectLocation:
    color: str
    slot_name: str
    centroid_px: tuple[float, float]
    camera_xyz_m: tuple[float, float, float]
    base_xyz_m: tuple[float, float, float]


@dataclass(frozen=True)
class HandEyeRuntimeMapper:
    tcp_T_camera: np.ndarray
    source_path: str = ""

    def __post_init__(self) -> None:
        matrix = np.asarray(self.tcp_T_camera, dtype=np.float64)
        validate_rigid_transform(matrix, "tcp_T_camera")
        object.__setattr__(self, "tcp_T_camera", matrix.copy())

    @classmethod
    def from_json(cls, path: str | Path) -> "HandEyeRuntimeMapper":
        source = Path(path)
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HandEyeRuntimeError(f"无法读取手眼标定结果: {source}: {exc}") from exc
        if raw.get("passed") is not True:
            raise HandEyeRuntimeError("手眼标定结果 passed 不是 true")
        try:
            matrix = np.asarray(raw["tcp_T_camera"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise HandEyeRuntimeError("手眼标定结果缺少有效 tcp_T_camera") from exc
        try:
            return cls(matrix, str(source))
        except (ValueError, HandEyeError) as exc:
            raise HandEyeRuntimeError(f"tcp_T_camera 无效: {exc}") from exc

    @staticmethod
    def _check_same_frame(
        tcp_pose_before: tuple[float, ...],
        tcp_pose_after: tuple[float, ...] | None,
        *,
        max_translation_m: float = 0.0002,
        max_rotation_rad: float = math.radians(0.1),
    ) -> None:
        if len(tcp_pose_before) != 6 or any(not math.isfinite(v) for v in tcp_pose_before):
            raise HandEyeRuntimeError("采图前 TCP 位姿无效")
        if tcp_pose_after is None:
            return
        if len(tcp_pose_after) != 6 or any(not math.isfinite(v) for v in tcp_pose_after):
            raise HandEyeRuntimeError("采图后 TCP 位姿无效")
        translation = math.dist(tcp_pose_before[:3], tcp_pose_after[:3])
        rotation = math.dist(tcp_pose_before[3:], tcp_pose_after[3:])
        if translation > max_translation_m or rotation > max_rotation_rad:
            raise HandEyeRuntimeError(
                f"采图期间 TCP 移动 {translation * 1000:.3f}mm / "
                f"旋转向量变化 {math.degrees(rotation):.3f}°，拒绝映射"
            )

    def map_frame(
        self,
        result: RGBDFrameResult,
        tcp_pose_before: tuple[float, ...],
        tcp_pose_after: tuple[float, ...] | None = None,
    ) -> dict[str, BaseObjectLocation]:
        if not result.valid:
            raise HandEyeRuntimeError(f"RGB-D 结果无效: {result.error}")
        self._check_same_frame(tcp_pose_before, tcp_pose_after)
        base_T_tcp = pose_vector_to_transform(tcp_pose_before)
        base_T_camera = base_T_tcp @ self.tcp_T_camera
        locations: dict[str, BaseObjectLocation] = {}
        for slot in result.slots:
            if not slot.valid or slot.color is None or slot.centroid_px is None:
                raise HandEyeRuntimeError(f"{slot.slot_name} 缺少有效颜色或质心")
            if slot.camera_xyz_m is None:
                raise HandEyeRuntimeError(f"{slot.slot_name} 缺少相机三维坐标")
            if slot.color in locations:
                raise HandEyeRuntimeError(f"颜色 {slot.color} 出现多次")
            camera_xyz = tuple(float(v) for v in slot.camera_xyz_m)
            if any(not math.isfinite(v) for v in camera_xyz) or camera_xyz[2] <= 0.0:
                raise HandEyeRuntimeError(f"{slot.slot_name} 相机坐标无效")
            point = np.concatenate((np.asarray(camera_xyz, dtype=np.float64), [1.0]))
            base_xyz = tuple(float(v) for v in (base_T_camera @ point)[:3])
            locations[slot.color] = BaseObjectLocation(
                slot.color, slot.slot_name, slot.centroid_px, camera_xyz, base_xyz
            )
        if set(locations) != {"red", "blue", "yellow", "green"}:
            raise HandEyeRuntimeError(f"颜色集合不完整: {sorted(locations)}")
        return locations

    def map_xy_by_color(
        self,
        result: RGBDFrameResult,
        tcp_pose_before: tuple[float, ...],
        tcp_pose_after: tuple[float, ...] | None = None,
    ) -> dict[str, tuple[float, float]]:
        return {
            color: (location.base_xyz_m[0], location.base_xyz_m[1])
            for color, location in self.map_frame(result, tcp_pose_before, tcp_pose_after).items()
        }
