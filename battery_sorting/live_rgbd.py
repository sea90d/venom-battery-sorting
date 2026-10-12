"""Live, same-frame RGB-D capture for the eye-in-hand competition setup.

The camera is mounted on the UR5 tool, so a pixel can only be converted to
base XY with the TCP pose measured during the same capture window.  This
module owns that read-only capture/consensus step and deliberately contains no
motion or gripper commands.  The caller may pass the resulting XY values to
``Mission.build_plan`` after the later T0 teaching is complete.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .config import VisionConfig
from .handeye_runtime import BaseObjectLocation, HandEyeRuntimeMapper
from .realsense import RealSenseCamera
from .rgbd import (
    RGBDError,
    RGBDFrameResult,
    analyse_rgbd_frame,
    check_centroid_stability,
    choose_rgbd_consensus,
    detection_signature,
    fuse_depth_frames,
)
from .robot import RobotBase


class LiveRGBDError(RuntimeError):
    """The live RGB-D observation is not safe to use for robot XY."""


@dataclass(frozen=True)
class SameFrameRGBDObservation:
    """Validated multi-frame observation and the TCP poses bracketing it."""

    result: RGBDFrameResult
    tcp_pose_before: tuple[float, ...]
    tcp_pose_after: tuple[float, ...]
    consistent_frames: int
    frames_collected: int
    stability_ok: bool
    stability_reason: str

    @property
    def valid(self) -> bool:
        return self.result.valid and self.stability_ok

    def map_to_base(
        self, mapper: HandEyeRuntimeMapper
    ) -> dict[str, BaseObjectLocation]:
        if not self.valid:
            raise LiveRGBDError(
                self.result.error or self.stability_reason or "RGB-D 验收未通过"
            )
        return mapper.map_frame(
            self.result, self.tcp_pose_before, self.tcp_pose_after
        )

    def map_xy_by_color(
        self, mapper: HandEyeRuntimeMapper
    ) -> dict[str, tuple[float, float]]:
        if not self.valid:
            raise LiveRGBDError(
                self.result.error or self.stability_reason or "RGB-D 验收未通过"
            )
        return mapper.map_xy_by_color(
            self.result, self.tcp_pose_before, self.tcp_pose_after
        )


def _tcp_pose(robot: RobotBase, label: str) -> tuple[float, ...]:
    try:
        pose = tuple(float(value) for value in robot.snapshot().tcp_pose)
    except Exception as exc:
        raise LiveRGBDError(f"{label}无法读取 TCP 位姿: {exc}") from exc
    if len(pose) != 6 or any(not math.isfinite(value) for value in pose):
        raise LiveRGBDError(f"{label} TCP 位姿不是 6 个有限数")
    return pose


def _check_capture_pair(colour: np.ndarray, depth: np.ndarray | None) -> np.ndarray:
    if not isinstance(colour, np.ndarray) or colour.dtype != np.uint8:
        raise LiveRGBDError("相机彩色帧必须是 uint8 numpy 数组")
    if colour.ndim != 3 or colour.shape[2] != 3:
        raise LiveRGBDError(f"相机彩色帧格式异常: {colour.shape}")
    if depth is None:
        raise LiveRGBDError("对齐深度帧缺失")
    if not isinstance(depth, np.ndarray) or depth.ndim != 2:
        raise LiveRGBDError("相机深度帧必须是二维 numpy 数组")
    if colour.shape[:2] != depth.shape:
        raise LiveRGBDError(
            f"彩色与深度未对齐: colour={colour.shape[:2]}, depth={depth.shape}"
        )
    return depth


def capture_same_frame_rgbd(
    camera: RealSenseCamera,
    robot: RobotBase,
    vision: VisionConfig,
    *,
    frames: int = 5,
    required_consistent: int = 3,
    centroid_stability_px: float = 10.0,
    min_height_m: float = 0.015,
    max_height_m: float = 0.120,
    mapping: Any | None = None,
    mapping_unavailable_reason: str = "手眼标定运行时映射尚未接入",
) -> SameFrameRGBDObservation:
    """Capture and validate one stationary scene without moving the UR5.

    ``camera`` must already be started.  ``robot.snapshot`` is read before the
    first frame and after the last frame; ``HandEyeRuntimeMapper`` later
    rejects the observation if the TCP moved during that interval.  No TCP
    force value is read or used here.
    """
    if not camera.is_active:
        raise LiveRGBDError("RealSense 相机未启动")
    if isinstance(frames, bool) or not isinstance(frames, int) or frames < 1:
        raise LiveRGBDError("frames 必须是正整数")
    if (
        isinstance(required_consistent, bool)
        or not isinstance(required_consistent, int)
        or not 1 <= required_consistent <= frames
    ):
        raise LiveRGBDError("required_consistent 必须在 1..frames")
    if not math.isfinite(centroid_stability_px) or centroid_stability_px <= 0.0:
        raise LiveRGBDError("centroid_stability_px 必须为正的有限数")
    if not (0.0 < min_height_m < max_height_m):
        raise LiveRGBDError("高度阈值必须满足 0 < min < max")

    tcp_before = _tcp_pose(robot, "采图前")
    colour_frames: list[np.ndarray] = []
    depth_frames: list[np.ndarray] = []
    per_frame: list[RGBDFrameResult] = []
    for _ in range(frames):
        try:
            colour, depth = camera.capture()
        except Exception as exc:
            raise LiveRGBDError(f"RGB-D 取帧失败: {exc}") from exc
        depth_array = _check_capture_pair(colour, depth)
        colour_frames.append(colour.copy())
        depth_frames.append(depth_array.copy())
        per_frame.append(
            analyse_rgbd_frame(
                colour,
                depth_array,
                vision,
                camera=camera,
                mapping=mapping,
                mapping_unavailable_reason=mapping_unavailable_reason,
                min_height_m=min_height_m,
                max_height_m=max_height_m,
            )
        )
    tcp_after = _tcp_pose(robot, "采图后")

    try:
        fused_depth = fuse_depth_frames(depth_frames)
        representative_colour = colour_frames[len(colour_frames) // 2]
        fused_result = analyse_rgbd_frame(
            representative_colour,
            fused_depth,
            vision,
            camera=camera,
            mapping=mapping,
            mapping_unavailable_reason=mapping_unavailable_reason,
            min_height_m=min_height_m,
            max_height_m=max_height_m,
        )
        consensus, count = choose_rgbd_consensus(per_frame, required_consistent)
    except RGBDError as exc:
        raise LiveRGBDError(f"RGB-D 多帧处理失败: {exc}") from exc

    winner_signature = detection_signature(consensus)
    consensus_frames = [
        result
        for result in per_frame
        if result.valid and detection_signature(result) == winner_signature
    ]
    stability_ok, stability_reason = check_centroid_stability(
        fused_result,
        consensus_frames,
        min_samples=required_consistent,
        max_deviation_px=centroid_stability_px,
    )
    signature_ok = (
        fused_result.valid
        and consensus.valid
        and detection_signature(fused_result) == winner_signature
    )
    errors: list[str] = []
    if not fused_result.valid:
        errors.append(fused_result.error or "融合 RGB-D 结果无效")
    if not consensus.valid:
        errors.append(consensus.error or "逐帧颜色一致性不足")
    if fused_result.valid and consensus.valid and not signature_ok:
        errors.append("融合帧颜色映射与逐帧共识不一致")
    if not stability_ok:
        errors.append(f"质心稳定性失败: {stability_reason}")
    final_valid = fused_result.valid and consensus.valid and signature_ok and stability_ok
    final_slots = fused_result.slots
    if not final_valid:
        final_slots = tuple(
            replace(
                slot,
                robot_xy_m=None,
                robot_xy_reason="RGB-D 多帧验证未通过",
            )
            for slot in fused_result.slots
        )
    result = RGBDFrameResult(
        fused_result.detection,
        fused_result.plane,
        final_slots,
        fused_result.foreground_mask,
        final_valid,
        "; ".join(errors),
    )
    return SameFrameRGBDObservation(
        result,
        tcp_before,
        tcp_after,
        count,
        len(per_frame),
        stability_ok,
        stability_reason,
    )

