"""RGB-D battery diagnostics, deliberately isolated from robot motion."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import cv2
import numpy as np

from .config import HomographyConfig, SlotROI, VisionConfig
from .realsense import RS_DEPTH_RANGE_M, RealSenseCamera, RealSenseError
from .vision import ColorDetector, DetectionError, DetectionResult, HomographyModel


class RGBDError(RuntimeError):
    """Depth segmentation or RGB-D coordinate diagnostics failed."""


@dataclass(frozen=True)
class DepthPlaneFit:
    """Paper plane represented by inverse depth: 1/z = a*u + b*v + c."""

    a: float
    b: float
    c: float
    sample_count: int
    inlier_count: int
    rmse_m: float
    valid: bool
    reason: str = ""

    def expected_depth(self, u: np.ndarray | float, v: np.ndarray | float) -> np.ndarray:
        inverse = self.a * np.asarray(u, dtype=np.float64) + self.b * np.asarray(
            v, dtype=np.float64
        ) + self.c
        result = np.full(np.shape(inverse), np.nan, dtype=np.float64)
        np.divide(1.0, inverse, out=result, where=np.isfinite(inverse) & (inverse > 0.0))
        return result


@dataclass(frozen=True)
class SlotRGBDResult:
    slot_name: str
    color: str | None
    centroid_px: tuple[float, float] | None
    object_depth_m: float | None
    depth_mad_m: float | None
    height_above_paper_m: float | None
    foreground_pixels: int
    camera_xyz_m: tuple[float, float, float] | None
    robot_xy_m: tuple[float, float] | None
    robot_xy_reason: str
    valid: bool
    reason: str = ""


@dataclass(frozen=True)
class RGBDFrameResult:
    detection: DetectionResult
    plane: DepthPlaneFit
    slots: tuple[SlotRGBDResult, ...]
    foreground_mask: np.ndarray
    valid: bool
    error: str = ""


def _invalid_plane(reason: str, sample_count: int = 0) -> DepthPlaneFit:
    return DepthPlaneFit(0.0, 0.0, 0.0, sample_count, 0, math.inf, False, reason)


def _valid_depth_mask(depth_m: np.ndarray) -> np.ndarray:
    return (
        np.isfinite(depth_m)
        & (depth_m >= RS_DEPTH_RANGE_M[0])
        & (depth_m <= RS_DEPTH_RANGE_M[1])
    )


def fit_paper_inverse_depth_plane(
    depth_m: np.ndarray,
    rois: tuple[SlotROI, ...],
    *,
    expansion_px: int = 50,
    sample_step: int = 2,
    min_samples: int = 500,
    min_inlier_ratio: float = 0.50,
    max_rmse_m: float = 0.006,
    mad_scale: float = 3.5,
    max_iterations: int = 8,
) -> DepthPlaneFit:
    """Fit the local paper surface around four ROIs, excluding the ROIs.

    For a pinhole camera a 3-D plane is affine in inverse depth, not depth.
    The local support window prevents walls, the robot and remote objects from
    dominating the fit.  Iterative MAD rejection removes remaining outliers.
    """
    if not isinstance(depth_m, np.ndarray) or depth_m.ndim != 2:
        raise RGBDError("depth_m 必须是二维 numpy 数组")
    if not rois:
        raise RGBDError("纸面拟合至少需要一个 ROI")
    if expansion_px < 1 or sample_step < 1 or min_samples < 6:
        raise RGBDError("纸面拟合参数无效")
    if not (0.0 < min_inlier_ratio <= 1.0):
        raise RGBDError("min_inlier_ratio 必须在 (0,1]")
    if max_rmse_m <= 0.0 or mad_scale <= 0.0 or max_iterations < 1:
        raise RGBDError("纸面拟合阈值必须为正")

    height, width = depth_m.shape
    left = max(0, min(roi.x for roi in rois) - expansion_px)
    top = max(0, min(roi.y for roi in rois) - expansion_px)
    right = min(width, max(roi.x + roi.width for roi in rois) + expansion_px)
    bottom = min(height, max(roi.y + roi.height for roi in rois) + expansion_px)
    if right <= left or bottom <= top:
        return _invalid_plane("局部纸面采样区域为空")

    support = np.zeros(depth_m.shape, dtype=bool)
    support[top:bottom:sample_step, left:right:sample_step] = True
    for roi in rois:
        support[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width] = False
    support &= _valid_depth_mask(depth_m)

    vv, uu = np.nonzero(support)
    samples = len(uu)
    if samples < min_samples:
        return _invalid_plane(
            f"局部纸面有效深度样本不足: {samples} < {min_samples}", samples
        )
    z = depth_m[vv, uu].astype(np.float64)
    design = np.column_stack(
        (uu.astype(np.float64), vv.astype(np.float64), np.ones(samples))
    )
    inverse_depth = 1.0 / z
    inliers = np.ones(samples, dtype=bool)

    for _ in range(max_iterations):
        if int(np.count_nonzero(inliers)) < 6:
            return _invalid_plane("纸面拟合内点少于 6 个", samples)
        coefficients, _, rank, _ = np.linalg.lstsq(
            design[inliers], inverse_depth[inliers], rcond=None
        )
        if rank < 3 or not np.all(np.isfinite(coefficients)):
            return _invalid_plane("纸面拟合退化或得到非有限参数", samples)
        predicted_inverse = design @ coefficients
        predicted_z = np.full(samples, np.nan, dtype=np.float64)
        np.divide(
            1.0,
            predicted_inverse,
            out=predicted_z,
            where=np.isfinite(predicted_inverse) & (predicted_inverse > 0.0),
        )
        residual_m = z - predicted_z
        current = residual_m[inliers & np.isfinite(residual_m)]
        if len(current) < 6:
            return _invalid_plane("纸面拟合残差无效", samples)
        centre = float(np.median(current))
        mad = float(np.median(np.abs(current - centre)))
        threshold = max(0.002, mad_scale * 1.4826 * mad)
        updated = np.isfinite(residual_m) & (np.abs(residual_m - centre) <= threshold)
        if np.array_equal(updated, inliers):
            break
        inliers = updated

    inlier_count = int(np.count_nonzero(inliers))
    ratio = inlier_count / samples
    if inlier_count < 6 or ratio < min_inlier_ratio:
        return DepthPlaneFit(
            *[float(value) for value in coefficients],
            samples,
            inlier_count,
            math.inf,
            False,
            f"纸面内点比例 {ratio:.1%} < {min_inlier_ratio:.1%}",
        )

    coefficients, _, rank, _ = np.linalg.lstsq(
        design[inliers], inverse_depth[inliers], rcond=None
    )
    if rank < 3 or not np.all(np.isfinite(coefficients)):
        return _invalid_plane("最终纸面拟合退化", samples)
    predicted_inverse = design[inliers] @ coefficients
    if np.any(predicted_inverse <= 0.0):
        return _invalid_plane("纸面拟合预测了非正深度", samples)
    residual_m = z[inliers] - 1.0 / predicted_inverse
    rmse_m = float(np.sqrt(np.mean(residual_m**2)))
    valid = math.isfinite(rmse_m) and rmse_m <= max_rmse_m
    reason = "" if valid else f"纸面拟合 RMSE {rmse_m * 1000:.2f} mm > {max_rmse_m * 1000:.2f} mm"
    return DepthPlaneFit(
        float(coefficients[0]),
        float(coefficients[1]),
        float(coefficients[2]),
        samples,
        inlier_count,
        rmse_m,
        valid,
        reason,
    )


def make_raised_foreground_mask(
    depth_m: np.ndarray,
    plane: DepthPlaneFit,
    rois: tuple[SlotROI, ...],
    *,
    min_height_m: float = 0.015,
    max_height_m: float = 0.120,
) -> np.ndarray:
    """Keep only pixels 15-120 mm closer than the predicted paper plane."""
    if not plane.valid:
        raise RGBDError(f"纸面拟合无效: {plane.reason}")
    if not (0.0 < min_height_m < max_height_m):
        raise RGBDError("凸起高度阈值必须满足 0 < min < max")
    height, width = depth_m.shape
    vv, uu = np.mgrid[0:height, 0:width]
    expected_z = plane.expected_depth(uu, vv)
    raised_m = expected_z - depth_m.astype(np.float64)
    candidate = (
        _valid_depth_mask(depth_m)
        & np.isfinite(expected_z)
        & (raised_m >= min_height_m)
        & (raised_m <= max_height_m)
    )
    output = np.zeros(depth_m.shape, dtype=np.uint8)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    for roi in rois:
        local = candidate[
            roi.y : roi.y + roi.height, roi.x : roi.x + roi.width
        ].astype(np.uint8) * 255
        local = cv2.morphologyEx(local, cv2.MORPH_OPEN, open_kernel)
        local = cv2.morphologyEx(local, cv2.MORPH_CLOSE, close_kernel)
        output[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width] = local
    return output


def fuse_depth_frames(depth_frames: list[np.ndarray]) -> np.ndarray:
    """Fuse aligned depth frames without spatially filling object boundaries.

    The pixel-wise median ignores non-finite and out-of-range samples.  A
    pixel missing in every frame stays missing (0.0), so this operation cannot
    invent depth by copying neighbouring pixels across an object edge.
    """
    if not depth_frames:
        raise RGBDError("depth_frames 不能为空")
    shape = depth_frames[0].shape
    if len(shape) != 2:
        raise RGBDError("depth_frames 必须是二维数组列表")
    for index, depth in enumerate(depth_frames):
        if not isinstance(depth, np.ndarray) or depth.ndim != 2:
            raise RGBDError(f"depth_frames[{index}] 必须是二维 numpy 数组")
        if depth.shape != shape:
            raise RGBDError(
                f"depth_frames[{index}] 形状 {depth.shape} 与首帧 {shape} 不一致"
            )

    stacked = np.stack(depth_frames, axis=0).astype(np.float32, copy=False)
    valid = (
        np.isfinite(stacked)
        & (stacked >= RS_DEPTH_RANGE_M[0])
        & (stacked <= RS_DEPTH_RANGE_M[1])
    )
    masked = np.where(valid, stacked, np.nan)
    all_missing = ~np.any(valid, axis=0)
    # Avoid an all-NaN-slice warning without turning a missing pixel into a
    # valid measurement: these locations are explicitly restored to 0 below.
    masked[:, all_missing] = 0.0
    fused = np.nanmedian(masked, axis=0, overwrite_input=True)
    fused[all_missing | ~np.isfinite(fused)] = 0.0
    return fused.astype(np.float32, copy=False)


def create_homography_if_calibrated(
    config: HomographyConfig,
) -> tuple[HomographyModel | None, str]:
    if not config.calibrated:
        return None, "单应性尚未标定"
    try:
        return HomographyModel(config), ""
    except DetectionError as exc:
        return None, f"单应性无效: {exc}"


def analyse_rgbd_frame(
    colour_bgr: np.ndarray,
    depth_m: np.ndarray,
    vision: VisionConfig,
    *,
    camera: RealSenseCamera | None = None,
    mapping: HomographyModel | None = None,
    mapping_unavailable_reason: str = "单应性尚未标定",
    min_height_m: float = 0.015,
    max_height_m: float = 0.120,
) -> RGBDFrameResult:
    """Analyse one aligned RGB-D pair without importing any robot code."""
    detector = ColorDetector(vision, colour_bgr.shape[1], colour_bgr.shape[0])
    plane = fit_paper_inverse_depth_plane(depth_m, vision.rois)
    if not plane.valid:
        empty = np.zeros(depth_m.shape, dtype=np.uint8)
        detection = detector.detect(colour_bgr, foreground_mask=empty)
        return RGBDFrameResult(
            detection, plane, (), empty, False, f"纸面拟合失败: {plane.reason}"
        )
    foreground = make_raised_foreground_mask(
        depth_m,
        plane,
        vision.rois,
        min_height_m=min_height_m,
        max_height_m=max_height_m,
    )
    detection = detector.detect(colour_bgr, foreground_mask=foreground)
    slot_results: list[SlotRGBDResult] = []
    for slot in detection.slots:
        roi = next(item for item in vision.rois if item.slot_name == slot.slot_name)
        local_fg = foreground[
            roi.y : roi.y + roi.height, roi.x : roi.x + roi.width
        ] > 0
        local_depth = depth_m[
            roi.y : roi.y + roi.height, roi.x : roi.x + roi.width
        ]
        values = local_depth[local_fg & _valid_depth_mask(local_depth)].astype(np.float64)
        object_depth: float | None = None
        depth_mad: float | None = None
        height_above: float | None = None
        camera_xyz: tuple[float, float, float] | None = None
        robot_xy: tuple[float, float] | None = None
        robot_reason = mapping_unavailable_reason if mapping is None else ""
        reason = slot.reason
        if slot.valid and slot.centroid is not None and len(values) >= 20:
            object_depth = float(np.median(values))
            depth_mad = float(np.median(np.abs(values - object_depth)))
            u, v = slot.centroid
            expected = float(plane.expected_depth(u, v))
            height_above = expected - object_depth
            if not (min_height_m <= height_above <= max_height_m):
                reason = f"估计凸起高度 {height_above * 1000:.1f} mm 超出范围"
            if not reason and camera is not None:
                try:
                    camera_xyz = camera.deproject_pixel(u, v, object_depth)
                except RealSenseError as exc:
                    reason = f"相机反投影失败: {exc}"
            if not reason and mapping is not None:
                try:
                    robot_xy = mapping.map_pixel_to_robot(u, v)
                except DetectionError as exc:
                    robot_reason = str(exc)
        elif slot.valid and len(values) < 20:
            reason = f"深度前景样本不足: {len(values)} < 20"
        elif slot.centroid is None and not reason:
            reason = "没有可靠质心"
        valid = slot.valid and not reason and object_depth is not None
        slot_results.append(
            SlotRGBDResult(
                slot.slot_name,
                slot.color,
                slot.centroid,
                object_depth,
                depth_mad,
                height_above,
                int(len(values)),
                camera_xyz,
                robot_xy,
                robot_reason,
                valid,
                reason,
            )
        )
    errors = [f"{slot.slot_name}: {slot.reason}" for slot in slot_results if not slot.valid]
    valid = detection.valid and len(slot_results) == 4 and not errors
    return RGBDFrameResult(
        detection,
        plane,
        tuple(slot_results),
        foreground,
        valid,
        "; ".join(errors) if errors else detection.error,
    )


def detection_signature(result: RGBDFrameResult) -> tuple[tuple[str, str | None], ...]:
    return tuple(sorted((slot.slot_name, slot.color) for slot in result.slots))


def choose_rgbd_consensus(
    results: list[RGBDFrameResult], required_consistent: int
) -> tuple[RGBDFrameResult, int]:
    if not results or not (1 <= required_consistent <= len(results)):
        raise RGBDError("RGB-D 多帧一致性参数无效")
    valid = [result for result in results if result.valid]
    if not valid:
        return replace(results[-1], valid=False, error=results[-1].error or "没有有效 RGB-D 帧"), 0
    counts: dict[tuple[tuple[str, str | None], ...], int] = {}
    for result in valid:
        signature = detection_signature(result)
        counts[signature] = counts.get(signature, 0) + 1
    winner, count = max(counts.items(), key=lambda item: item[1])
    representative = next(result for result in reversed(valid) if detection_signature(result) == winner)
    if count < required_consistent:
        return replace(
            representative,
            valid=False,
            error=f"RGB-D 多帧一致性不足: {count}/{len(results)}，要求 {required_consistent}",
        ), count
    return representative, count


def check_centroid_stability(
    fused_result: RGBDFrameResult,
    consensus_results: list[RGBDFrameResult],
    *,
    min_samples: int,
    max_deviation_px: float = 10.0,
) -> tuple[bool, str]:
    """Reject a fused centroid that is unsupported or unstable across frames."""
    if min_samples < 1:
        raise RGBDError("质心稳定性最少样本数必须为正")
    if not math.isfinite(max_deviation_px) or max_deviation_px <= 0.0:
        raise RGBDError("质心稳定性阈值必须为正的有限数")
    if not consensus_results:
        return False, "没有颜色一致帧可用于质心稳定性检查"

    problems: list[str] = []
    for fused_slot in fused_result.slots:
        if not fused_slot.valid or fused_slot.centroid_px is None:
            problems.append(f"{fused_slot.slot_name}: 融合结果无可靠质心")
            continue
        centroids: list[tuple[float, float]] = []
        for frame_result in consensus_results:
            frame_slot = next(
                (slot for slot in frame_result.slots if slot.slot_name == fused_slot.slot_name),
                None,
            )
            if frame_slot is not None and frame_slot.valid and frame_slot.centroid_px is not None:
                centroids.append(frame_slot.centroid_px)
        if len(centroids) < min_samples:
            problems.append(
                f"{fused_slot.slot_name}: 有效逐帧质心 {len(centroids)} < {min_samples}"
            )
            continue

        points = np.asarray(centroids, dtype=np.float64)
        median = np.median(points, axis=0)
        fused = np.asarray(fused_slot.centroid_px, dtype=np.float64)
        fused_offset = float(np.linalg.norm(fused - median))
        frame_offsets = np.linalg.norm(points - median, axis=1)
        median_spread = float(np.median(frame_offsets))
        if fused_offset > max_deviation_px:
            problems.append(
                f"{fused_slot.slot_name}: 融合质心偏离逐帧中位 "
                f"{fused_offset:.1f}px > {max_deviation_px:.1f}px"
            )
        if median_spread > max_deviation_px:
            problems.append(
                f"{fused_slot.slot_name}: 逐帧质心离散中位 "
                f"{median_spread:.1f}px > {max_deviation_px:.1f}px"
            )
    return (not problems, "; ".join(problems))


def annotate_rgbd(colour_bgr: np.ndarray, result: RGBDFrameResult) -> np.ndarray:
    annotated = colour_bgr.copy()
    overlay = np.zeros_like(annotated)
    overlay[result.foreground_mask > 0] = (255, 0, 255)
    annotated = cv2.addWeighted(annotated, 1.0, overlay, 0.25, 0.0)
    for slot in result.slots:
        detection = next(
            item for item in result.detection.slots if item.slot_name == slot.slot_name
        )
        colour = (0, 200, 0) if slot.valid else (0, 0, 255)
        if detection.centroid is not None:
            point = (int(round(detection.centroid[0])), int(round(detection.centroid[1])))
            cv2.drawMarker(annotated, point, colour, cv2.MARKER_CROSS, 18, 2)
        label = f"{slot.slot_name}:{slot.color or '-'}"
        if slot.height_above_paper_m is not None:
            label += f" h={slot.height_above_paper_m * 1000:.0f}mm"
        cv2.putText(
            annotated,
            label,
            (int(detection.centroid[0]) - 45, int(detection.centroid[1]) - 20)
            if detection.centroid is not None
            else (10, 25 + 22 * detection.slot_index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            colour,
            1,
            cv2.LINE_AA,
        )
    return annotated
