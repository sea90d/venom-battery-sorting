"""USB 相机封装与固定四槽 HSV 颜色识别。"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass

import cv2
import numpy as np

from .config import COLORS, CameraConfig, HomographyConfig, SlotROI, VisionConfig

LOGGER = logging.getLogger(__name__)


class CameraError(RuntimeError):
    """相机打开、参数设置或采集失败。"""


class DetectionError(RuntimeError):
    """图像或识别配置不满足要求。"""


class Camera:
    def __init__(self, config: CameraConfig, *, strict_parameters: bool = False) -> None:
        self.config = config
        self.strict_parameters = strict_parameters
        self._capture: cv2.VideoCapture | None = None

    def open(self) -> None:
        if self._capture is not None:
            raise CameraError("相机已打开")
        capture = cv2.VideoCapture(self.config.device_id)
        self._capture = capture
        try:
            if not capture.isOpened():
                raise CameraError(f"无法打开相机 {self.config.device_id}")
            self._set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width, "width", 0.5)
            self._set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height, "height", 0.5)
            self._set(cv2.CAP_PROP_FPS, self.config.fps, "fps", 0.5)
            if self.config.exposure is not None:
                self._set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25, "manual exposure mode", 0.05)
                self._set(cv2.CAP_PROP_EXPOSURE, self.config.exposure, "exposure", 0.25)
            if self.config.gain is not None:
                self._set(cv2.CAP_PROP_GAIN, self.config.gain, "gain", 0.25)
            if self.config.white_balance_temperature is not None:
                self._set(cv2.CAP_PROP_AUTO_WB, 0, "manual white balance mode", 0.05)
                self._set(
                    cv2.CAP_PROP_WB_TEMPERATURE,
                    self.config.white_balance_temperature,
                    "white balance temperature",
                    100.0,
                )
            if self.config.focus is not None:
                self._set(cv2.CAP_PROP_AUTOFOCUS, 0, "manual focus mode", 0.05)
                self._set(cv2.CAP_PROP_FOCUS, self.config.focus, "focus", 1.0)
            self.capture()
        except Exception:
            self.close()
            raise

    def _set(
        self, prop: int, value: float, name: str, readback_tolerance: float
    ) -> None:
        assert self._capture is not None
        if not self._capture.set(prop, value):
            message = f"相机不支持或拒绝设置 {name}={value}"
            if self.strict_parameters:
                raise CameraError(message)
            LOGGER.warning(message)
            return
        actual = float(self._capture.get(prop))
        if not np.isfinite(actual) or abs(actual - float(value)) > readback_tolerance:
            message = f"相机 {name} 请求 {value}，回读 {actual}，参数可能未锁定"
            if self.strict_parameters:
                raise CameraError(message)
            LOGGER.warning(message)

    def capture(self) -> np.ndarray:
        if self._capture is None:
            raise CameraError("相机尚未打开")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise CameraError("相机取帧失败")
        height, width = frame.shape[:2]
        if (width, height) != (self.config.width, self.config.height):
            raise CameraError(
                f"相机实际输出 {width}x{height}，配置要求 {self.config.width}x{self.config.height}"
            )
        return frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@dataclass(frozen=True)
class SlotDetection:
    slot_index: int
    slot_name: str
    color: str | None
    coverage: float
    confidence: float
    margin: float
    centroid: tuple[float, float] | None
    centroid_offset: tuple[float, float] | None
    color_coverage: dict[str, float]
    color_purity: dict[str, float]
    valid: bool
    contour_area_px: float | None = None
    long_axis_angle_deg: float | None = None
    reason: str = ""


@dataclass(frozen=True)
class DetectionResult:
    slots: tuple[SlotDetection, ...]
    valid: bool
    error: str = ""

    def color_to_slot(self) -> dict[str, str]:
        if not self.valid:
            raise DetectionError(self.error or "识别结果无效")
        return {slot.color: slot.slot_name for slot in self.slots if slot.color is not None}

    def color_to_centroid(self) -> dict[str, tuple[float, float]]:
        if not self.valid:
            raise DetectionError(self.error or "识别结果无效")
        result: dict[str, tuple[float, float]] = {}
        for slot in self.slots:
            if slot.color is None or slot.centroid is None:
                raise DetectionError(f"{slot.slot_name} 缺少可靠颜色或质心")
            result[slot.color] = slot.centroid
        if set(result) != COLORS:
            raise DetectionError("四种颜色未全部得到唯一可靠质心")
        return result


def choose_consensus(
    results: list[DetectionResult], required_consistent_frames: int
) -> DetectionResult:
    """要求若干帧产生完全相同的 slot->color 映射。"""
    if not results:
        raise DetectionError("多帧一致性检查没有输入帧")
    if not (1 <= required_consistent_frames <= len(results)):
        raise DetectionError("required_consistent_frames 超出实际帧数")
    valid_results = [result for result in results if result.valid]
    signatures = [
        tuple(sorted((slot.slot_name, slot.color) for slot in result.slots))
        for result in valid_results
    ]
    counts = Counter(signatures)
    if counts:
        signature, count = counts.most_common(1)[0]
        if count >= required_consistent_frames:
            return valid_results[signatures.index(signature)]
    representative = valid_results[0] if valid_results else results[-1]
    best = counts.most_common(1)[0][1] if counts else 0
    return DetectionResult(
        representative.slots,
        False,
        f"多帧一致性不足: 最多 {best}/{len(results)} 帧一致，要求 {required_consistent_frames}",
    )


class ColorDetector:
    """识别四个固定槽位中的颜色；质心不参与机器人坐标计算。"""

    def __init__(self, config: VisionConfig, image_width: int, image_height: int) -> None:
        self.config = config
        self.image_width = image_width
        self.image_height = image_height
        if len(config.rois) != 4 or {roi.slot_name for roi in config.rois} != {
            "slot1", "slot2", "slot3", "slot4"
        }:
            raise DetectionError("检测器要求显式配置 slot1..slot4 四个 ROI")
        names = [color.name for color in config.colors]
        if set(names) != COLORS or len(names) != len(COLORS):
            raise DetectionError("检测器要求 red/blue/yellow/green 各配置一次")
        if (
            config.morph_kernel_size < 1
            or config.morph_kernel_size > 31
            or config.morph_kernel_size % 2 == 0
        ):
            raise DetectionError("形态学核必须是 1..31 的奇数")
        for roi in config.rois:
            if (
                roi.x < 0
                or roi.y < 0
                or roi.width <= 0
                or roi.height <= 0
                or roi.x + roi.width > image_width
                or roi.y + roi.height > image_height
            ):
                raise DetectionError(f"{roi.slot_name} ROI 超出画面")
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (config.morph_kernel_size, config.morph_kernel_size)
        )

    def detect(
        self,
        frame: np.ndarray,
        *,
        foreground_mask: np.ndarray | None = None,
    ) -> DetectionResult:
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
            raise DetectionError("输入必须是 uint8 numpy 图像")
        if frame.shape != (self.image_height, self.image_width, 3):
            raise DetectionError(
                f"输入形状 {frame.shape}，期望 {(self.image_height, self.image_width, 3)}"
            )
        normalized_foreground: np.ndarray | None = None
        if foreground_mask is not None:
            if not isinstance(foreground_mask, np.ndarray):
                raise DetectionError("foreground_mask 必须是 numpy 数组")
            if foreground_mask.shape != (self.image_height, self.image_width):
                raise DetectionError(
                    f"foreground_mask 形状 {foreground_mask.shape}，期望 "
                    f"{(self.image_height, self.image_width)}"
                )
            if foreground_mask.dtype not in (np.dtype(np.uint8), np.dtype(np.bool_)):
                raise DetectionError("foreground_mask 必须是 uint8 或 bool 二值图")
            normalized_foreground = (foreground_mask > 0).astype(np.uint8) * 255
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        slots = tuple(
            self._detect_slot(
                hsv[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width],
                roi,
                index,
                None
                if normalized_foreground is None
                else normalized_foreground[
                    roi.y : roi.y + roi.height,
                    roi.x : roi.x + roi.width,
                ],
            )
            for index, roi in enumerate(self.config.rois)
        )
        valid_colors = [slot.color for slot in slots if slot.valid and slot.color is not None]
        reasons = [f"{slot.slot_name}: {slot.reason}" for slot in slots if not slot.valid]
        if len(valid_colors) != 4:
            reasons.append("并非四个槽位都得到可靠颜色")
        if len(set(valid_colors)) != len(valid_colors):
            reasons.append("四个槽位出现重复颜色")
        valid = len(valid_colors) == 4 and len(set(valid_colors)) == 4 and not reasons
        return DetectionResult(slots, valid, "; ".join(reasons))

    def _detect_slot(
        self,
        hsv: np.ndarray,
        roi: SlotROI,
        slot_index: int,
        foreground_mask: np.ndarray | None = None,
    ) -> SlotDetection:
        area = int(hsv.shape[0] * hsv.shape[1])
        if area <= 0:
            return self._invalid(slot_index, roi.slot_name, "ROI 为空")
        masks: dict[str, np.ndarray] = {}
        for color in self.config.colors:
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for threshold in color.ranges:
                lower = np.array([threshold.h_min, threshold.s_min, threshold.v_min], dtype=np.uint8)
                upper = np.array([threshold.h_max, threshold.s_max, threshold.v_max], dtype=np.uint8)
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
            for _ in range(self.config.morph_iterations):
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
            if foreground_mask is not None:
                mask = cv2.bitwise_and(mask, foreground_mask)
            masks[color.name] = mask

        stack = np.stack([(mask > 0).astype(np.uint8) for mask in masks.values()], axis=0)
        overlap_pixels = int(np.count_nonzero(np.sum(stack, axis=0) > 1))
        union = np.any(stack > 0, axis=0)
        colored_pixels = int(np.count_nonzero(union))
        coverage = colored_pixels / area
        counts = {name: int(np.count_nonzero(mask)) for name, mask in masks.items()}
        color_coverage = {name: count / area for name, count in counts.items()}
        color_purity = {
            name: (count / colored_pixels if colored_pixels else 0.0) for name, count in counts.items()
        }
        ranking = sorted(color_purity.items(), key=lambda item: item[1], reverse=True)
        winner, confidence = ranking[0]
        runner_up = ranking[1][1]
        margin = confidence - runner_up
        reason = ""
        if overlap_pixels:
            reason = f"HSV 阈值在 {overlap_pixels} 个像素上重叠"
        elif coverage < self.config.min_coverage:
            reason = f"覆盖率 {coverage:.3f} < {self.config.min_coverage:.3f}"
        elif confidence < self.config.min_confidence:
            reason = f"颜色纯度 {confidence:.3f} < {self.config.min_confidence:.3f}"
        elif counts[winner] <= sorted(counts.values(), reverse=True)[1]:
            reason = "最高颜色像素与次高颜色平局"
        elif margin < self.config.min_margin:
            reason = f"冠亚颜色差 {margin:.3f} < {self.config.min_margin:.3f}"
        geometry = _largest_component_geometry(masks[winner], roi) if colored_pixels else None
        centroid = None if geometry is None else geometry[0]
        contour_area = None if geometry is None else geometry[1]
        long_axis_angle = None if geometry is None else geometry[2]
        offset = None if centroid is None else (
            centroid[0] - roi.expected_center_x,
            centroid[1] - roi.expected_center_y,
        )
        if not reason and centroid is None:
            reason = "获胜颜色没有有效连通域"
        if not reason and offset is not None:
            offset_norm = float(np.hypot(offset[0], offset[1]))
            if offset_norm > roi.max_centroid_offset_px:
                reason = (
                    f"质心偏离示教槽中心 {offset_norm:.1f}px > "
                    f"{roi.max_centroid_offset_px:.1f}px"
                )
        if (
            not reason
            and roi.expected_component_area_px is not None
            and contour_area is not None
        ):
            relative_error = abs(contour_area - roi.expected_component_area_px) / (
                roi.expected_component_area_px
            )
            if relative_error > roi.area_tolerance_fraction:
                reason = (
                    f"最大色块面积 {contour_area:.0f}px 偏离期望 "
                    f"{roi.expected_component_area_px:.0f}px 达 {relative_error:.1%} > "
                    f"{roi.area_tolerance_fraction:.1%}"
                )
        if (
            not reason
            and roi.expected_long_axis_angle_deg is not None
            and long_axis_angle is not None
        ):
            angle_error = _angle_diff_180(
                roi.expected_long_axis_angle_deg, long_axis_angle
            )
            if angle_error > roi.max_angle_error_deg:
                reason = (
                    f"最大色块长轴 {long_axis_angle:.1f}° 偏离期望 "
                    f"{roi.expected_long_axis_angle_deg:.1f}° 达 {angle_error:.1f}° > "
                    f"{roi.max_angle_error_deg:.1f}°"
                )
        return SlotDetection(
            slot_index=slot_index,
            slot_name=roi.slot_name,
            color=None if reason else winner,
            coverage=coverage,
            confidence=confidence,
            margin=margin,
            centroid=centroid,
            centroid_offset=offset,
            color_coverage=color_coverage,
            color_purity=color_purity,
            valid=not reason,
            contour_area_px=contour_area,
            long_axis_angle_deg=long_axis_angle,
            reason=reason,
        )

    @staticmethod
    def _invalid(slot_index: int, slot_name: str, reason: str) -> SlotDetection:
        return SlotDetection(
            slot_index=slot_index,
            slot_name=slot_name,
            color=None,
            coverage=0.0,
            confidence=0.0,
            margin=0.0,
            centroid=None,
            centroid_offset=None,
            color_coverage={},
            color_purity={},
            valid=False,
            reason=reason,
        )


def _largest_component_geometry(
    mask: np.ndarray, roi: SlotROI
) -> tuple[tuple[float, float], float, float] | None:
    """Return full-image centroid, contour area and long-axis angle in [-90, 90)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    moments = cv2.moments(largest)
    if moments["m00"] <= 0:
        return None
    centroid = (
        float(moments["m10"] / moments["m00"] + roi.x),
        float(moments["m01"] / moments["m00"] + roi.y),
    )
    box = cv2.boxPoints(cv2.minAreaRect(largest))
    edges = np.roll(box, -1, axis=0) - box
    edge = edges[int(np.argmax(np.linalg.norm(edges, axis=1)))]
    angle = math.degrees(math.atan2(float(edge[1]), float(edge[0])))
    angle = (angle + 90.0) % 180.0 - 90.0
    return centroid, float(cv2.contourArea(largest)), angle


def _angle_diff_180(first: float, second: float) -> float:
    """Smallest absolute difference for an undirected long axis."""
    return abs((first - second + 90.0) % 180.0 - 90.0)


class HomographyModel:
    """Validated planar mapping from image pixels to UR base-frame XY."""

    def __init__(self, config: HomographyConfig) -> None:
        if not config.calibrated:
            raise DetectionError("vision.homography.calibrated=false，无法生成机器人 XY")
        if len(config.pixel_points) < 4:
            raise DetectionError("单应性至少需要 4 个标定点对")
        if len(config.pixel_points) != len(config.robot_xy_points_m):
            raise DetectionError("单应性像素点与机器人 XY 点数量不一致")
        pixel = np.asarray(config.pixel_points, dtype=np.float64)
        robot = np.asarray(config.robot_xy_points_m, dtype=np.float64)
        if not np.all(np.isfinite(pixel)) or not np.all(np.isfinite(robot)):
            raise DetectionError("单应性标定点包含非有限数值")
        if len(np.unique(pixel, axis=0)) != len(pixel):
            raise DetectionError("单应性像素标定点存在重复")
        if len(np.unique(robot, axis=0)) != len(robot):
            raise DetectionError("单应性机器人 XY 标定点存在重复")
        self._pixel_hull = cv2.convexHull(pixel.astype(np.float32))
        robot_hull = cv2.convexHull(robot.astype(np.float32))
        if cv2.contourArea(self._pixel_hull) < 1.0:
            raise DetectionError("单应性像素标定点共线或覆盖面积过小")
        if cv2.contourArea(robot_hull) < 1e-8:
            raise DetectionError("单应性机器人 XY 标定点共线或覆盖面积过小")

        matrix, _ = cv2.findHomography(pixel, robot, method=0)
        if matrix is None or matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise DetectionError("单应性矩阵计算失败")
        try:
            inverse = np.linalg.inv(matrix)
        except np.linalg.LinAlgError as exc:
            raise DetectionError("单应性矩阵不可逆") from exc
        if not np.all(np.isfinite(inverse)):
            raise DetectionError("单应性逆矩阵包含非有限数值")

        projected = cv2.perspectiveTransform(
            pixel.reshape(-1, 1, 2), matrix
        ).reshape(-1, 2)
        if not np.all(np.isfinite(projected)):
            raise DetectionError("单应性标定点重投影得到非有限坐标")
        errors = np.linalg.norm(projected - robot, axis=1)
        self._reprojection_rmse_m = float(np.sqrt(np.mean(errors**2)))
        self._max_reprojection_error_m = float(np.max(errors))
        if self._max_reprojection_error_m > config.max_reprojection_error_m:
            raise DetectionError(
                f"单应性最大重投影误差 {self._max_reprojection_error_m * 1000:.2f} mm "
                f"超过 {config.max_reprojection_error_m * 1000:.2f} mm"
            )
        self._matrix = matrix

    @property
    def reprojection_rmse_m(self) -> float:
        return self._reprojection_rmse_m

    @property
    def max_reprojection_error_m(self) -> float:
        return self._max_reprojection_error_m

    def map_pixel_to_robot(self, u: float, v: float) -> tuple[float, float]:
        if not (math.isfinite(u) and math.isfinite(v)):
            raise DetectionError("待映射像素坐标必须为有限数值")
        if cv2.pointPolygonTest(self._pixel_hull, (float(u), float(v)), False) < 0:
            raise DetectionError(
                f"像素点 ({u:.1f}, {v:.1f}) 位于单应性标定点凸包之外，拒绝外推"
            )
        homogeneous = self._matrix @ np.array([u, v, 1.0], dtype=np.float64)
        denominator = float(homogeneous[2])
        if not math.isfinite(denominator) or abs(denominator) < 1e-12:
            raise DetectionError("单应性映射齐次分母无效")
        x = float(homogeneous[0] / denominator)
        y = float(homogeneous[1] / denominator)
        if not (math.isfinite(x) and math.isfinite(y)):
            raise DetectionError("单应性映射得到非有限机器人 XY")
        return x, y
