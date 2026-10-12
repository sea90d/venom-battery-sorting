"""Vision-only acceptance checks for the post-release T0 frame.

The target ROI is supplied by the later T0 setup.  No robot or gripper module
is imported here, so the check can be run offline on recorded RGB-D frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import ColorThreshold
from .rgbd import DepthPlaneFit, _valid_depth_mask


class VisualAcceptanceError(RuntimeError):
    """Invalid target ROI or acceptance input."""


@dataclass(frozen=True)
class PlacementAcceptanceResult:
    passed: bool
    color: str
    roi: tuple[int, int, int, int]
    coverage: float
    component_area_px: float
    centroid_px: tuple[float, float] | None
    object_depth_m: float | None
    height_above_paper_m: float | None
    reason: str = ""


def _mask_for_color(image_bgr: np.ndarray, threshold: ColorThreshold) -> np.ndarray:
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8:
        raise VisualAcceptanceError("验收图像必须是 uint8 BGR 图像")
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for item in threshold.ranges:
        lower = np.array([item.h_min, item.s_min, item.v_min], dtype=np.uint8)
        upper = np.array([item.h_max, item.s_max, item.v_max], dtype=np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
    return mask


def verify_placement(
    image_bgr: np.ndarray,
    target_roi: tuple[int, int, int, int],
    threshold: ColorThreshold,
    *,
    depth_m: np.ndarray | None = None,
    paper_plane: DepthPlaneFit | None = None,
    foreground_mask: np.ndarray | None = None,
    min_coverage: float = 0.08,
    min_component_area_px: float = 250.0,
    min_height_m: float = 0.015,
    max_height_m: float = 0.120,
) -> PlacementAcceptanceResult:
    """Verify one color battery is present and raised inside a T0 ROI.

    A passed result requires a sufficiently large color component.  When a
    depth frame and paper plane are supplied, it also requires the component
    to be physically raised above the paper, preventing printed color blocks
    from being accepted as a placed battery.
    """
    if len(target_roi) != 4:
        raise VisualAcceptanceError("target_roi 必须为 (x,y,width,height)")
    x, y, width, height = (int(value) for value in target_roi)
    image_height, image_width = image_bgr.shape[:2]
    if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > image_width or y + height > image_height:
        raise VisualAcceptanceError("target_roi 超出图像")
    if not (0.0 < min_coverage <= 1.0 and 0.0 < min_height_m < max_height_m):
        raise VisualAcceptanceError("验收阈值无效")
    mask = _mask_for_color(image_bgr, threshold)
    if foreground_mask is not None:
        if foreground_mask.shape != mask.shape:
            raise VisualAcceptanceError("foreground_mask 与图像尺寸不一致")
        mask = cv2.bitwise_and(mask, foreground_mask.astype(np.uint8))
    local = mask[y : y + height, x : x + width]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    local = cv2.morphologyEx(local, cv2.MORPH_OPEN, kernel)
    local = cv2.morphologyEx(local, cv2.MORPH_CLOSE, kernel)
    coverage = float(np.count_nonzero(local)) / float(width * height)
    contours, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return PlacementAcceptanceResult(False, threshold.name, target_roi, coverage, 0.0, None, None, None, "目标区域没有颜色连通域")
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    moments = cv2.moments(contour)
    centroid = None
    if moments["m00"] > 0.0:
        centroid = (x + moments["m10"] / moments["m00"], y + moments["m01"] / moments["m00"])
    reason = ""
    object_depth = None
    height_above = None
    if coverage < min_coverage:
        reason = f"目标颜色覆盖率 {coverage:.3f} < {min_coverage:.3f}"
    elif area < min_component_area_px:
        reason = f"最大颜色连通域 {area:.0f}px < {min_component_area_px:.0f}px"
    if depth_m is not None:
        if depth_m.shape != mask.shape:
            raise VisualAcceptanceError("depth_m 与图像尺寸不一致")
        component = np.zeros_like(local, dtype=np.uint8)
        cv2.drawContours(component, [contour], -1, 255, -1)
        values = depth_m[y : y + height, x : x + width][
            (component > 0) & _valid_depth_mask(depth_m[y : y + height, x : x + width])
        ]
        if len(values) < 20:
            reason = reason or f"目标区域有效深度样本不足: {len(values)} < 20"
        else:
            object_depth = float(np.median(values))
            if paper_plane is not None:
                if not paper_plane.valid or centroid is None:
                    reason = reason or "纸面拟合或目标质心无效"
                else:
                    expected = float(paper_plane.expected_depth(*centroid))
                    height_above = expected - object_depth
                    if not (min_height_m <= height_above <= max_height_m):
                        reason = reason or f"电池凸起高度 {height_above * 1000:.1f}mm 超出范围"
    passed = not reason
    return PlacementAcceptanceResult(
        passed, threshold.name, target_roi, coverage, area, centroid,
        object_depth, height_above, reason,
    )
