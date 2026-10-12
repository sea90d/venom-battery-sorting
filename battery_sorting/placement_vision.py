"""Offline, fail-closed detection of the printed T0 and P placement regions.

This module only consumes already-loaded images. A valid result means the
printed geometry is unambiguous; a caller must still map the pixel with
same-frame camera/TCP data before robot use.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence

import cv2
import numpy as np


class TargetKind(str, Enum):
    T0_RECTANGLE = "t0_rectangle"
    P_RING = "p_ring"


@dataclass(frozen=True)
class PixelROI:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class PlacementDetection:
    label: str
    kind: TargetKind
    valid: bool
    center_px: tuple[float, float] | None = None
    outer_points_px: tuple[tuple[float, float], ...] = ()
    inner_points_px: tuple[tuple[float, float], ...] = ()
    long_axis_angle_deg: float | None = None
    confidence: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class _EllipseCandidate:
    center: tuple[float, float]
    axes: tuple[float, float]
    angle_deg: float
    contour: np.ndarray
    score: float

    @property
    def diameter(self) -> float:
        return 0.5 * (self.axes[0] + self.axes[1])


def _invalid(label: str, kind: TargetKind, reason: str) -> PlacementDetection:
    return PlacementDetection(label=label, kind=kind, valid=False, reason=reason)


def _validate_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise ValueError("image 必须是 numpy 数组")
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image 必须是 uint8 BGR 三通道图像")
    if image.shape[0] < 32 or image.shape[1] < 32:
        raise ValueError("image 尺寸过小")


def _crop(
    image: np.ndarray,
    roi: PixelROI | tuple[int, int, int, int] | None,
) -> tuple[np.ndarray, tuple[int, int]]:
    if roi is None:
        return image, (0, 0)
    if not isinstance(roi, PixelROI):
        try:
            roi = PixelROI(*(int(value) for value in roi))
        except (TypeError, ValueError) as exc:
            raise ValueError("ROI 必须是 (x,y,width,height)") from exc
    if roi.x < 0 or roi.y < 0 or roi.width <= 0 or roi.height <= 0:
        raise ValueError("ROI 坐标或尺寸无效")
    if roi.x + roi.width > image.shape[1] or roi.y + roi.height > image.shape[0]:
        raise ValueError("ROI 超出图像边界")
    if roi.width < 32 or roi.height < 32:
        raise ValueError("ROI 尺寸过小")
    return image[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width], (roi.x, roi.y)


def _edge_contours(
    image: np.ndarray, *, thresholds: tuple[int, int] | None = None,
) -> list[np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    median = float(np.median(gray))
    if thresholds is None:
        low = int(max(20.0, 0.45 * median))
        high = int(max(low + 20, min(220.0, 0.90 * median)))
    else:
        low, high = thresholds
    edges = cv2.Canny(gray, low, high)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    contours, _hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return contours


def _touches_boundary(points: np.ndarray, width: int, height: int, margin: int = 3) -> bool:
    flat = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return bool(
        np.any(flat[:, 0] <= margin)
        or np.any(flat[:, 1] <= margin)
        or np.any(flat[:, 0] >= width - 1 - margin)
        or np.any(flat[:, 1] >= height - 1 - margin)
    )


def _shift_points(points: np.ndarray, offset: tuple[int, int]) -> tuple[tuple[float, float], ...]:
    flat = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return tuple((float(x + offset[0]), float(y + offset[1])) for x, y in flat)


def _deduplicate_rectangles(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    unique: list[dict[str, object]] = []
    for item in sorted(candidates, key=lambda value: float(value["score"]), reverse=True):
        cx, cy = item["center"]  # type: ignore[misc]
        long_side, short_side = item["size"]  # type: ignore[misc]
        duplicate = False
        for kept in unique:
            kx, ky = kept["center"]  # type: ignore[misc]
            klong, kshort = kept["size"]  # type: ignore[misc]
            if (
                math.hypot(float(cx) - float(kx), float(cy) - float(ky))
                <= 0.04 * max(float(long_side), float(klong))
                and abs(float(long_side) - float(klong))
                <= 0.10 * max(float(long_side), float(klong))
                and abs(float(short_side) - float(kshort))
                <= 0.10 * max(float(short_side), float(kshort))
            ):
                duplicate = True
                break
        if not duplicate:
            unique.append(item)
    return unique


def detect_t0(
    image: np.ndarray,
    roi: PixelROI | tuple[int, int, int, int] | None = None,
) -> PlacementDetection:
    """Detect one unambiguous 200 x 100 mm T0 rounded rectangle."""

    try:
        _validate_image(image)
        crop, offset = _crop(image, roi)
    except ValueError as exc:
        return _invalid("T0", TargetKind.T0_RECTANGLE, str(exc))

    height, width = crop.shape[:2]
    crop_area = float(width * height)
    candidates: list[dict[str, object]] = []
    for contour in _edge_contours(crop):
        area = abs(float(cv2.contourArea(contour)))
        if area < max(500.0, 0.002 * crop_area) or area > 0.80 * crop_area:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0.0:
            continue
        rect = cv2.minAreaRect(contour)
        side_a, side_b = (float(rect[1][0]), float(rect[1][1]))
        long_side, short_side = max(side_a, side_b), min(side_a, side_b)
        if short_side < 30.0:
            continue
        aspect = long_side / short_side
        if not 1.65 <= aspect <= 2.35:
            continue
        box = cv2.boxPoints(rect)
        if _touches_boundary(box, width, height):
            continue
        box_area = long_side * short_side
        rectangularity = area / box_area if box_area > 0.0 else 0.0
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        solidity = area / hull_area if hull_area > 0.0 else 0.0
        approx = cv2.approxPolyDP(contour, 0.015 * perimeter, True)
        if not (4 <= len(approx) <= 16 and rectangularity >= 0.72 and solidity >= 0.90):
            continue
        aspect_score = max(0.0, 1.0 - abs(aspect - 2.0) / 0.35)
        score = (
            0.50 * aspect_score
            + 0.30 * min(1.0, rectangularity)
            + 0.20 * min(1.0, solidity)
        )
        candidates.append(
            {
                "score": score,
                "center": (float(rect[0][0]), float(rect[0][1])),
                "size": (long_side, short_side),
                "box": box,
                "angle": float(rect[2] if side_a >= side_b else rect[2] + 90.0),
            }
        )

    strong = [
        item
        for item in _deduplicate_rectangles(candidates)
        if float(item["score"]) >= 0.68
    ]
    if not strong:
        return _invalid("T0", TargetKind.T0_RECTANGLE, "未找到完整且比例可信的 T0 外框")
    if len(strong) != 1:
        return _invalid("T0", TargetKind.T0_RECTANGLE, f"发现 {len(strong)} 个 T0 候选，结果有歧义")
    item = strong[0]
    center = item["center"]  # type: ignore[assignment]
    return PlacementDetection(
        label="T0",
        kind=TargetKind.T0_RECTANGLE,
        valid=True,
        center_px=(float(center[0] + offset[0]), float(center[1] + offset[1])),
        outer_points_px=_shift_points(item["box"], offset),  # type: ignore[arg-type]
        long_axis_angle_deg=float(item["angle"]),
        confidence=float(item["score"]),
    )


def _ellipse_candidates(image: np.ndarray) -> list[_EllipseCandidate]:
    height, width = image.shape[:2]
    image_area = float(height * width)
    raw: list[_EllipseCandidate] = []
    # Light grey printed rings can be weaker than the paper's median-derived
    # thresholds, especially along a fold. Extract a second edge set, while
    # retaining all completeness, shape, concentricity and ambiguity checks.
    contours = _edge_contours(image) + _edge_contours(image, thresholds=(40, 120))
    for contour in contours:
        if len(contour) < 12:
            continue
        area = abs(float(cv2.contourArea(contour)))
        if area < max(350.0, 0.0005 * image_area) or area > 0.40 * image_area:
            continue
        try:
            ellipse = cv2.fitEllipse(contour)
        except cv2.error:
            continue
        axis_a, axis_b = float(ellipse[1][0]), float(ellipse[1][1])
        long_axis, short_axis = max(axis_a, axis_b), min(axis_a, axis_b)
        if short_axis < 24.0 or short_axis / long_axis < 0.58:
            continue
        # A fitted near-circle has an arbitrary rotation. Its rotated bounding
        # square can cross the ROI even when the ellipse itself is complete.
        angles=np.linspace(0,2*math.pi,96,endpoint=False)
        local=np.column_stack((np.cos(angles)*axis_a/2,np.sin(angles)*axis_b/2))
        theta=math.radians(ellipse[2])
        rotation=np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]])
        outline=local@rotation.T+ellipse[0]
        if _touches_boundary(outline, width, height):
            continue
        ellipse_area = math.pi * axis_a * axis_b / 4.0
        fill = area / ellipse_area if ellipse_area > 0.0 else 0.0
        perimeter = float(cv2.arcLength(contour, True))
        circularity = (
            4.0 * math.pi * area / (perimeter * perimeter)
            if perimeter > 0.0
            else 0.0
        )
        if not 0.68 <= fill <= 1.25 or circularity < 0.58:
            continue
        axis_score = min(1.0, (short_axis / long_axis) / 0.82)
        score = 0.55 * min(1.0, circularity) + 0.45 * axis_score
        raw.append(
            _EllipseCandidate(
                center=(float(ellipse[0][0]), float(ellipse[0][1])),
                axes=(axis_a, axis_b),
                angle_deg=float(ellipse[2]),
                contour=contour,
                score=score,
            )
        )

    raw.extend(_ring_outer_edges(image, raw))
    unique: list[_EllipseCandidate] = []
    for item in sorted(raw, key=lambda candidate: candidate.score, reverse=True):
        if any(
            math.dist(item.center, kept.center)
            <= max(3.0, 0.025 * max(item.diameter, kept.diameter))
            and abs(item.diameter - kept.diameter)
            <= 0.08 * max(item.diameter, kept.diameter)
            for kept in unique
        ):
            continue
        unique.append(item)
    return unique


def _ring_outer_edges(
    image: np.ndarray, candidates: Sequence[_EllipseCandidate],
) -> list[_EllipseCandidate]:
    """Fit measured outer edges even when a paper crease joins their contour.

    A candidate inner ring only selects an annulus to inspect. An outer ring
    still needs measured edges in at least 95% of angular sectors, a small
    fitting residual, concentricity and a complete in-image outline.
    """
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    rows, columns = np.nonzero(cv2.Canny(gray, 20, 60))
    pixels = np.column_stack((columns, rows)).astype(float)
    recovered = []
    for inner in candidates:
        theta = math.radians(inner.angle_deg)
        rotation = np.array([[math.cos(theta), -math.sin(theta)],
                             [math.sin(theta), math.cos(theta)]])
        local = (pixels - inner.center) @ rotation / (np.array(inner.axes) / 2)
        radii = np.linalg.norm(local, axis=1)
        annulus = (radii >= 1.68) & (radii <= 2.10)
        selected, local, radii = pixels[annulus], local[annulus], radii[annulus]
        sectors = ((np.arctan2(local[:, 1], local[:, 0]) + math.pi) * 72 /
                   (2 * math.pi)).astype(int) % 72
        outline = []
        for sector in range(72):
            indices = np.flatnonzero(sectors == sector)
            if len(indices):
                median = np.median(radii[indices])
                outline.append(selected[indices[np.argmin(abs(radii[indices] - median))]])
        coverage = len(outline) / 72
        if coverage < .95:
            continue
        contour = np.asarray(outline, np.float32).reshape(-1, 1, 2)
        ellipse = cv2.fitEllipse(contour)
        diameter = sum(ellipse[1]) / 2
        if math.dist(ellipse[0], inner.center) > .02 * diameter:
            continue
        angle = math.radians(ellipse[2])
        axes_rotation = np.array([[math.cos(angle), -math.sin(angle)],
                                  [math.sin(angle), math.cos(angle)]])
        normalized = ((contour[:, 0] - ellipse[0]) @ axes_rotation /
                      (np.array(ellipse[1]) / 2))
        residual = float(np.percentile(abs(np.linalg.norm(normalized, axis=1) - 1), 95))
        if residual > .035:
            continue
        angles = np.linspace(0, 2 * math.pi, 96, endpoint=False)
        boundary = (np.column_stack((np.cos(angles), np.sin(angles))) *
                    (np.array(ellipse[1]) / 2)) @ axes_rotation.T + ellipse[0]
        if _touches_boundary(boundary, image.shape[1], image.shape[0]):
            continue
        recovered.append(_EllipseCandidate(
            center=tuple(ellipse[0]), axes=tuple(ellipse[1]), angle_deg=ellipse[2],
            contour=contour, score=.55 * coverage + .45 * (1 - residual / .06),
        ))
    return recovered


def _ring_pairs(
    candidates: Sequence[_EllipseCandidate],
) -> list[tuple[_EllipseCandidate, _EllipseCandidate, float]]:
    pairs: list[tuple[_EllipseCandidate, _EllipseCandidate, float]] = []
    for outer in candidates:
        options: list[tuple[float, _EllipseCandidate]] = []
        for inner in candidates:
            if inner is outer or inner.diameter >= outer.diameter:
                continue
            ratio = inner.diameter / outer.diameter
            offset = math.dist(inner.center, outer.center) / outer.diameter
            if 0.43 <= ratio <= 0.64 and offset <= 0.055:
                penalty = abs(ratio - (80.0 / 150.0)) / 0.11 + offset / 0.055
                options.append((penalty, inner))
        if not options:
            continue
        penalty, inner = min(options, key=lambda item: item[0])
        score = max(
            0.0,
            0.45 * outer.score
            + 0.35 * inner.score
            + 0.20 * (1.0 - min(1.0, penalty / 2.0)),
        )
        pairs.append((outer, inner, score))

    unique: list[tuple[_EllipseCandidate, _EllipseCandidate, float]] = []
    for pair in sorted(pairs, key=lambda item: item[2], reverse=True):
        if any(
            math.dist(pair[0].center, kept[0].center)
            <= 0.10 * max(pair[0].diameter, kept[0].diameter)
            for kept in unique
        ):
            continue
        unique.append(pair)
    return unique


def detect_sequence_targets(
    image: np.ndarray,
    roi: PixelROI | tuple[int, int, int, int] | None = None,
) -> tuple[PlacementDetection, PlacementDetection, PlacementDetection]:
    """Detect exactly three concentric-ring targets and label the map layout."""

    labels = ("P1", "P2", "P3")
    try:
        _validate_image(image)
        crop, offset = _crop(image, roi)
    except ValueError as exc:
        return tuple(_invalid(label, TargetKind.P_RING, str(exc)) for label in labels)  # type: ignore[return-value]

    pairs = [pair for pair in _ring_pairs(_ellipse_candidates(crop)) if pair[2] >= 0.66]
    if len(pairs) != 3:
        reason = f"需要恰好 3 个完整同心环，实际找到 {len(pairs)} 个"
        return tuple(_invalid(label, TargetKind.P_RING, reason) for label in labels)  # type: ignore[return-value]

    diameters = [pair[0].diameter for pair in pairs]
    if min(diameters) / max(diameters) < 0.68:
        reason = "三个 P 外框尺寸不一致"
        return tuple(_invalid(label, TargetKind.P_RING, reason) for label in labels)  # type: ignore[return-value]

    by_y = sorted(pairs, key=lambda pair: pair[0].center[1])
    top = by_y[:2]
    bottom = by_y[2]
    median_diameter = float(np.median(diameters))
    if (
        abs(top[0][0].center[1] - top[1][0].center[1]) > 0.35 * median_diameter
        or bottom[0].center[1]
        <= max(pair[0].center[1] for pair in top) + 0.25 * median_diameter
    ):
        reason = "P 区布局不符合上方 P1/P2、下方 P3"
        return tuple(_invalid(label, TargetKind.P_RING, reason) for label in labels)  # type: ignore[return-value]

    ordered = [*sorted(top, key=lambda pair: pair[0].center[0]), bottom]
    output: list[PlacementDetection] = []
    for label, (outer, inner, score) in zip(labels, ordered):
        outer_box = cv2.boxPoints((outer.center, outer.axes, outer.angle_deg))
        inner_box = cv2.boxPoints((inner.center, inner.axes, inner.angle_deg))
        output.append(
            PlacementDetection(
                label=label,
                kind=TargetKind.P_RING,
                valid=True,
                # Both independently fitted rings measure the same printed
                # center. Average them so a paper fold pulling one contour
                # slightly off-center does not become the placement center.
                center_px=((outer.center[0] + inner.center[0]) / 2 + offset[0],
                           (outer.center[1] + inner.center[1]) / 2 + offset[1]),
                outer_points_px=_shift_points(outer_box, offset),
                inner_points_px=_shift_points(inner_box, offset),
                confidence=score,
            )
        )
    return tuple(output)  # type: ignore[return-value]


def pixel_to_base_on_horizontal_plane(
    pixel: tuple[float, float],
    intrinsics: CameraIntrinsics,
    base_T_camera: np.ndarray,
    plane_z_m: float,
) -> tuple[float, float, float]:
    """Intersect one colour-camera ray with a horizontal base-frame plane."""

    values = (
        *pixel,
        intrinsics.fx,
        intrinsics.fy,
        intrinsics.cx,
        intrinsics.cy,
        plane_z_m,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("像素、内参和纸面 Z 必须是有限数")
    if intrinsics.fx <= 0.0 or intrinsics.fy <= 0.0:
        raise ValueError("相机焦距必须为正")
    transform = np.asarray(base_T_camera, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("base_T_camera 必须是有限的 4x4 矩阵")
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
        raise ValueError("base_T_camera 齐次矩阵末行无效")
    rotation = transform[:3, :3]
    if (
        not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        or np.linalg.det(rotation) <= 0.0
    ):
        raise ValueError("base_T_camera 旋转矩阵无效")
    ray_camera = np.array(
        [
            (pixel[0] - intrinsics.cx) / intrinsics.fx,
            (pixel[1] - intrinsics.cy) / intrinsics.fy,
            1.0,
        ],
        dtype=np.float64,
    )
    origin = transform[:3, 3]
    direction = rotation @ ray_camera
    if abs(float(direction[2])) < 1e-9:
        raise ValueError("像素射线与纸面平行")
    distance = (plane_z_m - float(origin[2])) / float(direction[2])
    if distance <= 0.0:
        raise ValueError("纸面交点位于相机后方")
    point = origin + distance * direction
    if not np.all(np.isfinite(point)):
        raise ValueError("纸面交点不是有限坐标")
    return float(point[0]), float(point[1]), float(point[2])


def annotate_placement(
    image: np.ndarray,
    results: PlacementDetection | Iterable[PlacementDetection],
) -> np.ndarray:
    """Return a diagnostic copy; invalid results never synthesize geometry."""

    _validate_image(image)
    canvas = image.copy()
    items = (results,) if isinstance(results, PlacementDetection) else tuple(results)
    for index, result in enumerate(items):
        colour = (0, 180, 0) if result.valid else (0, 0, 255)
        if result.valid and result.outer_points_px:
            outer = np.rint(result.outer_points_px).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [outer], True, colour, 2, cv2.LINE_AA)
            if result.inner_points_px:
                inner = np.rint(result.inner_points_px).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(canvas, [inner], True, colour, 2, cv2.LINE_AA)
        if result.center_px is not None:
            center = tuple(int(round(value)) for value in result.center_px)
            cv2.drawMarker(canvas, center, colour, cv2.MARKER_CROSS, 18, 2)
            anchor = (center[0] + 8, center[1] - 8)
        else:
            anchor = (12, 24 + 22 * index)
        text = (
            f"{result.label} {result.confidence:.2f}"
            if result.valid
            else f"{result.label} INVALID: {result.reason}"
        )
        cv2.putText(
            canvas,
            text,
            anchor,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            colour,
            1,
            cv2.LINE_AA,
        )
    return canvas
