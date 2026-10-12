from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytest

from battery_sorting.config import BatterySortingConfig, HomographyConfig
from battery_sorting.vision import (
    Camera,
    CameraError,
    ColorDetector,
    DetectionError,
    HomographyModel,
    _angle_diff_180,
    choose_consensus,
)

from .conftest import SLOT_COLORS, bgr_from_hsv, synthetic_frame


def detector(config: BatterySortingConfig) -> ColorDetector:
    return ColorDetector(config.vision, config.camera.width, config.camera.height)


def test_four_unique_colors(valid_config: BatterySortingConfig, frame: np.ndarray) -> None:
    result = detector(valid_config).detect(frame)
    assert result.valid, result.error
    assert result.color_to_slot() == {
        "red": "slot1", "blue": "slot2", "yellow": "slot3", "green": "slot4"
    }
    assert all(slot.confidence > 0.99 for slot in result.slots)


def test_red_second_hue_range(valid_config: BatterySortingConfig) -> None:
    frame = synthetic_frame(valid_config, hue_overrides={"slot1": 175})
    result = detector(valid_config).detect(frame)
    assert result.valid
    assert result.color_to_slot()["red"] == "slot1"


def test_roi_order_does_not_change_physical_slot_mapping(valid_config: BatterySortingConfig) -> None:
    reordered = replace(valid_config.vision, rois=tuple(reversed(valid_config.vision.rois)))
    result = ColorDetector(reordered, valid_config.camera.width, valid_config.camera.height).detect(
        synthetic_frame(valid_config)
    )
    assert result.valid
    assert result.color_to_slot()["red"] == "slot1"
    assert result.color_to_slot()["green"] == "slot4"


def test_duplicate_color_rejected(valid_config: BatterySortingConfig) -> None:
    colors = dict(SLOT_COLORS)
    colors["slot2"] = "red"
    result = detector(valid_config).detect(synthetic_frame(valid_config, colors))
    assert not result.valid
    assert "重复颜色" in result.error


def test_low_coverage_rejected(valid_config: BatterySortingConfig) -> None:
    frame = synthetic_frame(valid_config)
    roi = next(roi for roi in valid_config.vision.rois if roi.slot_name == "slot1")
    frame[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width] = 0
    frame[roi.y + 50 : roi.y + 70, roi.x + 80 : roi.x + 100] = bgr_from_hsv(0)
    result = detector(valid_config).detect(frame)
    assert not result.valid
    assert "覆盖率" in result.error


def test_centroid_offset_rejected(valid_config: BatterySortingConfig) -> None:
    frame = synthetic_frame(valid_config)
    roi = next(roi for roi in valid_config.vision.rois if roi.slot_name == "slot1")
    frame[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width] = 0
    frame[roi.y : roi.y + 60, roi.x : roi.x + 60] = bgr_from_hsv(0)
    result = detector(valid_config).detect(frame)
    assert not result.valid
    assert "质心偏离" in result.error


def test_color_tie_always_rejected(valid_config: BatterySortingConfig) -> None:
    vision = replace(valid_config.vision, min_confidence=0.50, min_margin=0.01)
    frame = synthetic_frame(valid_config)
    roi = next(roi for roi in vision.rois if roi.slot_name == "slot1")
    frame[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width // 2] = bgr_from_hsv(0)
    frame[roi.y : roi.y + roi.height, roi.x + roi.width // 2 : roi.x + roi.width] = bgr_from_hsv(110)
    result = ColorDetector(vision, valid_config.camera.width, valid_config.camera.height).detect(frame)
    assert not result.valid
    assert "平局" in result.error


def test_frame_shape_mismatch_fails_closed(valid_config: BatterySortingConfig) -> None:
    with pytest.raises(DetectionError, match="输入形状"):
        detector(valid_config).detect(np.zeros((100, 100, 3), dtype=np.uint8))


def test_foreground_mask_filters_printed_colours(
    valid_config: BatterySortingConfig, frame: np.ndarray,
) -> None:
    foreground = np.zeros(
        (valid_config.camera.height, valid_config.camera.width), dtype=np.uint8
    )
    result = detector(valid_config).detect(frame, foreground_mask=foreground)
    assert not result.valid
    assert all(not slot.valid for slot in result.slots)


def test_foreground_mask_preserves_raised_colours(
    valid_config: BatterySortingConfig, frame: np.ndarray,
) -> None:
    foreground = np.full(
        (valid_config.camera.height, valid_config.camera.width), 255, dtype=np.uint8
    )
    result = detector(valid_config).detect(frame, foreground_mask=foreground)
    assert result.valid


def test_foreground_mask_rejects_wrong_shape_and_dtype(
    valid_config: BatterySortingConfig, frame: np.ndarray,
) -> None:
    with pytest.raises(DetectionError, match="形状"):
        detector(valid_config).detect(
            frame, foreground_mask=np.zeros((10, 10), dtype=np.uint8)
        )
    with pytest.raises(DetectionError, match="uint8 或 bool"):
        detector(valid_config).detect(
            frame,
            foreground_mask=np.zeros(
                (valid_config.camera.height, valid_config.camera.width),
                dtype=np.float32,
            ),
        )


def test_multiframe_consensus(valid_config: BatterySortingConfig) -> None:
    good = detector(valid_config).detect(synthetic_frame(valid_config))
    duplicate_colors = dict(SLOT_COLORS)
    duplicate_colors["slot2"] = "red"
    bad = detector(valid_config).detect(synthetic_frame(valid_config, duplicate_colors))
    assert choose_consensus([good, good, good, bad, bad], 3).valid
    assert not choose_consensus([good, good, bad, bad, bad], 3).valid


class _FakeCapture:
    def __init__(
        self,
        width: int,
        height: int,
        *,
        reject_sets: bool = False,
        readback_offset: float = 0.0,
        output_size: tuple[int, int] | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.reject_sets = reject_sets
        self.readback_offset = readback_offset
        self.output_size = output_size or (width, height)
        self.values: dict[int, float] = {}
        self.released = False

    def isOpened(self) -> bool:
        return True

    def set(self, prop: int, value: float) -> bool:
        self.values[prop] = float(value)
        return not self.reject_sets

    def get(self, prop: int) -> float:
        return self.values.get(prop, 0.0) + self.readback_offset

    def read(self) -> tuple[bool, np.ndarray]:
        width, height = self.output_size
        return True, np.zeros((height, width, 3), dtype=np.uint8)

    def release(self) -> None:
        self.released = True


def test_strict_camera_rejects_unsupported_parameter(
    valid_config: BatterySortingConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeCapture(valid_config.camera.width, valid_config.camera.height, reject_sets=True)
    monkeypatch.setattr("battery_sorting.vision.cv2.VideoCapture", lambda _device: fake)
    with pytest.raises(CameraError, match="拒绝设置"):
        Camera(valid_config.camera, strict_parameters=True).open()
    assert fake.released


def test_strict_camera_rejects_bad_readback(
    valid_config: BatterySortingConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeCapture(
        valid_config.camera.width,
        valid_config.camera.height,
        readback_offset=10.0,
    )
    monkeypatch.setattr("battery_sorting.vision.cv2.VideoCapture", lambda _device: fake)
    with pytest.raises(CameraError, match="参数可能未锁定"):
        Camera(valid_config.camera, strict_parameters=True).open()
    assert fake.released


def test_camera_rejects_unexpected_frame_resolution(
    valid_config: BatterySortingConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeCapture(
        valid_config.camera.width,
        valid_config.camera.height,
        output_size=(640, 480),
    )
    monkeypatch.setattr("battery_sorting.vision.cv2.VideoCapture", lambda _device: fake)
    with pytest.raises(CameraError, match="实际输出 640x480"):
        Camera(valid_config.camera).open()
    assert fake.released


def test_largest_component_geometry_is_reported(
    valid_config: BatterySortingConfig, frame: np.ndarray,
) -> None:
    result = detector(valid_config).detect(frame)
    assert result.valid
    for slot in result.slots:
        assert slot.contour_area_px is not None and slot.contour_area_px > 0.0
        assert slot.long_axis_angle_deg == pytest.approx(0.0, abs=0.1)


def test_component_area_outside_calibration_is_rejected(
    valid_config: BatterySortingConfig, frame: np.ndarray,
) -> None:
    rois = tuple(
        replace(roi, expected_component_area_px=1.0, area_tolerance_fraction=0.1)
        if roi.slot_name == "slot1"
        else roi
        for roi in valid_config.vision.rois
    )
    result = ColorDetector(
        replace(valid_config.vision, rois=rois),
        valid_config.camera.width,
        valid_config.camera.height,
    ).detect(frame)
    assert not result.valid
    assert "最大色块面积" in result.error


def test_long_axis_angle_outside_calibration_is_rejected(
    valid_config: BatterySortingConfig, frame: np.ndarray,
) -> None:
    rois = tuple(
        replace(roi, expected_long_axis_angle_deg=45.0, max_angle_error_deg=2.0)
        if roi.slot_name == "slot1"
        else roi
        for roi in valid_config.vision.rois
    )
    result = ColorDetector(
        replace(valid_config.vision, rois=rois),
        valid_config.camera.width,
        valid_config.camera.height,
    ).detect(frame)
    assert not result.valid
    assert "最大色块长轴" in result.error


def test_rotated_long_axis_calibration_passes(valid_config: BatterySortingConfig) -> None:
    frame = synthetic_frame(valid_config)
    roi = next(item for item in valid_config.vision.rois if item.slot_name == "slot1")
    frame[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width] = 0
    local_center = (roi.width / 2.0, roi.height / 2.0)
    box = cv2.boxPoints((local_center, (100.0, 70.0), 20.0)).astype(np.int32)
    cv2.fillConvexPoly(
        frame[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width],
        box,
        bgr_from_hsv(0),
    )
    rois = tuple(
        replace(item, expected_long_axis_angle_deg=20.0, max_angle_error_deg=2.0)
        if item.slot_name == "slot1"
        else item
        for item in valid_config.vision.rois
    )
    result = ColorDetector(
        replace(valid_config.vision, rois=rois),
        valid_config.camera.width,
        valid_config.camera.height,
    ).detect(frame)
    assert result.valid, result.error
    slot1 = next(slot for slot in result.slots if slot.slot_name == "slot1")
    assert slot1.long_axis_angle_deg == pytest.approx(20.0, abs=1.0)


def test_long_axis_angle_has_180_degree_period() -> None:
    assert _angle_diff_180(85.0, -85.0) == pytest.approx(10.0)
    assert _angle_diff_180(-90.0, 89.0) == pytest.approx(1.0)
    assert _angle_diff_180(45.0, -45.0) == pytest.approx(90.0)


def _homography_config() -> HomographyConfig:
    return HomographyConfig(
        pixel_points=(
            (100.0, 100.0),
            (500.0, 100.0),
            (100.0, 380.0),
            (500.0, 380.0),
            (300.0, 240.0),
        ),
        robot_xy_points_m=(
            (0.05, 0.05),
            (0.25, 0.05),
            (0.05, 0.25),
            (0.25, 0.25),
            (0.15, 0.15),
        ),
        calibrated=True,
        max_reprojection_error_m=0.002,
    )


def test_homography_maps_pixel_to_robot_xy() -> None:
    model = HomographyModel(_homography_config())
    x, y = model.map_pixel_to_robot(300.0, 240.0)
    assert x == pytest.approx(0.15, abs=1e-8)
    assert y == pytest.approx(0.15, abs=1e-8)
    assert model.max_reprojection_error_m < 1e-8


def test_homography_rejects_uncalibrated_and_duplicate_points() -> None:
    with pytest.raises(DetectionError, match="calibrated=false"):
        HomographyModel(replace(_homography_config(), calibrated=False))
    duplicate = replace(
        _homography_config(),
        pixel_points=((100.0, 100.0),) * 5,
    )
    with pytest.raises(DetectionError, match="重复"):
        HomographyModel(duplicate)


def test_homography_rejects_collinear_points() -> None:
    collinear = HomographyConfig(
        pixel_points=((0.0, 0.0), (10.0, 10.0), (20.0, 20.0), (30.0, 30.0)),
        robot_xy_points_m=((0.0, 0.0), (0.1, 0.1), (0.2, 0.2), (0.3, 0.3)),
        calibrated=True,
    )
    with pytest.raises(DetectionError, match="共线"):
        HomographyModel(collinear)


def test_homography_rejects_large_reprojection_error() -> None:
    bad_robot = list(_homography_config().robot_xy_points_m)
    bad_robot[-1] = (0.19, 0.12)
    inconsistent = replace(
        _homography_config(),
        robot_xy_points_m=tuple(bad_robot),
        max_reprojection_error_m=0.0001,
    )
    with pytest.raises(DetectionError, match="重投影误差"):
        HomographyModel(inconsistent)


def test_homography_refuses_extrapolation_outside_calibration_hull() -> None:
    model = HomographyModel(_homography_config())
    with pytest.raises(DetectionError, match="凸包之外"):
        model.map_pixel_to_robot(50.0, 50.0)
