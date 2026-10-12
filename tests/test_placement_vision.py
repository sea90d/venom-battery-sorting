from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from battery_sorting.placement_vision import (
    CameraIntrinsics,
    PixelROI,
    annotate_placement,
    detect_sequence_targets,
    detect_t0,
    pixel_to_base_on_horizontal_plane,
)


ROOT = Path(__file__).resolve().parents[1]


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    return image


def rounded_rectangle(
    image: np.ndarray,
    center: tuple[int, int],
    size: tuple[int, int],
    radius: int = 18,
) -> None:
    cx, cy = center
    width, height = size
    x1, y1 = cx - width // 2, cy - height // 2
    x2, y2 = cx + width // 2, cy + height // 2
    cv2.line(image, (x1 + radius, y1), (x2 - radius, y1), (0, 0, 0), 4)
    cv2.line(image, (x1 + radius, y2), (x2 - radius, y2), (0, 0, 0), 4)
    cv2.line(image, (x1, y1 + radius), (x1, y2 - radius), (0, 0, 0), 4)
    cv2.line(image, (x2, y1 + radius), (x2, y2 - radius), (0, 0, 0), 4)
    cv2.ellipse(image, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, (0, 0, 0), 4)
    cv2.ellipse(image, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, (0, 0, 0), 4)
    cv2.ellipse(image, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, (0, 0, 0), 4)
    cv2.ellipse(image, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, (0, 0, 0), 4)


def ring(
    image: np.ndarray,
    center: tuple[int, int],
    outer: tuple[int, int] = (65, 58),
) -> None:
    inner = (round(outer[0] * 80 / 150), round(outer[1] * 80 / 150))
    cv2.ellipse(image, center, outer, 0, 0, 360, (0, 0, 0), 4)
    cv2.ellipse(image, center, inner, 0, 0, 360, (0, 0, 0), 4)


def test_synthetic_t0_and_annotation() -> None:
    image = np.full((360, 640, 3), 255, np.uint8)
    rounded_rectangle(image, (320, 180), (300, 150))
    result = detect_t0(image)
    assert result.valid, result.reason
    assert result.center_px == pytest.approx((320, 180), abs=3)
    assert annotate_placement(image, result).shape == image.shape


def test_t0_roi_returns_global_pixel() -> None:
    image = np.full((480, 800, 3), 255, np.uint8)
    rounded_rectangle(image, (560, 260), (280, 140))
    result = detect_t0(image, PixelROI(390, 150, 350, 230))
    assert result.valid, result.reason
    assert result.center_px == pytest.approx((560, 260), abs=3)


def test_t0_rejects_blank_occluded_and_ambiguous() -> None:
    blank = np.full((360, 700, 3), 255, np.uint8)
    assert not detect_t0(blank).valid

    occluded = blank.copy()
    rounded_rectangle(occluded, (350, 180), (300, 150))
    cv2.rectangle(occluded, (340, 90), (700, 270), (255, 255, 255), -1)
    assert not detect_t0(occluded).valid

    ambiguous = np.full((500, 800, 3), 255, np.uint8)
    rounded_rectangle(ambiguous, (220, 160), (260, 130))
    rounded_rectangle(ambiguous, (570, 340), (260, 130))
    result = detect_t0(ambiguous)
    assert not result.valid
    assert "歧义" in result.reason


def test_sequence_layout_labels_top_left_top_right_then_bottom() -> None:
    image = np.full((600, 800, 3), 255, np.uint8)
    for center in ((240, 180), (570, 190), (500, 430)):
        ring(image, center)
    results = detect_sequence_targets(image)
    assert all(result.valid for result in results), [result.reason for result in results]
    assert [result.label for result in results] == ["P1", "P2", "P3"]
    assert results[0].center_px == pytest.approx((240, 180), abs=3)
    assert results[1].center_px == pytest.approx((570, 190), abs=3)
    assert results[2].center_px == pytest.approx((500, 430), abs=3)


@pytest.mark.parametrize("frame", range(1, 10))
def test_field_sequence_low_contrast_p2_outer_ring(frame) -> None:
    image = read_image(ROOT / "output" / "competition_console" /
                       "20261012_165514" / f"target_view_{frame}.png")
    results = detect_sequence_targets(image)
    assert all(item.valid for item in results), [item.reason for item in results]
    for item, center in zip(results, [(130, 184), (304, 188), (304, 364)]):
        assert item.center_px == pytest.approx(center, abs=2)
        edges = np.roll(item.outer_points_px, -1, axis=0) - item.outer_points_px
        assert np.linalg.norm(edges, axis=1) == pytest.approx([150]*4, abs=9)


def test_field_ring_scan_maps_all_frames_without_robot_motion(tmp_path):
    from battery_sorting.closed_loop import Setup, Runtime
    from battery_sorting.config import Pose
    from battery_sorting.observation import Observation, recorded_camera_model
    from battery_sorting.robot import FakeRobot

    setup = Setup(ROOT / "config" / "closed_loop_sequence.json")
    runs = []
    bundles = sorted((ROOT / "output" / "competition_console" /
                      "20261012_165514").glob("*_targets_*"))
    assert len(bundles) == 9
    for index, bundle in enumerate(bundles):
        obs = Observation.load(bundle)
        camera = recorded_camera_model(obs.metadata)
        robot = FakeRobot(Pose(*obs.metadata["tcp_after"]))
        runtime = Runtime(setup, camera, robot, None, tmp_path / str(index))
        runtime.task = "sequence"
        runtime.frame = lambda label, observation=obs: observation
        runtime.scan_targets("sequence")
        assert set(runtime.targets) == {"P1", "P2", "P3"}
        assert robot.records == []
        runs.append(runtime.targets)
    for label in ("P1", "P2", "P3"):
        centers = np.array([run[label]["center"] for run in runs])
        assert np.max(np.linalg.norm(centers - centers[0], axis=1)) < setup.raw["max_center_spread_m"]


def test_sequence_roi_returns_global_pixels_for_all_targets() -> None:
    image = np.full((800, 1100, 3), 255, np.uint8)
    expected = ((430, 300), (760, 310), (690, 550))
    for center in expected:
        ring(image, center)
    results = detect_sequence_targets(image, PixelROI(300, 180, 600, 500))
    assert all(result.valid for result in results), [result.reason for result in results]
    for result, center in zip(results, expected):
        assert result.center_px == pytest.approx(center, abs=3)


def test_sequence_rejects_missing_inner_occlusion_and_extra_ring() -> None:
    missing = np.full((600, 800, 3), 255, np.uint8)
    ring(missing, (240, 180))
    ring(missing, (570, 190))
    cv2.ellipse(missing, (500, 430), (65, 58), 0, 0, 360, (0, 0, 0), 4)
    assert not any(result.valid for result in detect_sequence_targets(missing))

    cropped = np.full((420, 700, 3), 255, np.uint8)
    for center in ((180, 130), (480, 130), (680, 360)):
        ring(cropped, center)
    assert not any(result.valid for result in detect_sequence_targets(cropped))

    extra = np.full((650, 900, 3), 255, np.uint8)
    for center in ((180, 150), (480, 150), (400, 430), (740, 430)):
        ring(extra, center)
    assert not any(result.valid for result in detect_sequence_targets(extra))


def test_official_bottom_maps_are_detected() -> None:
    basic = read_image(ROOT / "tmp" / "pdfs" / "board-1.png")
    sequence = read_image(ROOT / "tmp" / "pdfs" / "board-2.png")
    # The rendered PDF page is tangent to the top of P1/P2. Add a diagnostic
    # canvas margin so the test represents the required full-target camera view.
    sequence = cv2.copyMakeBorder(
        sequence, 120, 120, 120, 120, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )
    t0 = detect_t0(basic)
    p = detect_sequence_targets(sequence)
    assert t0.valid, t0.reason
    assert t0.center_px == pytest.approx((2398, 661), abs=30)
    assert all(item.valid for item in p), [item.reason for item in p]


def test_historical_t0_roi_is_rejected_because_checkerboard_occludes_it() -> None:
    historical = read_image(
        ROOT / "output" / "hover_validation" / "capture_20260910_retry" / "colour.png"
    )
    result = detect_t0(historical, PixelROI(580, 285, 250, 180))
    assert not result.valid


def test_pixel_ray_intersection_with_horizontal_plane() -> None:
    intrinsics = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    base_T_camera = np.eye(4)
    base_T_camera[2, 3] = -1.0
    point = pixel_to_base_on_horizontal_plane(
        (370.0, 240.0), intrinsics, base_T_camera, 0.0
    )
    assert point == pytest.approx((0.1, 0.0, 0.0))


@pytest.mark.parametrize(
    "bad",
    [np.zeros((10, 10), np.uint8), np.zeros((40, 40, 3), np.float32), "not-image"],
)
def test_invalid_image_is_fail_closed(bad) -> None:
    assert not detect_t0(bad).valid
