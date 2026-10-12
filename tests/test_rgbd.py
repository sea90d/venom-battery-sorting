from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from battery_sorting.config import BatterySortingConfig
from battery_sorting.rgbd import (
    DepthPlaneFit,
    analyse_rgbd_frame,
    check_centroid_stability,
    choose_rgbd_consensus,
    create_homography_if_calibrated,
    fit_paper_inverse_depth_plane,
    fuse_depth_frames,
    make_raised_foreground_mask,
)

from .conftest import synthetic_frame


def _inverse_depth_plane(
    width: int, height: int, a: float = 2.0e-5, b: float = -1.0e-5,
    c: float = 1.45,
) -> np.ndarray:
    vv, uu = np.mgrid[0:height, 0:width]
    return (1.0 / (a * uu + b * vv + c)).astype(np.float32)


def _raised_depth(config: BatterySortingConfig, height_m: float = 0.05) -> np.ndarray:
    depth = _inverse_depth_plane(config.camera.width, config.camera.height)
    for roi in config.vision.rois:
        depth[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width] -= height_m
    return depth


def test_inverse_depth_plane_fit_recovers_tilt(valid_config: BatterySortingConfig) -> None:
    depth = _inverse_depth_plane(valid_config.camera.width, valid_config.camera.height)
    fit = fit_paper_inverse_depth_plane(depth, valid_config.vision.rois)
    assert fit.valid, fit.reason
    assert fit.a == pytest.approx(2.0e-5, abs=1e-8)
    assert fit.b == pytest.approx(-1.0e-5, abs=1e-8)
    assert fit.c == pytest.approx(1.45, abs=1e-5)
    assert fit.rmse_m < 1e-6


def test_flat_printed_colours_are_removed(valid_config: BatterySortingConfig) -> None:
    frame = synthetic_frame(valid_config)
    depth = _inverse_depth_plane(valid_config.camera.width, valid_config.camera.height)
    mapping, reason = create_homography_if_calibrated(valid_config.vision.homography)
    result = analyse_rgbd_frame(
        frame, depth, valid_config.vision, mapping=mapping,
        mapping_unavailable_reason=reason,
    )
    assert not result.valid
    assert np.count_nonzero(result.foreground_mask) == 0


def test_raised_batteries_are_retained_and_mapped(
    valid_config: BatterySortingConfig,
) -> None:
    frame = synthetic_frame(valid_config)
    depth = _raised_depth(valid_config)
    mapping, reason = create_homography_if_calibrated(valid_config.vision.homography)
    assert mapping is not None and reason == ""
    result = analyse_rgbd_frame(
        frame, depth, valid_config.vision, mapping=mapping,
        mapping_unavailable_reason=reason,
    )
    assert result.valid, result.error
    assert {slot.color for slot in result.slots} == {"red", "blue", "yellow", "green"}
    assert all(slot.robot_xy_m is not None for slot in result.slots)
    assert all(slot.height_above_paper_m == pytest.approx(0.05, abs=0.002) for slot in result.slots)


def test_same_frame_camera_xyz_is_reported(valid_config: BatterySortingConfig) -> None:
    class FakeCamera:
        @staticmethod
        def deproject_pixel(u: float, v: float, depth_m: float) -> tuple[float, float, float]:
            return (u / 1000.0, v / 1000.0, depth_m)

    mapping, reason = create_homography_if_calibrated(valid_config.vision.homography)
    result = analyse_rgbd_frame(
        synthetic_frame(valid_config),
        _raised_depth(valid_config),
        valid_config.vision,
        camera=FakeCamera(),  # type: ignore[arg-type]
        mapping=mapping,
        mapping_unavailable_reason=reason,
    )
    assert result.valid
    for slot in result.slots:
        assert slot.camera_xyz_m is not None
        assert slot.camera_xyz_m[2] == pytest.approx(slot.object_depth_m)


def test_uncalibrated_mapping_stays_null(valid_config: BatterySortingConfig) -> None:
    frame = synthetic_frame(valid_config)
    depth = _raised_depth(valid_config)
    uncalibrated = replace(valid_config.vision.homography, calibrated=False)
    mapping, reason = create_homography_if_calibrated(uncalibrated)
    result = analyse_rgbd_frame(
        frame, depth, valid_config.vision, mapping=mapping,
        mapping_unavailable_reason=reason,
    )
    assert result.valid
    assert all(slot.robot_xy_m is None for slot in result.slots)
    assert all("未标定" in slot.robot_xy_reason for slot in result.slots)


def test_bad_depth_fails_closed(valid_config: BatterySortingConfig) -> None:
    result = analyse_rgbd_frame(
        synthetic_frame(valid_config),
        np.zeros((valid_config.camera.height, valid_config.camera.width), dtype=np.float32),
        valid_config.vision,
    )
    assert not result.valid
    assert "纸面拟合失败" in result.error


def test_height_threshold_keeps_only_real_protrusions(
    valid_config: BatterySortingConfig,
) -> None:
    paper = 0.70
    depth = np.full(
        (valid_config.camera.height, valid_config.camera.width), paper, dtype=np.float32
    )
    first, second, third, fourth = valid_config.vision.rois
    for roi, height_m in zip(
        (first, second, third, fourth), (0.0, 0.010, 0.050, 0.150)
    ):
        depth[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width] = paper - height_m
    plane = DepthPlaneFit(0.0, 0.0, 1.0 / paper, 1000, 1000, 0.0, True)
    mask = make_raised_foreground_mask(depth, plane, valid_config.vision.rois)
    assert not np.any(mask[first.y : first.y + first.height, first.x : first.x + first.width])
    assert not np.any(mask[second.y : second.y + second.height, second.x : second.x + second.width])
    assert np.all(mask[third.y : third.y + third.height, third.x : third.x + third.width])
    assert not np.any(mask[fourth.y : fourth.y + fourth.height, fourth.x : fourth.x + fourth.width])


def test_rgbd_consensus_requires_strict_count(valid_config: BatterySortingConfig) -> None:
    frame = synthetic_frame(valid_config)
    depth = _raised_depth(valid_config)
    mapping, reason = create_homography_if_calibrated(valid_config.vision.homography)
    good = analyse_rgbd_frame(
        frame, depth, valid_config.vision, mapping=mapping,
        mapping_unavailable_reason=reason,
    )
    bad = analyse_rgbd_frame(
        frame, np.zeros_like(depth), valid_config.vision, mapping=mapping,
        mapping_unavailable_reason=reason,
    )
    chosen, count = choose_rgbd_consensus([good, good, good, bad, bad], 3)
    assert chosen.valid and count == 3
    rejected, count = choose_rgbd_consensus([good, good, bad, bad, bad], 3)
    assert not rejected.valid and count == 2


def test_depth_fusion_fills_complementary_holes() -> None:
    first = np.full((4, 6), 0.61, dtype=np.float32)
    second = first.copy()
    first[:, :3] = 0.0
    second[:, 3:] = 0.0
    fused = fuse_depth_frames([first, second])
    assert fused.dtype == np.float32
    assert np.all(fused == pytest.approx(0.61))


def test_depth_fusion_all_missing_stays_missing() -> None:
    first = np.full((3, 3), 0.61, dtype=np.float32)
    second = first.copy()
    first[1, 1] = 0.0
    second[1, 1] = np.nan
    fused = fuse_depth_frames([first, second])
    assert fused[1, 1] == 0.0
    assert fused[0, 0] == pytest.approx(0.61)


def test_depth_fusion_ignores_out_of_range_samples() -> None:
    first = np.array([[0.05, 4.0, 0.60]], dtype=np.float32)
    second = np.array([[0.61, 0.62, 0.64]], dtype=np.float32)
    fused = fuse_depth_frames([first, second])
    assert fused.tolist()[0] == pytest.approx([0.61, 0.62, 0.62])


def _shift_centroids(result: RGBDFrameResult, dx: float, dy: float) -> RGBDFrameResult:
    detection_slots = tuple(
        replace(
            slot,
            centroid=None if slot.centroid is None else (slot.centroid[0] + dx, slot.centroid[1] + dy),
        )
        for slot in result.detection.slots
    )
    rgbd_slots = tuple(
        replace(
            slot,
            centroid_px=None
            if slot.centroid_px is None
            else (slot.centroid_px[0] + dx, slot.centroid_px[1] + dy),
        )
        for slot in result.slots
    )
    return replace(
        result,
        detection=replace(result.detection, slots=detection_slots),
        slots=rgbd_slots,
    )


def test_centroid_instability_is_rejected(valid_config: BatterySortingConfig) -> None:
    mapping, reason = create_homography_if_calibrated(valid_config.vision.homography)
    base = analyse_rgbd_frame(
        synthetic_frame(valid_config),
        _raised_depth(valid_config),
        valid_config.vision,
        mapping=mapping,
        mapping_unavailable_reason=reason,
    )
    per_frame = [_shift_centroids(base, dx, 0.0) for dx in (-1.0, 0.0, 1.0)]
    fused = _shift_centroids(base, 25.0, 0.0)
    ok, rejection = check_centroid_stability(
        fused, per_frame, min_samples=3, max_deviation_px=10.0
    )
    assert not ok
    assert "融合质心偏离" in rejection
