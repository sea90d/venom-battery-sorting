"""Tests for battery_sorting.realsense: depth sampling, multi-frame, config, error paths.

These tests use synthesized numpy arrays and mocked pyrealsense2; they never
access real hardware, so they are safe to run in CI.
"""

from __future__ import annotations

import math
import builtins
from dataclasses import replace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from battery_sorting.geometry import CameraIntrinsics, GeometryError
from battery_sorting.realsense import (
    RS_DEPTH_RANGE_M,
    DepthProbe,
    DepthSample,
    RealSenseCamera,
    RealSenseDeviceInfo,
    RealSenseError,
    StreamProfile,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _depth_with_centre(
    width: int = 1280,
    height: int = 720,
    centre_m: float = 1.0,
    *,
    noise: float = 0.0,
    hole_radius: int = 0,
) -> np.ndarray:
    """Create a float32 depth map with a constant value plus optional noise."""
    depth = np.full((height, width), centre_m, dtype=np.float32)
    if noise > 0:
        rng = np.random.default_rng(42)
        depth += rng.normal(0.0, noise, depth.shape).astype(np.float32)
    if hole_radius > 0:
        cy, cx = height // 2, width // 2
        y, x = np.ogrid[:height, :width]
        mask = (x - cx) ** 2 + (y - cy) ** 2 <= hole_radius**2
        depth[mask] = 0.0
    return depth


def _make_intrinsics_property(intr: CameraIntrinsics) -> MagicMock:
    return MagicMock(return_value=intr)


# ── _sample_neighbourhood ────────────────────────────────────────────────────

def test_sample_neighbourhood_uniform() -> None:
    depth = _depth_with_centre(centre_m=1.5)
    median, mad, count = RealSenseCamera._sample_neighbourhood(depth, 640, 360, radius=5)
    assert median == pytest.approx(1.5)
    assert mad == pytest.approx(0.0)
    assert count == 121  # 11x11


def test_sample_neighbourhood_with_noise() -> None:
    depth = _depth_with_centre(centre_m=1.5, noise=0.01)
    median, mad, count = RealSenseCamera._sample_neighbourhood(depth, 640, 360, radius=5)
    assert median == pytest.approx(1.5, abs=0.01)
    assert mad >= 0.0
    assert count == 121


def test_sample_neighbourhood_all_zeros() -> None:
    depth = np.zeros((720, 1280), dtype=np.float32)
    median, mad, count = RealSenseCamera._sample_neighbourhood(depth, 640, 360, radius=5)
    assert median == 0.0
    assert mad == 0.0
    assert count == 0


def test_sample_neighbourhood_hole() -> None:
    depth = _depth_with_centre(centre_m=1.5, hole_radius=2)
    median, mad, count = RealSenseCamera._sample_neighbourhood(depth, 642, 362, radius=5)
    assert count > 0
    assert count < 121  # some pixels are zero
    assert median == pytest.approx(1.5)


def test_sample_neighbourhood_out_of_range() -> None:
    depth = np.full((720, 1280), 0.1, dtype=np.float32)  # below min range
    median, mad, count = RealSenseCamera._sample_neighbourhood(depth, 640, 360, radius=5)
    assert count == 0  # all filtered out


def test_sample_neighbourhood_edge_clamps() -> None:
    depth = _depth_with_centre(centre_m=1.0)
    median, mad, count = RealSenseCamera._sample_neighbourhood(depth, 0, 0, radius=5)
    assert count == 36  # 6x6


# ── DepthSample ──────────────────────────────────────────────────────────────

def test_depth_sample_quality_ok() -> None:
    s = DepthSample(
        depth_m=1.0,
        camera_xyz=(0.0, 0.0, 1.0),
        base_xyz=None,
        valid_count=20,
        mad_m=0.01,
        quality_ok=True,
    )
    assert s.quality_ok


def test_depth_sample_rejects_below_range() -> None:
    with pytest.raises(RealSenseError, match="量程"):
        DepthSample(
            depth_m=0.1,
            camera_xyz=(0.0, 0.0, 0.1),
            base_xyz=None,
            valid_count=20,
            mad_m=0.01,
            quality_ok=True,
        )


def test_depth_sample_rejects_above_range() -> None:
    with pytest.raises(RealSenseError, match="量程"):
        DepthSample(
            depth_m=5.0,
            camera_xyz=(0.0, 0.0, 5.0),
            base_xyz=None,
            valid_count=20,
            mad_m=0.01,
            quality_ok=True,
        )


def test_depth_sample_quality_false_skips_range_check() -> None:
    s = DepthSample(
        depth_m=0.1,
        camera_xyz=(0.0, 0.0, 0.1),
        base_xyz=None,
        valid_count=5,
        mad_m=0.1,
        quality_ok=False,
    )
    assert not s.quality_ok


# ── DepthProbe ───────────────────────────────────────────────────────────────

def test_depth_probe_not_converged() -> None:
    p = DepthProbe(sample=None, frame_count=2, converged=False, median_depth_m=None, mad_m=None)
    assert not p.converged
    assert p.median_depth_m is None


def test_depth_probe_converged() -> None:
    s = DepthSample(
        depth_m=1.0,
        camera_xyz=(0.0, 0.0, 1.0),
        base_xyz=None,
        valid_count=20,
        mad_m=0.01,
        quality_ok=True,
    )
    p = DepthProbe(sample=s, frame_count=5, converged=True, median_depth_m=1.0, mad_m=0.01)
    assert p.converged
    assert p.median_depth_m == pytest.approx(1.0)


# ── list_devices ─────────────────────────────────────────────────────────────

@patch("battery_sorting.realsense.RealSenseCamera._import_rs")
def test_list_devices_empty(mock_import: MagicMock) -> None:
    rs = MagicMock()
    ctx = MagicMock()
    ctx.devices = []
    rs.context.return_value = ctx
    mock_import.return_value = rs
    devices = RealSenseCamera.list_devices()
    assert devices == []


@patch("battery_sorting.realsense.RealSenseCamera._import_rs")
def test_list_devices_one(mock_import: MagicMock) -> None:
    rs = MagicMock()
    dev = MagicMock()
    dev.get_info.side_effect = lambda info: {
        "serial": "1234567890",
        "name": "Intel RealSense D455",
        "usb": "3.2",
    }.get(info, "")

    rs.camera_info.serial_number = "serial"
    rs.camera_info.name = "name"
    rs.camera_info.usb_type_descriptor = "usb"

    ctx = MagicMock()
    ctx.devices = [dev]
    rs.context.return_value = ctx
    mock_import.return_value = rs

    devices = RealSenseCamera.list_devices()
    assert len(devices) == 1
    assert devices[0].serial == "1234567890"


# ── probe_depth (with mocked capture) ────────────────────────────────────────

def _mock_camera() -> RealSenseCamera:
    cam = RealSenseCamera()
    cam._rs = MagicMock()
    cam._pipeline = MagicMock()
    cam._profile = MagicMock()
    cam._align = MagicMock()
    cam._color_intrinsics = CameraIntrinsics(fx=600.0, fy=600.0, cx=640.0, cy=360.0)
    cam._aligned_depth_intrinsics = object()
    cam._camera_to_base = None
    cam._deproject_pixel = MagicMock(  # type: ignore[method-assign]
        side_effect=lambda u, v, depth: (
            (u - 640.0) * depth / 600.0,
            (v - 360.0) * depth / 600.0,
            depth,
        )
    )
    return cam


def test_probe_depth_quality_ok() -> None:
    cam = _mock_camera()
    colour = np.zeros((720, 1280, 3), dtype=np.uint8)
    depth = _depth_with_centre(centre_m=1.5)
    cam.capture = MagicMock(return_value=(colour, depth))  # type: ignore[method-assign]
    sample = cam.probe_depth(640.0, 360.0, radius=5, min_valid=8, max_mad_m=0.05)
    assert sample.quality_ok
    assert sample.depth_m == pytest.approx(1.5)
    assert sample.valid_count >= 8
    assert sample.camera_xyz[2] == pytest.approx(1.5)


def test_probe_depth_low_valid_count() -> None:
    cam = _mock_camera()
    colour = np.zeros((720, 1280, 3), dtype=np.uint8)
    depth = np.zeros((720, 1280), dtype=np.float32)  # all zero
    cam.capture = MagicMock(return_value=(colour, depth))  # type: ignore[method-assign]
    sample = cam.probe_depth(640.0, 360.0, radius=5, min_valid=8, max_mad_m=0.05)
    assert not sample.quality_ok
    assert sample.valid_count == 0


def test_probe_depth_no_depth_frame() -> None:
    cam = _mock_camera()
    colour = np.zeros((720, 1280, 3), dtype=np.uint8)
    cam.capture = MagicMock(return_value=(colour, None))  # type: ignore[method-assign]
    with pytest.raises(RealSenseError, match="无可用的对齐深度图"):
        cam.probe_depth(640.0, 360.0)


def test_probe_depth_no_intrinsics() -> None:
    cam = _mock_camera()
    cam._aligned_depth_intrinsics = None
    colour = np.zeros((720, 1280, 3), dtype=np.uint8)
    depth = _depth_with_centre(centre_m=1.5)
    cam.capture = MagicMock(return_value=(colour, depth))  # type: ignore[method-assign]
    del cam._deproject_pixel
    with pytest.raises(RealSenseError, match="未初始化"):
        cam.probe_depth(640.0, 360.0)


def test_probe_depth_with_extrinsics() -> None:
    cam = _mock_camera()
    cam._camera_to_base = np.eye(4, dtype=np.float64)
    colour = np.zeros((720, 1280, 3), dtype=np.uint8)
    depth = _depth_with_centre(centre_m=1.5)
    cam.capture = MagicMock(return_value=(colour, depth))  # type: ignore[method-assign]
    sample = cam.probe_depth(640.0, 360.0, radius=5, min_valid=8, max_mad_m=0.05)
    assert sample.quality_ok
    assert sample.base_xyz is not None
    assert sample.base_xyz[2] == pytest.approx(1.5)


# ── probe_depth_multi ────────────────────────────────────────────────────────

def test_probe_depth_multi_converges() -> None:
    cam = _mock_camera()
    colour = np.zeros((720, 1280, 3), dtype=np.uint8)
    depth = _depth_with_centre(centre_m=1.5)

    call_count = [0]

    def fake_capture() -> tuple[np.ndarray, np.ndarray]:
        call_count[0] += 1
        return colour, depth

    cam.capture = fake_capture  # type: ignore[method-assign]
    probe = cam.probe_depth_multi(
        640.0, 360.0, frames=5, required_consistent=3, settle_time=0.0,
    )
    assert probe.converged
    assert probe.frame_count == 5
    assert probe.median_depth_m == pytest.approx(1.5)
    assert probe.mad_m == pytest.approx(0.0)
    assert probe.sample is not None


def test_probe_depth_multi_not_converged() -> None:
    cam = _mock_camera()
    colour = np.zeros((720, 1280, 3), dtype=np.uint8)
    depth_empty = np.zeros((720, 1280), dtype=np.float32)

    cam.capture = MagicMock(return_value=(colour, depth_empty))  # type: ignore[method-assign]
    probe = cam.probe_depth_multi(
        640.0, 360.0, frames=5, required_consistent=3, settle_time=0.0,
    )
    assert not probe.converged
    assert probe.frame_count == 0


def test_probe_depth_multi_rejects_high_cross_frame_mad() -> None:
    cam = _mock_camera()
    colour = np.zeros((720, 1280, 3), dtype=np.uint8)
    levels = iter((1.0, 1.1, 1.2, 1.3, 1.4))

    def varying_capture() -> tuple[np.ndarray, np.ndarray]:
        return colour, _depth_with_centre(centre_m=next(levels))

    cam.capture = varying_capture  # type: ignore[method-assign]
    probe = cam.probe_depth_multi(
        640.0,
        360.0,
        frames=5,
        required_consistent=5,
        max_frame_mad_m=0.05,
        settle_time=0.0,
    )
    assert not probe.converged
    assert probe.sample is None
    assert probe.median_depth_m == pytest.approx(1.2)
    assert probe.mad_m == pytest.approx(0.1)


def test_probe_depth_multi_aggregated_sample_is_consistent() -> None:
    cam = _mock_camera()
    colour = np.zeros((720, 1280, 3), dtype=np.uint8)
    levels = iter((1.000, 1.002, 1.004))
    cam.capture = lambda: (  # type: ignore[method-assign]
        colour,
        _depth_with_centre(centre_m=next(levels)),
    )
    probe = cam.probe_depth_multi(
        640.0,
        360.0,
        frames=3,
        required_consistent=3,
        max_frame_mad_m=0.005,
        settle_time=0.0,
    )
    assert probe.converged and probe.sample is not None
    assert probe.sample.depth_m == pytest.approx(probe.median_depth_m)
    assert probe.sample.camera_xyz[2] == pytest.approx(probe.median_depth_m)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"frames": 0}, "frames"),
        ({"frames": 3, "required_consistent": 4}, "required_consistent"),
        ({"radius": 0}, "radius"),
        ({"min_valid": 0}, "min_valid"),
        ({"max_mad_m": float("nan")}, "必须为正"),
        ({"max_frame_mad_m": 0.0}, "必须为正"),
        ({"settle_time": -0.1}, "settle_time"),
    ],
)
def test_probe_depth_multi_rejects_bad_parameters(
    kwargs: dict[str, float | int], message: str
) -> None:
    cam = _mock_camera()
    with pytest.raises(RealSenseError, match=message):
        cam.probe_depth_multi(640.0, 360.0, **kwargs)


# ── set_camera_to_base ───────────────────────────────────────────────────────

def test_set_camera_to_base_valid() -> None:
    cam = _mock_camera()
    m = np.eye(4, dtype=np.float64)
    cam.set_camera_to_base(m)
    assert cam.is_extrinsics_calibrated


def test_set_camera_to_base_invalid() -> None:
    cam = _mock_camera()
    with pytest.raises(GeometryError, match="4x4"):
        cam.set_camera_to_base(np.eye(3))


# ── import error ─────────────────────────────────────────────────────────────

def test_import_error_is_converted_to_clear_realsense_error() -> None:
    real_import = builtins.__import__

    def controlled_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "pyrealsense2":
            raise ImportError("simulated missing wheel")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=controlled_import):
        with pytest.raises(RealSenseError, match="未安装 pyrealsense2"):
            RealSenseCamera._import_rs()


def test_deproject_uses_aligned_native_intrinsics() -> None:
    cam = RealSenseCamera()
    native_intrinsics = object()
    cam._aligned_depth_intrinsics = native_intrinsics
    rs = MagicMock()
    rs.rs2_deproject_pixel_to_point.return_value = [0.1, -0.2, 0.8]
    cam._rs = rs
    assert cam._deproject_pixel(20.0, 30.0, 0.8) == (0.1, -0.2, 0.8)
    rs.rs2_deproject_pixel_to_point.assert_called_once_with(
        native_intrinsics, [20.0, 30.0], 0.8
    )


def test_public_deproject_does_not_capture_a_new_frame() -> None:
    cam = _mock_camera()
    cam._aligned_depth_intrinsics = MagicMock(width=1280, height=720)
    cam.capture = MagicMock()  # type: ignore[method-assign]
    point = cam.deproject_pixel(640.0, 360.0, 1.2)
    assert point == pytest.approx((0.0, 0.0, 1.2))
    cam.capture.assert_not_called()


def test_public_deproject_rejects_missing_intrinsic_size() -> None:
    cam = _mock_camera()
    cam._aligned_depth_intrinsics = object()
    with pytest.raises(RealSenseError, match="width/height"):
        cam.deproject_pixel(20.0, 30.0, 0.8)


def test_normalize_and_project_colour_use_native_intrinsics() -> None:
    cam = RealSenseCamera()
    intr = MagicMock(width=1280, height=720, model="inverse", coeffs=[1, 2, 3, 4, 5])
    cam._color_rs_intrinsics = intr
    rs = MagicMock()
    rs.rs2_deproject_pixel_to_point.return_value = [0.1, -0.2, 1.0]
    rs.rs2_project_point_to_pixel.return_value = [700.0, 230.0]
    cam._rs = rs
    normalized = cam.normalize_color_pixels(np.array([[650.0, 350.0]]))
    assert np.allclose(normalized, [[0.1, -0.2]])
    projected = cam.project_color_points(np.array([[0.1, -0.2, 1.0]]))
    assert np.allclose(projected, [[700.0, 230.0]])
    assert cam.color_distortion == ("inverse", (1.0, 2.0, 3.0, 4.0, 5.0))


@pytest.mark.parametrize(
    "values",
    [
        (0, 480, 30, 848, 480, 30),
        (1280, True, 30, 848, 480, 30),
        (1280, 720, -1, 848, 480, 30),
    ],
)
def test_stream_profile_requires_positive_integers(values: tuple[object, ...]) -> None:
    with pytest.raises(ValueError, match="正整数"):
        StreamProfile(*values)  # type: ignore[arg-type]
